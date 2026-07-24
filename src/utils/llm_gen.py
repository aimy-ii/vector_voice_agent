"""Клиент OpenAI-совместимой LLM для голосового хода.

Отличия от офисного сценария, где ждать не жалко:

* **Короткий бюджет.** Таймауты в секундах, ретрай ровно один и только на
  сетевой сбой. Промах — деградация в заглушку, а не ожидание: в звонке
  пауза дороже плохого ответа.
* **Структурный вывод отдаётся потокенно.** `with_structured_output` с
  Pydantic-классом собирает ответ целиком и отдаёт одним куском — в трубке
  это секунды молчания. Поэтому схема передаётся словарём: тогда LangChain
  ставит `JsonOutputParser`, который в стриминге отдаёт частичные объекты,
  и мы выталкиваем прирост текстового поля в эфир по мере генерации.
* **Прокси включается флагом** `IS_PROXY`, а не наличием переменных.
* **HTTP-клиент на процесс.** Создаётся лениво под текущий event loop и
  переиспользуется: иначе каждый вызов модели — новое TCP/TLS (и SOCKS при
  прокси), а keepalive не работает.
"""

from __future__ import annotations

import asyncio
import atexit
import logging
import time
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Callable

import httpx
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from core.config import settings

log = logging.getLogger(__name__)

#: Один повтор на сетевой сбой. Больше в звонок не помещается.
LLM_RETRY_ATTEMPTS = 2

#: Пауза перед повтором, секунды.
LLM_RETRY_DELAY = 0.25

_llm_semaphore = asyncio.Semaphore(max(1, settings.llm_max_concurrency))

#: Сетевые ошибки, на которых повтор осмыслен.
_RETRYABLE = (
    httpx.ConnectError,
    httpx.ConnectTimeout,
    httpx.NetworkError,
    httpx.PoolTimeout,
    httpx.ReadError,
    httpx.RemoteProtocolError,
)

#: httpx-клиенты по id event loop: чужой цикл на воркере сервера недопустим.
_http_clients: dict[int, httpx.AsyncClient] = {}
#: Кэш ChatOpenAI: (id цикла, id клиента, модель, температура, потолок) → клиент.
_chat_cache: dict[tuple[str, str, str, float, int], ChatOpenAI] = {}


class LLMTurnFailed(RuntimeError):
    """Модель не ответила в бюджет хода — узел уходит в заглушку."""


def response_format_from(model: type[BaseModel], *, name: str) -> dict[str, Any]:
    """Собирает JSON-схему словарём из Pydantic-модели.

    Словарь нужен намеренно: при передаче самого класса LangChain ставит
    парсер, который срабатывает один раз на финальном сообщении, и потокенной
    отдачи не будет. Словарь включает `JsonOutputParser` — он отдаёт частичные
    объекты по мере генерации.

    Args:
        model: Pydantic-модель ожидаемого ответа.
        name: имя схемы для провайдера.

    Returns:
        Словарь вида ``{"name": ..., "schema": ..., "strict": False}``.
    """
    return {
        "name": name,
        "description": (model.__doc__ or name).strip(),
        "schema": model.model_json_schema(),
        "strict": False,
    }


def _client_kwargs() -> dict[str, Any]:
    """Параметры httpx.AsyncClient из настроек.

    Returns:
        Словарь аргументов конструктора клиента.
    """
    kwargs: dict[str, Any] = {
        "timeout": httpx.Timeout(
            connect=settings.llm_connect_timeout,
            read=settings.llm_read_timeout,
            write=settings.llm_read_timeout,
            pool=settings.llm_connect_timeout,
        ),
        "limits": httpx.Limits(max_keepalive_connections=10, max_connections=20),
        "trust_env": False,
    }
    if settings.proxy_url:
        kwargs["proxy"] = settings.proxy_url
    return kwargs


def _http_client_for_loop() -> httpx.AsyncClient:
    """Лениво создаёт или отдаёт общий httpx-клиент для текущего event loop.

    Returns:
        Живой ``httpx.AsyncClient``, привязанный к работающему циклу.
    """
    loop = asyncio.get_running_loop()
    key = id(loop)
    client = _http_clients.get(key)
    if client is None or client.is_closed:
        # Старый транспорт мёртв — ChatOpenAI с ним в кэше больше не годны.
        stale = [cache_key for cache_key in _chat_cache if cache_key[0] == str(key)]
        for cache_key in stale:
            _chat_cache.pop(cache_key, None)
        client = httpx.AsyncClient(**_client_kwargs())
        _http_clients[key] = client
    return client


def _close_http_clients() -> None:
    """Отпускает процессные httpx-клиенты при выключении интерпретатора.

    У ``httpx.AsyncClient`` есть только ``aclose()``; в синхронном atexit
    await недоступен — просто сбрасываем ссылки, процесс всё равно гаснет.
    """
    _http_clients.clear()
    _chat_cache.clear()


atexit.register(_close_http_clients)


@asynccontextmanager
async def get_llm(
    *,
    fast: bool = False,
    temperature: float | None = None,
    max_tokens: int | None = None,
) -> AsyncIterator[ChatOpenAI]:
    """Асинхронный клиент модели с бюджетом хода и прокси по флагу.

    HTTP-клиент общий на процесс (на текущий event loop). Объекты
    ``ChatOpenAI`` кэшируются по модели, температуре, потолку токенов и
    идентификатору живого httpx-клиента.

    Args:
        fast: True — быстрая модель (`LLM_MODEL_FAST`), False — основная.
        temperature: температура; None — значение из настроек.
        max_tokens: потолок токенов ответа; None — из настроек.

    Yields:
        Готовый клиент `ChatOpenAI`.
    """
    model_name = settings.fast_model if fast else settings.llm_model
    temp = settings.llm_temperature if temperature is None else temperature
    tokens = settings.llm_max_tokens if max_tokens is None else max_tokens
    loop_id = id(asyncio.get_running_loop())
    http_client = _http_client_for_loop()
    cache_key = (str(loop_id), str(id(http_client)), model_name, float(temp), int(tokens))
    chat = _chat_cache.get(cache_key)
    if chat is None:
        chat = ChatOpenAI(
            base_url=settings.llm_base_url,
            api_key=settings.llm_api_key or "not-needed",
            model=model_name,
            temperature=temp,
            max_tokens=tokens,
            streaming=True,
            http_async_client=http_client,
            # Без этого LangChain ломает транспорт httpx при SOCKS-прокси:
            # https://github.com/langchain-ai/langchain/issues/11334
            http_socket_options=(),
        )
        _chat_cache[cache_key] = chat
    yield chat


async def astream_structured(
    llm: ChatOpenAI,
    messages: list[Any],
    *,
    schema: dict[str, Any],
    text_field: str | None = None,
    on_delta: Callable[[str], None] | None = None,
    budget: float | None = None,
    purpose: str | None = None,
) -> dict[str, Any]:
    """Делает вызов модели и опционально стримит прирост текста.

    Пока модель генерирует JSON, парсер отдаёт частично собранные объекты.
    Из каждого берётся поле `text_field`, и в `on_delta` уходит только новый
    хвост — так реплика начинает звучать раньше, чем ответ дописан.
    Служебные вызовы (чекер, резолверы) передают ``text_field=None``.

    Args:
        llm: клиент модели из `get_llm`.
        messages: сообщения запроса.
        schema: JSON-схема словарём (см. `response_format_from`).
        text_field: имя поля с текстом реплики; None — без стрима в эфир.
        on_delta: колбэк прироста текста; None — стримить не нужно.
        budget: потолок ожидания в секундах; None — из настроек.
        purpose: назначение вызова для лога (``чекер``, ``город``,
            ``филиал``, ``генератор``); None — не логировать итог.

    Returns:
        Последний собранный объект ответа.

    Raises:
        LLMTurnFailed: модель не ответила в бюджет или ответ пуст.
    """
    limit = settings.turn_budget_seconds if budget is None else budget
    structured = llm.with_structured_output(schema, method="json_schema")
    model_name = getattr(llm, "model_name", None) or getattr(llm, "model", "?")
    started = time.perf_counter()

    async def _run() -> dict[str, Any]:
        sent = 0
        final: dict[str, Any] = {}
        async for partial in structured.astream(messages):
            if not isinstance(partial, dict):
                continue
            final = partial
            if on_delta is None or not text_field:
                continue
            value = partial.get(text_field)
            if isinstance(value, str) and len(value) > sent:
                on_delta(value[sent:])
                sent = len(value)
        return final

    last: Exception | None = None
    try:
        for attempt in range(1, LLM_RETRY_ATTEMPTS + 1):
            try:
                async with _llm_semaphore:
                    result = await asyncio.wait_for(_run(), timeout=limit)
                if result:
                    return result
                last = LLMTurnFailed("Пустой ответ модели")
            except TimeoutError as exc:
                log.warning("Модель не уложилась в бюджет хода %.1f с", limit)
                raise LLMTurnFailed("Бюджет хода исчерпан") from exc
            except _RETRYABLE as exc:
                last = exc
                log.warning("Сетевой сбой вызова модели (попытка %s): %s", attempt, exc)
                if attempt < LLM_RETRY_ATTEMPTS:
                    await asyncio.sleep(LLM_RETRY_DELAY)
            except Exception as exc:  # noqa: BLE001
                log.error("Ошибка вызова модели: %s", exc)
                raise LLMTurnFailed(str(exc)) from exc

        raise LLMTurnFailed(str(last))
    finally:
        if purpose:
            elapsed_ms = int((time.perf_counter() - started) * 1000)
            log.info(
                "[llm|done] %s, модель %s, %s мс",
                purpose,
                model_name,
                elapsed_ms,
            )
