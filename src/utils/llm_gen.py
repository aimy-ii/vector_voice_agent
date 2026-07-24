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
"""

from __future__ import annotations

import asyncio
import logging
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


@asynccontextmanager
async def get_llm(
    *,
    fast: bool = False,
    temperature: float | None = None,
) -> AsyncIterator[ChatOpenAI]:
    """Асинхронный клиент модели с бюджетом хода и прокси по флагу.

    Args:
        fast: True — быстрая модель (`LLM_MODEL_FAST`), False — основная.
        temperature: температура; None — значение из настроек.

    Yields:
        Готовый клиент `ChatOpenAI`.
    """
    client_kwargs: dict[str, Any] = {
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
        client_kwargs["proxy"] = settings.proxy_url

    async with httpx.AsyncClient(**client_kwargs) as http_client:
        yield ChatOpenAI(
            base_url=settings.llm_base_url,
            api_key=settings.llm_api_key or "not-needed",
            model=settings.fast_model if fast else settings.llm_model,
            temperature=settings.llm_temperature if temperature is None else temperature,
            max_tokens=settings.llm_max_tokens,
            streaming=True,
            http_async_client=http_client,
            # Без этого LangChain ломает транспорт httpx при SOCKS-прокси:
            # https://github.com/langchain-ai/langchain/issues/11334
            http_socket_options=(),
        )


async def astream_structured(
    llm: ChatOpenAI,
    messages: list[Any],
    *,
    schema: dict[str, Any],
    text_field: str | None = None,
    on_delta: Callable[[str], None] | None = None,
    budget: float | None = None,
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

    Returns:
        Последний собранный объект ответа.

    Raises:
        LLMTurnFailed: модель не ответила в бюджет или ответ пуст.
    """
    limit = settings.turn_budget_seconds if budget is None else budget
    structured = llm.with_structured_output(schema, method="json_schema")

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
