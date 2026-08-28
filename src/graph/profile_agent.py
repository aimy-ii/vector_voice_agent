"""Агент профиля: что удалось понять из хвоста диалога.

Быстрая модель, короткая схема. Разбор профиля — оптимизация, не обязанность:
ошибка или таймаут → пустой результат, лайв-канал не роняем.
Статусов и походов за данными не делает — только фиксирует результат разговора.
"""

from __future__ import annotations

import logging
from typing import Mapping, Protocol, Sequence

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from graph.names import given_name
from graph.phone import phone_number
from graph.profile_form import REWRITABLE_MARK, field_pairs
from graph.profile_tidy import tidy_value
from script.build import CompiledScript
from utils.llm_gen import astream_structured, get_llm, response_format_from

log = logging.getLogger(__name__)

#: Поля имени — значение прогоняется через ``given_name``.
_NAME_KEYS = frozenset({"caller_name", "student_name"})

#: Поля номера — значение принимается только если в нём есть номер.
_PHONE_KEYS = frozenset({"caller_phone"})

#: Сколько последних сообщений отдаём агенту как хвост диалога.
_HISTORY_TAIL = 8


class ProfileValue(BaseModel):
    """Одно понятое поле профиля."""

    key: str = Field(description="Имя поля строго из переданного перечня.")
    value: str = Field(description="Значение так, как его назвал клиент.")


class ProfileGuess(BaseModel):
    """Что удалось понять из реплики клиента."""

    values: list[ProfileValue] = Field(default_factory=list)


class ProfileAgent(Protocol):
    """Контракт агента профиля."""

    async def guess(
        self,
        reply: str,
        known: Mapping[str, str],
        fields: Sequence[tuple[str, str]],
        history: Sequence[BaseMessage] = (),
    ) -> ProfileGuess:
        """Разбирает хвост диалога: какие поля профиля удалось понять."""


def profile_fields_of(script: CompiledScript) -> list[tuple[str, str]]:
    """Перечень полей профиля: ключ и человеческое название.

    Старые форматы скрипта несут свои поля внутри данных — берём их. Формат
    продаж полей не несёт: перечень объявлен формой в ``graph.profile_form``.

    Args:
        script: скомпилированный скрипт.

    Returns:
        Список пар ``(key, title)``.
    """
    if script.profile_fields:
        return [(key, field.title) for key, field in script.profile_fields.items()]
    return field_pairs()


def _format_history(history: Sequence[BaseMessage], *, limit: int = _HISTORY_TAIL) -> str:
    """Хвост диалога текстом для промпта агента."""
    if not history:
        return "— пусто"
    tail = list(history)[-max(1, limit) :]
    lines: list[str] = []
    for message in tail:
        role = "бот" if message.type == "ai" else "клиент"
        text = str(getattr(message, "content", "") or "").strip()
        if text:
            lines.append(f"{role}: {text}")
    return "\n".join(lines) if lines else "— пусто"


class LlmProfileAgent:
    """Агент профиля на быстрой модели."""

    async def guess(
        self,
        reply: str,
        known: Mapping[str, str],
        fields: Sequence[tuple[str, str]],
        history: Sequence[BaseMessage] = (),
    ) -> ProfileGuess:
        """Один вызов быстрой модели; при ошибке — пустой результат."""
        if not fields or not (reply or "").strip():
            return ProfileGuess()

        field_lines = "\n".join(f"- {key}: {title}" for key, title in fields)
        known_lines: list[str] = []
        for key, _title in fields:
            value = str(known.get(key) or "").strip()
            if value:
                known_lines.append(f"- {key}: {value}")
        known_block = "\n".join(known_lines) if known_lines else "- пока ничего"
        history_block = _format_history(history)

        system = (
            "Ты разбираешь хвост диалога и заполняешь поля профиля.\n"
            "Ключ поля — строго из перечня. Значение — только то, что "
            "прозвучало в диалоге, без домыслов.\n"
            "Уже заполненные поля не перезаписывай и не дублируй.\n"
            f"Исключение — поля с пометкой «{REWRITABLE_MARK}»: если человек "
            "назвал по такому полю другое значение, верни новое.\n"
            "Без вопроса бота короткая реплика клиента («механика») "
            "сама по себе может быть ответом — смотри хвост.\n"
            "В диалоге нет нового про поля из перечня — верни пустой список."
        )
        human = (
            f"Хвост диалога:\n{history_block}\n"
            f"Текущая реплика клиента: {reply}\n"
            f"Перечень полей:\n{field_lines}\n"
            f"Уже известно:\n{known_block}"
        )
        schema = response_format_from(ProfileGuess, name="vector_profile")
        try:
            async with get_llm(fast=True, temperature=0.0) as llm:
                raw = await astream_structured(
                    llm,
                    [SystemMessage(content=system), HumanMessage(content=human)],
                    schema=schema,
                    text_field=None,
                    purpose="профиль",
                )
            return ProfileGuess.model_validate(raw)
        except Exception as exc:  # noqa: BLE001
            log.warning("Агент профиля не ответил: %s", exc)
            return ProfileGuess()


async def guess_profile(
    reply: str,
    *,
    known: Mapping[str, str],
    fields: Sequence[tuple[str, str]],
    history: Sequence[BaseMessage] = (),
    agent: ProfileAgent | None = None,
    rewritable: frozenset[str] = frozenset(),
) -> ProfileGuess:
    """Точка входа агента профиля с валидацией результата.

    Args:
        reply: реплика клиента.
        known: уже заполненный профиль.
        fields: перечень ``(key, title)``.
        history: хвост истории разговора (без одной реплики смысл теряется).
        agent: подмена для офлайн-тестов.
        rewritable: ключи, которые разрешено уточнять. Пустое множество —
            прежнее поведение: заполненное поле не трогаем.

    Returns:
        Угаданные значения: ключи вне перечня и пустые отброшены, заполненные
        не перезаписываются кроме уточняемых, повтор того же значения отброшен,
        имена прогнаны через ``given_name``, номер — через ``phone_number``
        и отброшен, если это не номер, остальные значения приведены к виду
        записи через ``tidy_value``. Ошибка агента наружу не летит —
        пустой результат.
    """
    worker = agent or LlmProfileAgent()
    allowed = {key for key, _title in fields}
    try:
        guess = await worker.guess(reply, known, fields, history)
    except Exception as exc:  # noqa: BLE001
        log.warning("Агент профиля упал: %s", exc)
        return ProfileGuess()

    out: list[ProfileValue] = []
    seen: set[str] = set()
    for item in guess.values:
        key = str(item.key or "").strip()
        value = str(item.value or "").strip()
        if not key or key not in allowed or not value or key in seen:
            continue
        current = str(known.get(key) or "").strip()
        if current and key not in rewritable:
            continue
        if key in _NAME_KEYS:
            value = given_name(value) or value
        if key in _PHONE_KEYS:
            value = phone_number(value)
        else:
            value = tidy_value(key, value)
        if not value or value == current:
            continue
        seen.add(key)
        out.append(ProfileValue(key=key, value=value))
    return ProfileGuess(values=out)
