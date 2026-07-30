"""Агент профиля: что удалось понять из реплики клиента.

Быстрая модель, короткая схема. Разбор профиля — оптимизация, не обязанность:
ошибка или таймаут → пустой результат, лайв-канал не роняем.
"""

from __future__ import annotations

import logging
import re
from typing import Mapping, Protocol, Sequence

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from graph.names import given_name
from script.build import AnyStep, CompiledScript
from script.models import SalesStep
from utils.llm_gen import astream_structured, get_llm, response_format_from

log = logging.getLogger(__name__)

#: Имена полей в требованиях шагов продаж (латиница + подчёркивания).
_FIELD_KEY = re.compile(r"\b([a-z][a-z0-9_]*)\b")

#: Поля имени — значение прогоняется через ``given_name``.
_NAME_KEYS = frozenset({"caller_name", "student_name"})


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
    ) -> ProfileGuess:
        """Разбирает реплику: какие поля профиля удалось понять."""


def profile_fields_of(script: CompiledScript) -> list[tuple[str, str]]:
    """Перечень полей профиля: ключ и человеческое название.

    Args:
        script: скомпилированный скрипт.

    Returns:
        Список пар ``(key, title)``. В формате продаж — имена полей из
        требований шагов; списков полей в коде нет.
    """
    if script.profile_fields:
        return [(key, field.title) for key, field in script.profile_fields.items()]

    keys: list[str] = []
    seen: set[str] = set()
    for step in script.steps.values():
        text = _step_requirements(step)
        for match in _FIELD_KEY.findall(text):
            if match in seen:
                continue
            seen.add(match)
            keys.append(match)
    return [(key, key) for key in keys]


def _step_requirements(step: AnyStep) -> str:
    """Текст требований шага; у старого формата — пусто."""
    if isinstance(step, SalesStep):
        return step.requirements or ""
    return ""


class LlmProfileAgent:
    """Агент профиля на быстрой модели."""

    async def guess(
        self,
        reply: str,
        known: Mapping[str, str],
        fields: Sequence[tuple[str, str]],
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

        system = (
            "Ты разбираешь реплику клиента и заполняешь поля профиля.\n"
            "Ключ поля — строго из перечня. Значение — как назвал клиент, "
            "без домыслов.\n"
            "Уже заполненные поля не перезаписывай и не дублируй.\n"
            "В реплике нет нового про поля из перечня — верни пустой список."
        )
        human = (
            f"Реплика клиента: {reply}\n"
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
    agent: ProfileAgent | None = None,
) -> ProfileGuess:
    """Точка входа агента профиля с валидацией результата.

    Args:
        reply: реплика клиента.
        known: уже заполненный профиль.
        fields: перечень ``(key, title)`` из скрипта.
        agent: подмена для офлайн-тестов.

    Returns:
        Угаданные значения: ключи вне перечня и пустые отброшены,
        заполненные не перезаписываются, имена прогнаны через ``given_name``.
        Ошибка агента наружу не летит — пустой результат.
    """
    worker = agent or LlmProfileAgent()
    allowed = {key for key, _title in fields}
    try:
        guess = await worker.guess(reply, known, fields)
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
        if str(known.get(key) or "").strip():
            continue
        if key in _NAME_KEYS:
            value = given_name(value) or value
        if not value:
            continue
        seen.add(key)
        out.append(ProfileValue(key=key, value=value))
    return ProfileGuess(values=out)
