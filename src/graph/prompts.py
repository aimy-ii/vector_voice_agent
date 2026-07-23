"""Сборка запроса к модели на один ход.

Промпт собирается из скомпилированного скрипта и фактов, уже принесённых из
справочника. Модель ничего не ищет сама: к моменту вызова всё нужное лежит
перед ней. Это и есть механизм, которым бот перестаёт выдумывать цену, адрес
и даты — ходить в справочник или нет, решает шаг, а не модель.

Все функции чистые: на вход данные, на выход строка. Тестируются офлайн.
"""

from __future__ import annotations

import json
import re
from typing import Any, Mapping, Sequence

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

from script.build import CompiledScript
from script.models import Step
from script.planner import render_step_text

#: Сколько последних реплик отдаём модели. Больше не нужно: состояние знает
#: всё остальное, а длинный контекст — это токены и задержка.
HISTORY_TURNS = 8

_PLACEHOLDER = re.compile(r"\{([a-z_]+)\}")


def fill_facts(text: str, facts: Mapping[str, Any]) -> str:
    """Подставляет факты в текст скрипта.

    Неизвестный плейсхолдер не роняет ход: он просто исчезает из фразы.
    В звонке это лучше исключения.

    Args:
        text: текст с плейсхолдерами вида `{price_line}`.
        facts: значения для подстановки.

    Returns:
        Текст с подставленными значениями.
    """

    def _replace(match: re.Match[str]) -> str:
        value = facts.get(match.group(1))
        return "" if value is None else str(value)

    return _PLACEHOLDER.sub(_replace, text).strip()


def persona_block(script: CompiledScript) -> str:
    """Собирает описание роли и правил разговора.

    Args:
        script: скомпилированный скрипт.

    Returns:
        Текстовый блок для системного сообщения.
    """
    persona = script.persona
    rules = "\n".join(f"- {rule}" for rule in script.rules)
    return (
        f"Ты — {persona.agent_name}, {persona.role} «{persona.company}». "
        f"Тон: {persona.tone}.\n"
        f"Ты разговариваешь по телефону с человеком, который позвонил сам.\n\n"
        f"Правила:\n{rules}"
    )


def profile_block(script: CompiledScript, profile: Mapping[str, str]) -> str:
    """Показывает, что уже известно о собеседнике и что ещё предстоит узнать.

    Роли разведены: звонит не всегда тот, кто будет учиться. Мама узнаёт про
    сына семнадцати лет — её имя и его имя это разные значения.

    Args:
        script: скомпилированный скрипт.
        profile: собранный профиль.

    Returns:
        Текстовый блок для системного сообщения.
    """
    known: list[str] = []
    unknown: list[str] = []
    for key, field in script.profile_fields.items():
        whose = "звонящий" if field.role == "caller" else "будущий курсант"
        value = str(profile.get(key, "")).strip()
        if value:
            known.append(f"- {key} ({field.title}, {whose}): {value}")
        else:
            unknown.append(f"- {key} ({field.title}, {whose})")

    parts = ["Что уже известно:"]
    parts.append("\n".join(known) if known else "- пока ничего")
    if unknown:
        parts.append("\nЧего ещё не знаем (переспрашивать известное — ошибка):")
        parts.append("\n".join(unknown))
    return "\n".join(parts)


def step_block(step: Step, profile: Mapping[str, str], facts: Mapping[str, Any]) -> str:
    """Описывает шаг, которым бот занимается на этом ходу.

    Args:
        step: описание шага.
        profile: собранный профиль.
        facts: факты, принесённые из справочника.

    Returns:
        Текстовый блок для системного сообщения.
    """
    lines = [f"Текущий шаг: {step.id} ({step.kind}).", f"Задача шага: {step.goal}"]

    text = render_step_text(step, profile)
    if text:
        filled = fill_facts(text, facts)
        if step.verbatim:
            lines.append(
                "Этот текст произносится дословно и уже отправлен в эфир отдельно. "
                "Не повторяй и не пересказывай его:\n" + filled
            )
        else:
            lines.append("Опорная формулировка (можно сказать своими словами):\n" + filled)

    if step.check_question:
        lines.append(f"Проверочный вопрос в конце: {step.check_question}")
    if step.options:
        lines.append(
            "Предложи выбор из вариантов, а не открытый вопрос: " + ", ".join(step.options)
        )
    if step.fills:
        lines.append("Этот шаг закрывается значениями полей: " + ", ".join(step.fills))
    return "\n".join(lines)


def facts_block(facts: Mapping[str, Any]) -> str:
    """Выкладывает факты справочника, которые можно называть вслух.

    Args:
        facts: факты, принесённые из справочника.

    Returns:
        Текстовый блок или пустая строка, если фактов нет.
    """
    payload = {k: v for k, v in facts.items() if v not in (None, "", [], {})}
    if not payload:
        return ""
    body = json.dumps(payload, ensure_ascii=False, indent=1)
    return (
        "Факты из справочника автошколы. Называть можно только их; чего здесь "
        f"нет — того не выдумывай:\n{body}"
    )


def aside_block(script: CompiledScript, done: Sequence[str]) -> str:
    """Перечисляет справки и возражения, из которых модель выбирает `aside_id`.

    Args:
        script: скомпилированный скрипт.
        done: что уже отработали за звонок.

    Returns:
        Текстовый блок для системного сообщения.
    """
    lines = ["Перечень посторонних вопросов и возражений (для поля aside_id):"]
    for help_id, item in script.helps.items():
        mark = " — уже отвечали" if help_id in done else ""
        lines.append(
            f"- {help_id}: справка о том, что {item.triggers[0] if item.triggers else help_id}{mark}"
        )
    for objection_id, item in script.objections.items():
        mark = " — уже отвечали" if objection_id in done else ""
        lines.append(f"- {objection_id}: возражение{mark}")
    lines.append("Ничего похожего в реплике нет — оставь aside_id пустым.")
    return "\n".join(lines)


def unknown_block(script: CompiledScript) -> str:
    """Объясняет, что делать, когда ответа нет ни в скрипте, ни в справочнике.

    В скриптах есть дыры: выдумывать нельзя, молчать невозможно.

    Args:
        script: скомпилированный скрипт.

    Returns:
        Текстовый блок для системного сообщения.
    """
    return (
        "Если ответа нет ни в фактах, ни в описании шага — не придумывай. "
        f"Скажи примерно так и веди разговор дальше: «{script.params.unknown}»"
    )


def build_turn_messages(
    *,
    script: CompiledScript,
    step: Step | None,
    profile: Mapping[str, str],
    facts: Mapping[str, Any],
    history: Sequence[BaseMessage],
    asides_done: Sequence[str],
) -> list[BaseMessage]:
    """Собирает сообщения запроса к модели.

    Args:
        script: скомпилированный скрипт.
        step: шаг этого хода или None, если скрипт пройден.
        profile: собранный профиль.
        facts: факты из справочника.
        history: история звонка без системных сообщений.
        asides_done: отработанные справки и возражения.

    Returns:
        Список сообщений: одно системное и хвост истории.
    """
    blocks = [persona_block(script), profile_block(script, profile)]
    if step is not None:
        blocks.append(step_block(step, profile, facts))
    else:
        blocks.append(
            "Все шаги скрипта закрыты. Отвечай на вопросы собеседника и мягко "
            "подводи разговор к завершению."
        )
    facts_text = facts_block(facts)
    if facts_text:
        blocks.append(facts_text)
    blocks.append(aside_block(script, asides_done))
    blocks.append(unknown_block(script))
    blocks.append(
        "Верни ответ строго по схеме. В поле reply — только то, что звучит "
        "вслух: одна-две фразы разговорного русского."
    )

    tail = list(history)[-HISTORY_TURNS:]
    if not tail:
        tail = [HumanMessage(content="(клиент молчит)")]
    return [SystemMessage(content="\n\n".join(blocks)), *tail]
