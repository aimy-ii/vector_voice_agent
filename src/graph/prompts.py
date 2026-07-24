"""Сборка запроса к генератору на один ход.

Порядок блоков: стабильное в начало одним куском, меняющееся хвостом —
провайдер кэширует общий префикс. Статика контекста стабильна весь разговор,
значит идёт впереди. Перечень городов в промпт не попадает никогда.
"""

from __future__ import annotations

import json
import re
from typing import Any, Mapping, Sequence

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

from graph.names import given_name
from script.build import CompiledScript
from script.models import Step
from script.planner import render_step_text

#: Сколько последних реплик отдаём модели.
HISTORY_TURNS = 8

_PLACEHOLDER = re.compile(r"\{([a-z_]+)\}")

#: Запрет озвучивать служебную механику — общий для всех блоков промпта.
_NO_MECHANICS = (
    "Не проговаривай вслух внутреннюю механику: ни «уточню детали», ни «это "
    "поможет подобрать», ни упоминаний поиска, базы, справочника, шагов, "
    "полей и системы. Собеседник разговаривает с менеджером, а не с программой."
)

_NATURALNESS = (
    "Естественность:\n"
    "- Реплика заканчивается ходом к собеседнику — вопросом или конкретным "
    "предложением. Исключение — только прощание.\n"
    "- Не переспрашивай и не подтверждай переспросом уже отвеченное. "
    "Никаких «ваш город Санкт-Петербург, верно?».\n"
    "- Побочные вопросы — норма: ответь и технично вернись к этапу.\n"
    "- Одна реплика клиента может закрыть несколько шагов — зафиксируй всё "
    "прозвучавшее в understood, даже если спрашивали не об этом.\n"
    "- Обращайся по имени и только по имени; отчество не используй никогда.\n"
    "- Не начинай со второго вступления, если перед тобой уже прозвучала "
    "фраза-заглушка: продолжай с места, без повторного «добрый день»."
)


def fill_facts(text: str, facts: Mapping[str, Any]) -> str:
    """Подставляет факты в текст скрипта.

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
        f"Правила:\n{rules}\n{_NO_MECHANICS}"
    )


def profile_block(script: CompiledScript, profile: Mapping[str, str]) -> str:
    """Показывает, что уже известно о собеседнике.

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
        raw = str(profile.get(key, "")).strip()
        value = given_name(raw) if key in {"caller_name", "student_name"} and raw else raw
        if value:
            known.append(f"- {key} ({field.title}, {whose}): {value}")
        else:
            unknown.append(f"- {key} ({field.title}, {whose})")

    parts = ["Что уже известно:"]
    parts.append("\n".join(known) if known else "- пока ничего")
    if unknown:
        parts.append("\nЧего ещё не знаем (переспрашивать известное — ошибка):")
        parts.append("\n".join(unknown))
    parts.append(_NO_MECHANICS)
    return "\n".join(parts)


def context_block(context_text: str) -> str:
    """Подшивает контекст разговора целиком.

    Args:
        context_text: статика и динамика одним документом.

    Returns:
        Текстовый блок или пустая строка.
    """
    text = (context_text or "").strip()
    if not text:
        return ""
    return (
        "Контекст разговора (справочный материал целиком; список филиалов и "
        f"перечень городов сюда не входят):\n{text}\n{_NO_MECHANICS}"
    )


def _describe_step(
    step: Step,
    profile: Mapping[str, str],
    facts: Mapping[str, Any],
    *,
    heading: str,
    attempts: int = 0,
) -> list[str]:
    """Собирает строки описания одного шага для промпта."""
    asked = "уже спрашивали, ответа нет" if attempts > 0 else "новый вопрос"
    lines = [f"{heading}: {step.id} ({step.kind}, {asked}).", f"Задача: {step.goal}"]
    text = render_step_text(step, profile)
    if text:
        filled = fill_facts(text, facts)
        if step.verbatim:
            lines.append(
                "Этот текст произносится дословно и уже отправлен в эфир "
                "отдельно. Не повторяй и не пересказывай его:\n" + filled
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
        lines.append("Шаг собирает поля: " + ", ".join(step.fills))
    return lines


def steps_block(
    steps: Sequence[Step],
    profile: Mapping[str, str],
    facts: Mapping[str, Any],
    *,
    attempts: Mapping[str, int],
) -> str:
    """Описывает шапку: уже заданные и один новый.

    Args:
        steps: шаги шапки.
        profile: профиль.
        facts: факты хода (без перечня городов).
        attempts: счётчики попыток.

    Returns:
        Текстовый блок.
    """
    if not steps:
        return (
            "Все шаги скрипта закрыты. Отвечай на вопросы собеседника и мягко "
            f"подводи разговор к завершению.\n{_NO_MECHANICS}\n{_NATURALNESS}"
        )
    lines = [
        "Шапка скрипта на этот ход. Новый вопрос — только один; уже "
        "спрашивавшиеся отрабатывай, когда уместно по разговору, не затыкай "
        "ими каждую реплику."
    ]
    for step in steps:
        lines.append("")
        lines.extend(
            _describe_step(
                step,
                profile,
                facts,
                heading="Шаг",
                attempts=int(attempts.get(step.id, 0)),
            )
        )
    lines.append("")
    lines.append(_NATURALNESS)
    lines.append(_NO_MECHANICS)
    return "\n".join(lines)


def step_block(
    step: Step,
    profile: Mapping[str, str],
    facts: Mapping[str, Any],
    *,
    next_step: Step | None = None,
    attempts: Mapping[str, int] | None = None,
) -> str:
    """Совместимая обёртка: текущий и следующий как шапка из одного-двух шагов."""
    counts = attempts or {}
    bundle = [step]
    if next_step is not None:
        bundle.append(next_step)
    return steps_block(bundle, profile, facts, attempts=counts)


def facts_block(facts: Mapping[str, Any]) -> str:
    """Выкладывает факты хода без перечня городов и полного списка филиалов.

    Args:
        facts: факты, принесённые из справочника / резолверов.

    Returns:
        Текстовый блок или пустая строка.
    """
    payload = {
        k: v
        for k, v in facts.items()
        if v not in (None, "", [], {}) and k not in {"city_choices", "branches_total"}
    }
    # Полный список филиалов города в промпт не кладём — только отобранные.
    if (
        "branches" in payload
        and isinstance(payload["branches"], list)
        and len(payload["branches"]) > 3
    ):
        payload["branches"] = payload["branches"][:3]
    if not payload:
        return ""
    body = json.dumps(payload, ensure_ascii=False, indent=1)
    return (
        "Факты этого хода, которыми можно оперировать. Называть можно только "
        f"их; чего здесь нет — того не выдумывай:\n{body}\n{_NO_MECHANICS}"
    )


def filler_spoken_block(spoken_filler: str | None) -> str:
    """Сообщает генератору, какая заглушка уже прозвучала.

    Args:
        spoken_filler: фраза, ушедшая в эфир перед генерацией.

    Returns:
        Текстовый блок или пустая строка.
    """
    text = (spoken_filler or "").strip()
    if not text:
        return ""
    return (
        f"Перед тобой в эфир уже ушла фраза: «{text}». "
        "Не начинай со второго вступления и не повторяй её.\n"
        f"{_NO_MECHANICS}"
    )


def aside_block(script: CompiledScript, done: Sequence[str]) -> str:
    """Перечисляет справки и возражения для `aside_id`."""
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
    lines.append(_NO_MECHANICS)
    return "\n".join(lines)


def unknown_block(script: CompiledScript) -> str:
    """Что делать, когда ответа нет ни в скрипте, ни в справочнике."""
    return (
        "Если точного ответа нет — не придумывай. "
        f"Скажи примерно так и веди разговор дальше: «{script.params.unknown}»\n"
        f"{_NO_MECHANICS}"
    )


def build_turn_messages(
    *,
    script: CompiledScript,
    steps: Sequence[Step] | None = None,
    step: Step | None = None,
    profile: Mapping[str, str],
    facts: Mapping[str, Any],
    history: Sequence[BaseMessage],
    asides_done: Sequence[str],
    next_step: Step | None = None,
    context_text: str = "",
    spoken_filler: str | None = None,
    attempts: Mapping[str, int] | None = None,
) -> list[BaseMessage]:
    """Собирает сообщения запроса к генератору.

    Порядок: статика контекста → персона → профиль → факты хода → шапка →
    заглушка → справки → unknown → инструкция схемы → хвост истории.

    Args:
        script: скомпилированный скрипт.
        steps: шапка шагов; если None — собирается из step/next_step.
        step: ведущий шаг (совместимость).
        profile: собранный профиль.
        facts: факты хода.
        history: история звонка без системных сообщений.
        asides_done: отработанные справки и возражения.
        next_step: следующий шаг (совместимость).
        context_text: документ контекста.
        spoken_filler: фраза-заглушка, уже ушедшая в эфир.
        attempts: счётчики попыток.

    Returns:
        Список сообщений: одно системное и хвост истории.
    """
    counts = dict(attempts or {})
    head: list[Step]
    if steps is not None:
        head = list(steps)
    elif step is not None:
        head = [step]
        if next_step is not None:
            head.append(next_step)
    else:
        head = []

    # Стабильное в начало: контекст (статика) + персона.
    blocks: list[str] = []
    ctx = context_block(context_text)
    if ctx:
        blocks.append(ctx)
    blocks.append(persona_block(script))
    blocks.append(profile_block(script, profile))

    facts_text = facts_block(facts)
    if facts_text:
        blocks.append(facts_text)

    blocks.append(steps_block(head, profile, facts, attempts=counts))
    filler = filler_spoken_block(spoken_filler)
    if filler:
        blocks.append(filler)
    blocks.append(aside_block(script, asides_done))
    blocks.append(unknown_block(script))
    blocks.append(
        "Верни ответ строго по схеме. В поле reply — только то, что звучит "
        "вслух: живой разговорный русский, без списков и канцелярита."
    )

    tail = list(history)[-HISTORY_TURNS:]
    if not tail:
        tail = [HumanMessage(content="(клиент молчит)")]
    return [SystemMessage(content="\n\n".join(blocks)), *tail]
