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

#: Запрет озвучивать служебную механику — общий для всех блоков промпта.
_NO_MECHANICS = (
    "Не проговаривай вслух внутреннюю механику: не объясняй, зачем задан "
    "вопрос, не упоминай поиск, базу, справочник, шаги, поля и систему. "
    "Собеседник разговаривает с менеджером, а не с программой."
)


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


def _describe_step(
    step: Step,
    profile: Mapping[str, str],
    facts: Mapping[str, Any],
    *,
    heading: str,
) -> list[str]:
    """Собирает строки описания одного шага для промпта.

    Args:
        step: описание шага.
        profile: собранный профиль.
        facts: факты справочника.
        heading: заголовок блока («закрывается» / «далее»).

    Returns:
        Список строк описания.
    """
    lines = [f"{heading}: {step.id} ({step.kind}).", f"Задача: {step.goal}"]
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
        lines.append("Шаг закрывается значениями полей: " + ", ".join(step.fills))
    return lines


def step_block(
    step: Step,
    profile: Mapping[str, str],
    facts: Mapping[str, Any],
    *,
    next_step: Step | None = None,
) -> str:
    """Описывает текущий и следующий шаги: что закрывается и куда идём.

    Модель видит обе задачи сразу: клиент отвечает на текущий шаг, а реплика
    должна уже вести к следующему — иначе разговор встаёт на подтверждении.

    Args:
        step: шаг, который клиент закрывает этим ответом.
        profile: собранный профиль.
        facts: факты, принесённые из справочника.
        next_step: шаг, который откроется после закрытия текущего.

    Returns:
        Текстовый блок для системного сообщения.
    """
    lines = _describe_step(step, profile, facts, heading="Сейчас закрывается")
    if next_step is not None:
        lines.append("")
        lines.extend(_describe_step(next_step, profile, facts, heading="Дальше идём к"))
        if next_step.verbatim:
            lines.append(
                "Следующий шаг — дословный блок: его произнесёт скрипт сам. "
                "Не зачитывай его и не дублируй вопрос, который уйдёт в эфир "
                "сразу после блока."
            )
    else:
        lines.append("После этого шага скрипт закончен — мягко подводи к завершению.")

    lines.append("")
    lines.append(
        "Поведение на этом ходу:\n"
        "- Клиент ответил на текущий шаг — зафиксируй значения в understood, "
        "поставь step_status=done, не переспрашивай и не подтверждай переспросом. "
        "Никаких «ваш город Санкт-Петербург, верно?». Достаточно короткой "
        "человеческой реакции («Отлично, Питер») или вообще ничего — и сразу дальше.\n"
        "- Реплика обязана заканчиваться ходом к собеседнику: вопросом или "
        "конкретным предложением. Реплика, после которой клиенту нечего "
        "сказать, обрывает разговор. Исключение — прощание в самом конце.\n"
        "- Клиент ответил сразу на несколько вещей — прими всё в understood "
        "и не спрашивай заново ничего из принятого.\n"
        "- Переспрашивай только когда ответ действительно не разобран, и "
        "по-человечески, а не «уточните, пожалуйста, значение поля»."
    )
    lines.append(_NO_MECHANICS)
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
        "Факты, которыми можно оперировать в разговоре. Называть можно только "
        f"их; чего здесь нет — того не выдумывай:\n{body}\n{_NO_MECHANICS}"
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
    lines.append(_NO_MECHANICS)
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
        "Если точного ответа нет — не придумывай. "
        f"Скажи примерно так и веди разговор дальше: «{script.params.unknown}»\n"
        f"{_NO_MECHANICS}"
    )


def build_turn_messages(
    *,
    script: CompiledScript,
    step: Step | None,
    profile: Mapping[str, str],
    facts: Mapping[str, Any],
    history: Sequence[BaseMessage],
    asides_done: Sequence[str],
    next_step: Step | None = None,
) -> list[BaseMessage]:
    """Собирает сообщения запроса к модели.

    Args:
        script: скомпилированный скрипт.
        step: шаг этого хода или None, если скрипт пройден.
        profile: собранный профиль.
        facts: факты из справочника.
        history: история звонка без системных сообщений.
        asides_done: отработанные справки и возражения.
        next_step: шаг после текущего, если текущий закроется этим ответом.

    Returns:
        Список сообщений: одно системное и хвост истории.
    """
    blocks = [persona_block(script), profile_block(script, profile)]
    if step is not None:
        blocks.append(step_block(step, profile, facts, next_step=next_step))
    else:
        blocks.append(
            "Все шаги скрипта закрыты. Отвечай на вопросы собеседника и мягко "
            "подводи разговор к завершению.\n" + _NO_MECHANICS
        )
    facts_text = facts_block(facts)
    if facts_text:
        blocks.append(facts_text)
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
