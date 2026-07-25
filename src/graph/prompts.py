"""Сборка запроса к генератору на один ход.

Порядок блоков: стабильное в начало одним куском, меняющееся хвостом —
провайдер кэширует общий префикс. Персона, правила, запрет механики,
естественность и unknown стабильны весь звонок; статика контекста — после
фиксации города и филиала; профиль, факты, шапка и справки меняются каждый
ход. Перечень городов в промпт не попадает никогда.
"""

from __future__ import annotations

import json
import re
from typing import Any, Mapping, Sequence

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

from graph.context import DYN_MISSING
from graph.names import given_name
from script.build import CompiledScript
from script.models import Step
from script.planner import render_step_text

#: Сколько последних реплик отдаём модели.
HISTORY_TURNS = 8

_PLACEHOLDER = re.compile(r"\{([a-z_]+)\}")

#: Запрет озвучивать служебную механику — ровно одно вхождение в персоне.
_NO_MECHANICS = (
    "Не проговаривай вслух внутреннюю механику: ни «уточню детали», ни «это "
    "поможет подобрать», ни упоминаний поиска, базы, справочника, шагов, "
    "полей и системы. Собеседник разговаривает с менеджером, а не с программой."
)

#: Пометка к тексту-образцу шага с флагом verbatim.
_SAMPLE_PREFIX = "Сформулируй в этом ключе, цифры не меняй:"


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


def naturalness_block(*, ask_for_move: bool) -> str:
    """Собирает блок естественности реплики.

    Args:
        ask_for_move: True — требовать заканчивать вопросом или предложением
            (в шапке есть незакрытый вопрос). False — этот пункт не включать.

    Returns:
        Текстовый блок для системного сообщения.
    """
    lines = ["Естественность:"]
    if ask_for_move:
        lines.append(
            "- Реплика заканчивается ходом к собеседнику — вопросом или конкретным "
            "предложением. Исключение — только прощание."
        )
    lines.extend(
        [
            "- Говори в режиме живого телефонного разговора, а не рекламного "
            "объявления. Один твой ответ — максимум 2–3 коротких предложения. "
            "Не вываливай всё сразу: скажи главное, детали — если человек спросит. "
            "Никаких списков через запятую на всю строку, никакой «миссии компании». "
            "Плохо: «В стоимость входит теория, практика с мастером, автомобиль, "
            "автодром, документы, топливо, внутренние экзамены и организация "
            "ГИБДД…». Хорошо: «Обучение под ключ — всё включено, доплат нет. "
            "Рассказать, что входит?».",
            "- Не переспрашивай и не подтверждай переспросом уже отвеченное. "
            "Никаких «ваш город Санкт-Петербург, верно?».",
            "- Задавай ТОЛЬКО те вопросы, что переданы тебе в шаге. Своих вопросов "
            "не придумывай и не забегай вперёд: если в шаге один вопрос — задай "
            "его и остановись, жди ответа. Не добавляй «заодно» следующий "
            "вопрос — его дадут на следующем ходу. Плохо: тебе дали шаг про "
            "имя, а ты спросил имя И город. Хорошо: спросил только то, что в "
            "шаге.",
            "- Если клиент задал побочный вопрос, переспросил или отвлёкся — "
            "отработай это ДО КОНЦА (ответь, повтори, разъясни), а потом "
            "ВЕРНИСЬ к тому вопросу, который ещё не закрыт. Не бросай "
            "незакрытое и не уезжай на новую тему. Плохо: ты спросил про "
            "запись → клиент «повторите» → ты рассказываешь про сроки (новая "
            "тема, вопрос про запись брошен). Хорошо: ты спросил про запись → "
            "клиент «повторите» → ты повторяешь свой вопрос проще → клиент "
            "отвечает → идёшь дальше.",
            "- Незакрытый вопрос не исчезает: пока человек на него не ответил, "
            "он остаётся твоей текущей задачей, даже если между вами был "
            "побочный обмен.",
            "- Одна реплика клиента может закрыть несколько шагов — зафиксируй всё "
            "прозвучавшее в understood, даже если спрашивали не об этом.",
            "- Обращайся по имени и только по имени; отчество не используй никогда.",
            "- Не начинай со второго вступления, если перед тобой уже прозвучала "
            "фраза-заглушка: продолжай с места, без повторного «добрый день».",
            "- НЕ оценивай и НЕ комментируй выбор клиента и его данные. Не хвали "
            "выбор («хороший выбор», «отличный вариант»), не оценивай возраст, "
            "опыт, город, коробку («возраст подходит», «отличная категория»). "
            "Просто прими сказанное и иди дальше. Клиент не спрашивал твоего "
            "одобрения. Плохо: «Механика — хороший выбор!», «Ваш возраст "
            "подходит». Хорошо: молча учесть и задать следующий вопрос.",
            "- Тем более не давай оценок, которые могут быть неверными "
            "(например, что механика проще для новичка — это не так). Не "
            "рассуждай о сложности/лёгкости выбора вообще.",
            "- Не приписывай ценность тому, о чём нет данных в контексте: про район "
            "и прочее без фактов не говори как о хорошем.",
            "- После того как что-то рассказал, мягко проверь, что клиент "
            "нормально это воспринял — живым коротким вопросом СВОИМИ словами, "
            "каждый раз по-разному. Не зачитывай одну и ту же дежурную фразу. "
            "Смысл проверки дан как образец — передай его естественно, а не "
            "дословно. Плохо (канцелярская пластинка): «Как вам в целом такой "
            "подход к обучению?». Хорошо (живо, по-разному): «Такой график "
            "удобен?» / «Звучит нормально?» / «Как вам?» / «Подходит так?».",
        ]
    )
    return "\n".join(lines)


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
        f"перечень городов сюда не входят):\n{text}"
    )


def _describe_step(
    step: Step,
    profile: Mapping[str, str],
    facts: Mapping[str, Any],
    *,
    heading: str,
    attempts: int = 0,
) -> list[str]:
    """Собирает строки описания одного шага для промпта.

    Args:
        step: шаг скрипта.
        profile: профиль для ветвления текста.
        facts: факты хода для подстановки.
        heading: заголовок строки («Шаг»).
        attempts: сколько раз шаг уже брали.

    Returns:
        Список строк описания.
    """
    asked = "уже спрашивали, ответа нет" if attempts > 0 else "новый вопрос"
    lines = [f"{heading}: {step.id} ({step.kind}, {asked}).", f"Задача: {step.goal}"]
    text = render_step_text(step, profile)
    if text:
        filled = fill_facts(text, facts)
        if filled:
            if step.verbatim:
                lines.append(f"{_SAMPLE_PREFIX}\n{filled}")
            else:
                lines.append("Опорная формулировка (можно сказать своими словами):\n" + filled)
    if step.check_question:
        lines.append(
            "Образец смысла проверки (обязателен в конце реплики; сформулируй "
            "своими словами, живо и коротко, не зачитывай дословно):\n"
            f"«{step.check_question}»"
        )
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
            "подводи разговор к завершению."
        )

    lines = [
        "Шапка скрипта на этот ход — текущие незакрытые шаги. Новый вопрос — "
        "только один; спрашивай только то, что в этих шагах, ничего сверх. "
        "Уже спрашивавшиеся (ответа ещё нет) — твоя незакрытая задача: "
        "вернись к ним после побочного обмена. Не затыкай ими каждую реплику."
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
    return "\n".join(lines)


def step_block(
    step: Step,
    profile: Mapping[str, str],
    facts: Mapping[str, Any],
    *,
    next_step: Step | None = None,
    attempts: Mapping[str, int] | None = None,
) -> str:
    """Совместимая обёртка: текущий и следующий как шапка из одного-двух шагов.

    Args:
        step: ведущий шаг.
        profile: профиль.
        facts: факты хода.
        next_step: следующий шаг или None.
        attempts: счётчики попыток.

    Returns:
        Текстовый блок шапки.
    """
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
        f"их; чего здесь нет — того не выдумывай:\n{body}"
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
        "Не начинай со второго вступления и не повторяй её."
    )


def dynamic_status_block(*, status: str, searching_retry: bool = False) -> str:
    """Инструкция генератору по статусу динамики контекста.

    Args:
        status: ``dynamic_status`` из контекста.
        searching_retry: повторный заход при «в поиске» после заглушки.

    Returns:
        Текстовый блок или пустая строка.
    """
    if status == DYN_MISSING:
        return (
            "По нужному факту в данных ничего нет. Тактично скажи, что этого "
            "нет, и веди разговор дальше — не выдумывай."
        )
    if searching_retry:
        return (
            "Поиск по факту ещё не завершён, пауза-заглушка уже звучала. "
            "На контекст по этому вопросу не опирайся — ответь технично и "
            "веди разговор дальше."
        )
    return ""


def aside_block(script: CompiledScript, done: Sequence[str]) -> str:
    """Перечисляет возражения для `aside_id`.

    Справки в перечень не входят — их отдаёт контекстер в динамику контекста.

    Args:
        script: скомпилированный скрипт.
        done: уже отработанные возражения.

    Returns:
        Текстовый блок перечня возражений.
    """
    lines = ["Перечень возражений (для поля aside_id):"]
    for objection_id, item in script.objections.items():
        mark = " — уже отвечали" if objection_id in done else ""
        lines.append(f"- {objection_id}: возражение{mark}")
    lines.append("Ничего похожего в реплике нет — оставь aside_id пустым.")
    return "\n".join(lines)


def unknown_block(script: CompiledScript) -> str:
    """Что делать, когда ответа нет ни в скрипте, ни в справочнике.

    Args:
        script: скомпилированный скрипт.

    Returns:
        Текстовый блок с формулировкой unknown.
    """
    return (
        "Если точного ответа нет — не придумывай. "
        f"Скажи примерно так и веди разговор дальше: «{script.params.unknown}»"
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
    dynamic_status: str = "",
    searching_retry: bool = False,
) -> list[BaseMessage]:
    """Собирает сообщения запроса к генератору.

    Порядок: персона + естественность + unknown → статика контекста →
    профиль → факты хода → шапка → возражения → заглушка → инструкция схемы →
    хвост истории.

    Args:
        script: скомпилированный скрипт.
        steps: шапка шагов; если None — собирается из step/next_step.
        step: ведущий шаг (совместимость).
        profile: собранный профиль.
        facts: факты хода.
        history: история звонка без системных сообщений.
        asides_done: отработанные возражения.
        next_step: следующий шаг (совместимость).
        context_text: документ контекста.
        spoken_filler: фраза-заглушка, уже ушедшая в эфир.
        attempts: счётчики попыток.
        dynamic_status: статус динамики контекста.
        searching_retry: повторный «в поиске» после заглушки.

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

    ask_for_move = bool(head)
    blocks: list[str] = [
        persona_block(script),
        naturalness_block(ask_for_move=ask_for_move),
        unknown_block(script),
    ]
    status_note = dynamic_status_block(status=dynamic_status, searching_retry=searching_retry)
    if status_note:
        blocks.append(status_note)
    if context_text and not searching_retry:
        ctx = context_block(context_text)
        if ctx:
            blocks.append(ctx)
    blocks.append(profile_block(script, profile))

    facts_text = facts_block(facts)
    if facts_text:
        blocks.append(facts_text)

    blocks.append(steps_block(head, profile, facts, attempts=counts))
    blocks.append(aside_block(script, asides_done))
    filler = filler_spoken_block(spoken_filler)
    if filler:
        blocks.append(filler)
    blocks.append(
        "Верни ответ строго по схеме. В поле reply — только то, что звучит "
        "вслух: живой разговорный русский, без списков и канцелярита."
    )

    tail = list(history)[-HISTORY_TURNS:]
    if not tail:
        tail = [HumanMessage(content="(клиент молчит)")]
    return [SystemMessage(content="\n\n".join(blocks)), *tail]
