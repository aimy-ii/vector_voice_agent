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

from core.config import settings
from graph.context import DYN_MISSING, DYN_READY
from graph.names import given_name
from script.build import AnyStep, CompiledScript
from script.models import SalesStep
from script.planner import render_step_text

#: Исторический лимит хвоста полной сборки; в ``build_turn_messages``
#: больше не применяется — история уходит целиком. Короткие сборки
#: берут свой лимит из настроек.
HISTORY_TURNS = 8

_PLACEHOLDER = re.compile(r"\{([a-z_]+)\}")

#: Общие правила речи — одинаковы для любого сценария, живут в коде.
SPEECH_RULES: tuple[str, ...] = (
    "К клиенту обращение только на «Вы», всегда, без исключений — независимо от его возраста, тона и от того, как он обращается сам. «Расскажи», «тебе», «твой» в адрес клиента — грубая ошибка.",
    "Цифры, адреса и названия берутся из переданных данных как есть: не округляются, не пересчитываются и не воспроизводятся по памяти.",
    "Если в данных есть точное число, адрес, срок или цена — называть их. Заменять на «несколько», «около», «разные» нельзя. Плохо: «В Санкт-Петербурге у нас есть несколько филиалов.» Хорошо: «В Санкт-Петербурге одиннадцать филиалов.»",
    "Если для ответа не хватает того, что человек ещё не называл, — попросить прямо: «чтобы назвать стоимость, подскажите, в каком городе будете учиться». Выдумывать нельзя.",
    "Тон разговорный, не рекламный. Без восклицаний, призывов и оборотов с сайта. Факт из данных взять можно — тон с сайта нет.",
    "Одна тема за реплику. Два-три коротких предложения — потолок длины. Не вываливать всё сразу: детали — если человек спросит.",
    "Реплика всегда заканчивается обращением к человеку — вопросом по делу или предложением выбрать. Заканчивать ничем нельзя. Пустые проверки («продолжим?», «что скажете?», «вас устраивает?») и согласие с содержанием — запрещены: на них отвечают «да», а разговор не двигается. Исключение — реплика ожидания, пока данные готовятся: ход к человеку не нужен.",
    "Вопрос звучит так, как спросил бы человек в разговоре. Служебные обороты «подскажите, пожалуйста», «по такому-то вопросу определились» — не встречаются. Короткий прямой вопрос: «В каком городе будете учиться?», «Механика или автомат?»",
    "Реплика не начинается с «Отлично», «Прекрасно», «Понятно», «Замечательно» и не пересказывает сказанное клиентом. Начало — сразу с сути.",
    "Не повторять дословно то, что уже говорил в этом разговоре — ни свои фразы, ни образцы.",
    "Не произносить служебные слова: шаг, пункт, этап, скрипт, перечень, список, порядок разговора. Клиент про них не знает и знать не должен. Нумерованных перечислений в речи нет.",
    "Не обещать сходить за данными и вернуться. Если данных нет — сказать об этом прямо и вести разговор дальше.",
    "На отвеченное переспроса нет. Если в одной реплике человек ответил сразу на несколько вещей — принять всё и не спрашивать заново.",
    "Утверждать наличие филиала и называть улицу, район, метро или ориентир "
    "можно только если эта строка дословно есть в контексте хода. "
    "Нет — не называть адресов. Подтверждать запись на филиал вне контекста "
    "запрещено. "
    "Плохо: «В районе Просвещения есть филиалы около Энгельса и Художников. "
    "Филиал на Энгельса записала.» "
    "Хорошо: «Сейчас уточню, какие филиалы ближе к Просвещения, и назову адреса.»",
    "Обращение по имени и только по имени, без отчества. Имя — редко: при знакомстве и в ключевых местах; подряд две реплики с именем не идут.",
    "Если реплика клиента бессвязна, оборвана или не отвечает на вопрос — переспросить коротко, не додумывать смысл.",
    "Не здороваться повторно, если разговор уже идёт. Не прощаться, пока человек сам не прощается словами: «до свидания», «мне пора», «я перезвоню».",
    "Язык только русский. Общие вопросы, на которые ответит любой человек, — коротко, с возвратом к теме.",
)

#: Запрет озвучивать служебную механику — ровно одно вхождение в персоне.
_NO_MECHANICS = (
    "Вслух внутренняя механика не проговаривается: ни упоминаний поиска, "
    "базы, справочника, полей и системы. Собеседник разговаривает с "
    "менеджером, а не с программой."
)

#: Пометка к тексту-образцу шага с флагом verbatim.
_SAMPLE_PREFIX = "Сформулировать в этом ключе, цифры не менять:"


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


def _gender_speech_rule(gender: str) -> str:
    """Собирает правило о роде речи из настройки ``agent_gender``.

    Args:
        gender: ``female`` или ``male``.

    Returns:
        Строка правила для блока персоны.
    """
    if gender == "male":
        return "О себе — в мужском роде: «понял», «посмотрел», «уточнил». Женский род — ошибка."
    return "О себе — в женском роде: «поняла», «посмотрела», «уточнила». Мужской род — ошибка."


def naturalness_block(*, ask_for_move: bool, pending_only: bool = False) -> str:
    """Собирает блок естественности реплики.

    Args:
        ask_for_move: True — требовать заканчивать вопросом или предложением
            (в шапке есть незакрытый вопрос). False — этот пункт не включать.
        pending_only: True — ход к собеседнику только по уже висящему вопросу,
            без новой темы; False — обычная формулировка пункта.

    Returns:
        Текстовый блок для системного сообщения.
    """
    lines = ["Естественность:"]
    if ask_for_move:
        if pending_only:
            lines.append(
                "- Реплика заканчивается обращением к человеку по уже висящему "
                "вопросу — переформулировать проще или помочь ответить, "
                "а не открывать новую тему. Исключение — только прощание."
            )
        else:
            lines.append(
                "- Реплика заканчивается обращением к человеку — вопросом или "
                "конкретным предложением. Исключение — только прощание."
            )
    lines.extend(
        [
            "- Живой телефонный разговор, не рекламное объявление. Два-три "
            "коротких предложения — потолок. Списков через запятую на всю "
            "строку и «миссии компании» нет. "
            "Плохо: «В стоимость входит теория, практика, автомобиль, автодром, "
            "документы, топливо, экзамены и организация ГИБДД…». "
            "Хорошо: «Обучение под ключ — всё включено, доплат нет. "
            "Рассказать, что входит?».",
            "- Не переспрашивать уже отвеченное. Никаких «ваш город Санкт-Петербург, верно?».",
            "- Если человек задал вопрос, переспросил или отвлёкся — ответить "
            "до конца, потом мягко вернуться к делу. Не бросать незакрытое "
            "и не уезжать на новую тему. Плохо: спросили про запись → "
            "«повторите» → рассказ про сроки. Хорошо: вопрос про запись "
            "повторяется проще → человек отвечает → дальше.",
            "- Одна реплика клиента может закрыть несколько тем — принять всё прозвучавшее.",
            "- По имени — редко, не подряд в двух репликах. Постоянное "
            "обращение по имени звучит как колл-центр.",
            "- Не начинать со второго вступления: без повторного «добрый день».",
            "- Не оценивать выбор клиента и его данные. Не хвалить "
            "(«хороший выбор»), не оценивать возраст или коробку "
            "(«возраст подходит»). Принять сказанное и идти дальше. "
            "Плохо: «Механика — хороший выбор!». Хорошо: молча учесть и "
            "задать следующий вопрос. Не рассуждать о сложности выбора.",
            "- Не приписывать ценность тому, о чём нет данных в контексте.",
            "- Пустая проверка в конце запрещена. Одинаковый конец не "
            "повторяется: формулировка вопроса каждый раз своими словами. "
            "Плохо: «Обучение под ключ, доплат не будет. Продолжу?». "
            "Хорошо: «Обучение под ключ, доплат не будет. Теорию удобнее "
            "очно или в приложении?».",
            "- Реплика добавляет то, чего человек ещё не знает. Повторять "
            "его слова другими словами — не ответ. Плохо: «На механике учат "
            "на машинах с ручной коробкой.» Хорошо: «На механике — Лада "
            "Гранта и Фольксваген Поло.»",
            "- Данные есть в контексте — произнести их сейчас, целиком, "
            "а не обещать назвать в следующей реплике. Плохо: «В районе "
            "Просвещения два филиала, назову их.» Хорошо: «В районе "
            "Просвещения два филиала: Комендантская площадь и Коломяжский "
            "проспект. Какой удобнее?»",
            "- То, что человек только что сказал, не повторять — ни дословно, "
            "ни как подтверждение выбора («вы выбрали», «поняла, вы будете»). "
            "Сразу по делу. Плохо: «Вы выбрали механику. Теперь расскажу про "
            "сроки.» Хорошо: «Срок обучения — два с половиной месяца.»",
        ]
    )
    return "\n".join(lines)


def persona_block() -> str:
    """Собирает описание роли и правил речи из настроек агента.

    Персона одна на агента и от сценария звонка не зависит. Правила речи
    и запрет механики склеиваются здесь; скрипт в блок не передаётся.

    Returns:
        Текстовый блок для системного сообщения.
    """
    rules = (*SPEECH_RULES, _gender_speech_rule(settings.agent_gender))
    rules_text = "\n".join(f"- {rule}" for rule in rules)
    return (
        f"Роль: {settings.agent_name}, {settings.agent_role} "
        f"«{settings.agent_company}». "
        f"Тон: {settings.agent_tone}.\n"
        "Это телефонный разговор с человеком, который позвонил сам.\n\n"
        f"Правила:\n{rules_text}\n{_NO_MECHANICS}"
    )


def profile_block(
    script: CompiledScript,
    profile: Mapping[str, str],
    *,
    pending_fields: Sequence[str] = (),
) -> str:
    """Показывает форму профиля в трёх состояниях.

    Секции: что известно, что уточняется прямо сейчас, чего ещё не знаем.
    Поле из ``pending_fields`` попадает во вторую секцию, даже если значение
    пустое, и не дублируется в третьей. В формате продаж перечень полей не
    объявлен — секции «чего не знаем» нет, а «уточняется» будет.

    Args:
        script: скомпилированный скрипт.
        profile: собранный профиль.
        pending_fields: поля, которые лайв-канал сейчас разбирает.

    Returns:
        Текстовый блок для системного сообщения.
    """
    pending = {str(key).strip() for key in pending_fields if str(key).strip()}
    known: list[str] = []
    clarifying: list[str] = []
    unknown: list[str] = []

    def _label(key: str, field: Any = None) -> str:
        if field is None:
            return key
        whose = "звонящий" if field.role == "caller" else "будущий курсант"
        return f"{key} ({field.title}, {whose})"

    if script.is_sales:
        for key, raw_value in sorted(profile.items()):
            raw = str(raw_value).strip()
            if key in pending:
                value = given_name(raw) if key in {"caller_name", "student_name"} and raw else raw
                line = f"- {key}: {value}" if value else f"- {key}"
                clarifying.append(line)
                continue
            if not raw:
                continue
            value = given_name(raw) if key in {"caller_name", "student_name"} else raw
            known.append(f"- {key}: {value}")
        for key in sorted(pending):
            if any(line.startswith(f"- {key}") for line in clarifying):
                continue
            clarifying.append(f"- {key}")
        parts = ["Что уже известно:"]
        parts.append("\n".join(known) if known else "- пока ничего")
        if clarifying:
            parts.append(
                "\nЧто уточняется прямо сейчас (поле разбирается — "
                "переспрашивать не надо, скоро будет известно):"
            )
            parts.append("\n".join(clarifying))
        return "\n".join(parts)

    for key, field in script.profile_fields.items():
        raw = str(profile.get(key, "")).strip()
        value = given_name(raw) if key in {"caller_name", "student_name"} and raw else raw
        label = _label(key, field)
        if key in pending:
            line = f"- {label}: {value}" if value else f"- {label}"
            clarifying.append(line)
        elif value:
            known.append(f"- {label}: {value}")
        else:
            unknown.append(f"- {label}")

    for key in sorted(pending):
        if key in script.profile_fields:
            continue
        if any(line.startswith(f"- {key}") for line in clarifying):
            continue
        clarifying.append(f"- {key}")

    parts = ["Что уже известно:"]
    parts.append("\n".join(known) if known else "- пока ничего")
    if clarifying:
        parts.append(
            "\nЧто уточняется прямо сейчас (поле разбирается — "
            "переспрашивать не надо, скоро будет известно):"
        )
        parts.append("\n".join(clarifying))
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


#: Пометка к образцам формулировок шага продаж.
_EXAMPLES_PREFIX = (
    "Образцы формулировок:\n"
    "Это примеры того, как ту же мысль формулируют другие менеджеры, "
    "а не текст для произнесения. Свою фразу строишь сам, от разговора. "
    "Дословное или почти дословное совпадение с образцом — ошибка."
)

#: Жёсткий запрет называть факты вне переданных данных — одна формулировка
#: для основной сборки и коротких (silence / filler / waiting).
_HARD_FACT_BAN = (
    "Жёсткий запрет: не называть цифр, цен, сроков, адресов, дат и "
    "названий, которых нет в переданных данных."
)


def _context_has_fact(context_text: str, facts: Mapping[str, Any], fact: str) -> bool:
    """Есть ли факт в контексте или фактах хода (грубый поиск по подстроке)."""
    needle = fact.strip().lower()
    if not needle:
        return True
    haystack = (context_text or "").lower()
    if needle in haystack:
        return True
    # Короткие ключевые слова из названия факта.
    tokens = [t for t in needle.replace(":", " ").split() if len(t) >= 4]
    if tokens and all(t in haystack for t in tokens[:2]):
        return True
    body = json.dumps(facts, ensure_ascii=False).lower() if facts else ""
    return bool(tokens) and all(t in body for t in tokens[:2])


def _missing_knowledge_line(
    step: SalesStep,
    *,
    context_text: str,
    facts: Mapping[str, Any],
) -> str | None:
    """Строка о нехватке данных из ``knowledge``, если их нет в контексте."""
    if not step.knowledge:
        return None
    missing = [
        fact
        for fact in step.knowledge
        if fact.strip() and not _context_has_fact(context_text, facts, fact)
    ]
    if not missing:
        return None
    joined = "; ".join(missing)
    return (
        f"В контексте нет данных: {joined}. "
        "Вести разговор без чисел и конкретных величин, не выдумывать."
    )


def _describe_sales_step(
    step: SalesStep,
    *,
    heading: str,
    context_text: str = "",
    facts: Mapping[str, Any] | None = None,
) -> list[str]:
    """Собирает строки описания шага продаж: название, требования, образцы.

    Args:
        step: шаг продаж.
        heading: заголовок строки («В перечне» / «Также в перечне»).
        context_text: документ контекста (для проверки ``knowledge``).
        facts: факты хода.

    Returns:
        Список строк описания.
    """
    lines = [
        f"{heading}: {step.id}.",
        f"Название: {step.name}",
        f"Требования:\n{step.requirements}",
    ]
    if step.examples:
        lines.append(_EXAMPLES_PREFIX)
        lines.extend(step.examples)
    gap = _missing_knowledge_line(step, context_text=context_text, facts=facts or {})
    if gap:
        lines.append(gap)
    return lines


def _describe_step(
    step: AnyStep,
    profile: Mapping[str, str],
    facts: Mapping[str, Any],
    *,
    heading: str,
    context_text: str = "",
) -> list[str]:
    """Собирает строки описания одного шага для промпта.

    Args:
        step: шаг скрипта.
        profile: профиль для ветвления текста.
        facts: факты хода для подстановки.
        heading: заголовок строки («В перечне» / «Также в перечне»).
        context_text: документ контекста (для проверки ``knowledge``).

    Returns:
        Список строк описания.
    """
    if isinstance(step, SalesStep):
        return _describe_sales_step(
            step,
            heading=heading,
            context_text=context_text,
            facts=facts,
        )

    lines = [f"{heading}: {step.id} ({step.kind}).", f"Задача: {step.goal}"]
    if step.why:
        lines.append(f"Зачем: {step.why}")
    text = render_step_text(step, profile)
    if text:
        filled = fill_facts(text, facts)
        if filled:
            if step.verbatim:
                lines.append(f"{_SAMPLE_PREFIX}\n{filled}")
            else:
                lines.append("Опорная формулировка (можно сказать своими словами):\n" + filled)
    if step.examples:
        lines.append(_EXAMPLES_PREFIX)
        lines.extend(step.examples)
    if step.avoid:
        lines.append(f"На этом шаге нельзя: {step.avoid}")
    if step.check_question:
        lines.append(
            "Образец смысла проверки (обязателен в конце реплики; сформулировать "
            "своими словами, живо и коротко, не зачитывать дословно):\n"
            f"«{step.check_question}»"
        )
    if step.options:
        lines.append(
            "Предложить выбор из вариантов, а не открытый вопрос: " + ", ".join(step.options)
        )
    if step.fills:
        lines.append("Шаг собирает поля: " + ", ".join(step.fills))
    return lines


def _describe_next_step_for_question(
    step: AnyStep,
    profile: Mapping[str, str],
    facts: Mapping[str, Any],
) -> list[str]:
    """Описывает следующий шаг как ориентир дальнейшего хода разговора.

    Args:
        step: следующий незакрытый шаг после первого в перечне.
        profile: профиль для ветвления текста.
        facts: факты хода для подстановки.

    Returns:
        Список строк: ориентир, без приказа рассказать содержание сейчас.
    """
    lines = [
        "Дальше по обычному ходу разговора — ориентир, куда двигаться, а не текст на сейчас:",
        f"Дальше: {step.id}.",
    ]
    if isinstance(step, SalesStep):
        lines.append(f"Название: {step.name}")
        lines.append(f"Ориентир:\n{step.requirements}")
        if step.examples:
            lines.append(_EXAMPLES_PREFIX)
            lines.extend(step.examples)
        return lines

    lines.append(f"Задача (ориентир): {step.goal}")
    if step.check_question:
        lines.append(
            f"Образец смысла вопроса (сформулировать своими словами):\n«{step.check_question}»"
        )
    text = render_step_text(step, profile)
    if text:
        filled = fill_facts(text, facts)
        if filled:
            lines.append("Опорная формулировка вопроса (можно своими словами):\n" + filled)
    if step.examples:
        lines.append(_EXAMPLES_PREFIX)
        lines.extend(step.examples)
    if step.options:
        lines.append(
            "Предложить выбор из вариантов, а не открытый вопрос: " + ", ".join(step.options)
        )
    return lines


def steps_block(
    steps: Sequence[AnyStep],
    profile: Mapping[str, str],
    facts: Mapping[str, Any],
    *,
    context_text: str = "",
    next_step: AnyStep | None = None,
) -> str:
    """Описывает перечень тем: что ещё выяснить и как обычно об этом говорят.

    Перечень — компас, куда двигать разговор, а не диктовка реплики.
    Порядок предпочтительный, не жёсткий. Следующий шаг (``next_step``)
    в перечень не входит: это ориентир дальнейшего хода.

    Args:
        steps: шаги перечня; порядок — подсказка, не приказ.
        profile: профиль.
        facts: факты хода (без перечня городов).
        context_text: документ контекста для проверки нехватки знаний.
        next_step: первый незакрытый шаг после перечня по порядку скрипта;
            ориентир, не предмет обязательного рассказа.

    Returns:
        Текстовый блок.
    """
    if not steps:
        return (
            "Все шаги скрипта закрыты. Отвечать на вопросы собеседника и мягко "
            "подводить разговор к завершению."
        )

    intro = (
        "В истории — весь разговор целиком. Ниже — что ещё не выяснено и о чём "
        "не рассказано, и то, как мы обычно об этом говорим.\n"
        "Реплика строится из всего этого разом: из того, что человек только что "
        "сказал, из хода разговора и из того, что осталось выяснить.\n"
        "Сказать нужно то, что уместно сейчас. Человек спросил — ответить на "
        "его вопрос. Ответил — принять ответ и двигаться дальше. Разговор ушёл "
        "в сторону — поддержать и мягко вернуть к делу. Видно, что важное "
        "упущено — сказать об этом самому.\n"
        "Порядок в перечне — подсказка, куда двигаться дальше, а не приказ. "
        "Не обязательно брать первый пункт, если сейчас в тему другое. "
        "Разговор всё равно идёт к записи на обучение, а не растекается."
    )
    lines = [intro]
    head_ids = {step.id for step in steps}
    for index, step in enumerate(steps):
        lines.append("")
        lines.extend(
            _describe_step(
                step,
                profile,
                facts,
                heading="В перечне" if index == 0 else "Также в перечне",
                context_text=context_text,
            )
        )
    if next_step is not None and next_step.id not in head_ids:
        lines.append("")
        lines.extend(_describe_next_step_for_question(next_step, profile, facts))
    lines.append("")
    lines.append(_HARD_FACT_BAN)
    return "\n".join(lines)


def step_block(
    step: AnyStep,
    profile: Mapping[str, str],
    facts: Mapping[str, Any],
    *,
    next_step: AnyStep | None = None,
    context_text: str = "",
) -> str:
    """Совместимая обёртка: шаг перечня и отдельно следующий как ориентир.

    Args:
        step: первый шаг перечня.
        profile: профиль.
        facts: факты хода.
        next_step: следующий шаг или None; ориентир дальнейшего хода.
        context_text: документ контекста.

    Returns:
        Текстовый блок перечня.
    """
    return steps_block(
        [step],
        profile,
        facts,
        context_text=context_text,
        next_step=next_step,
    )


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
        f"их; чего здесь нет — того не выдумывать:\n{body}"
    )


def dynamic_status_block(*, status: str, searching_retry: bool = False) -> str:
    """Инструкция генератору по статусу динамики контекста.

    Ветка ``DYN_SEARCHING`` сюда не входит — её обслуживает
    ``build_waiting_messages``.

    Args:
        status: ``dynamic_status`` из контекста.
        searching_retry: устаревший флаг; игнорируется.

    Returns:
        Текстовый блок или пустая строка.
    """
    del searching_retry
    if status == DYN_MISSING:
        return (
            "По нужному факту в данных ничего нет. Тактично сказать, что этого "
            "нет, и вести разговор дальше — не выдумывать."
        )
    if status == DYN_READY:
        return ""
    return ""


def continuation_block(*, turn_kind: str) -> str:
    """Правило для хода-продолжения собственной речи бота.

    Args:
        turn_kind: ``client`` или ``continuation``.

    Returns:
        Текстовый блок или пустая строка.
    """
    if turn_kind != "continuation":
        return ""
    return (
        "Это продолжение вашей собственной речи: реплики клиента не было. "
        "Продолжать с того места, где остановились, ответом по существу того, "
        "что обещали уточнить. Не здороваться, не начинать с подтверждающего "
        "слова, не переспрашивать сказанное человеком. "
        "Прозвучало: «Сейчас уточню, какие филиалы рядом с Просвещения.» "
        "Плохо: «Понятно. Итак, по филиалам…» "
        "Хорошо: «Ближайшие — Комендантская площадь и Коломяжский проспект. "
        "Какой удобнее?»"
    )


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
    lines.append("Ничего похожего в реплике нет — оставить aside_id пустым.")
    return "\n".join(lines)


def closed_steps_block(closed_steps: Sequence[AnyStep]) -> str:
    """Запрет пересказывать уже закрытые шаги.

    Args:
        closed_steps: шаги, закрытые к этому ходу.

    Returns:
        Текстовый блок или пустая строка.
    """
    if not closed_steps:
        return ""
    names: list[str] = []
    for step in closed_steps:
        name = getattr(step, "name", None) or getattr(step, "goal", None) or step.id
        names.append(str(name).strip() or step.id)
    listed = "; ".join(names)
    return (
        "Уже закрытые шаги (к их содержанию не возвращаться):\n"
        f"{listed}.\n"
        "То, что по ним уже сказано, повторять и пересказывать другими словами "
        "нельзя — человек это слышал. Ссылаться на сказанное коротко можно, "
        "разворачивать заново нельзя. "
        "Плохо: шаг про состав курса закрыт, а следующая реплика снова "
        "перечисляет, что входит в стоимость. "
        "Хорошо: коротко сослаться и идти дальше по текущему шагу."
    )


def unknown_block(script: CompiledScript) -> str:
    """Что делать, когда ответа нет ни в скрипте, ни в справочнике.

    Args:
        script: скомпилированный скрипт.

    Returns:
        Текстовый блок с формулировкой unknown.
    """
    return (
        "Если точного ответа нет — не придумывать. "
        f"Сказать примерно так и вести разговор дальше: «{script.params.unknown}». "
        "Улицу, район, метро или ориентир филиала называть можно только если "
        "эта строка дословно есть в контексте хода. Нет — не называть адресов "
        "вообще и сказать, что сейчас уточнишь, какие филиалы рядом. "
        "Подтверждать запись на филиал, которого нет в контексте, запрещено. "
        "Плохо: «В районе Просвещения есть филиалы около Энгельса и Художников. "
        "Филиал на Энгельса записала.» "
        "Хорошо: «Сейчас уточню, какие филиалы ближе к Просвещения, и назову адреса.»"
    )


def build_filler_messages(
    script: CompiledScript,
    *,
    messages: Sequence[BaseMessage],
    history_limit: int,
) -> list[BaseMessage]:
    """Собирает запрос на короткую живую реакцию, пока ответа ещё нет.

    Реплика без содержания: показать, что услышал, и взять паузу.
    Промпт предельно короткий — эта реакция должна родиться быстрее всего.

    Args:
        script: скомпилированный скрипт (нужна только персона).
        messages: история разговора.
        history_limit: сколько последних сообщений оставить.

    Returns:
        Список сообщений: одно системное и короткий хвост истории.
    """
    _ = script
    system = (
        f"{settings.agent_name}, к клиенту на «Вы», живой тон. "
        "Отреагируй на то, что человек только что сказал, как человек, "
        "который думает вслух: зацепись за услышанное и возьми паузу. "
        "Два-пять слов. "
        "Не сообщай данных, цифр и адресов; не задавай вопросов; "
        "не начинай новую тему; не здоровайся; не обещай конкретных сроков. "
        f"{_HARD_FACT_BAN}"
    )
    limit = max(1, int(history_limit))
    tail = list(messages)[-limit:]
    if not tail:
        tail = [HumanMessage(content="(клиент молчит)")]
    return [SystemMessage(content=system), *tail]


def build_silence_messages(
    script: CompiledScript,
    *,
    messages: Sequence[BaseMessage],
    profile: Mapping[str, str],
    step: AnyStep | None,
    attempt: int,
    history_limit: int,
) -> list[BaseMessage]:
    """Собирает запрос к генератору, когда человек молчит.

    Реплики от клиента не было. Задача — мягко вернуть его в разговор,
    опираясь на то, о чём только что говорили. Молчание не согласие
    и не отказ: из него нельзя делать выводов и ничего решать за человека.

    Args:
        script: скомпилированный скрипт.
        messages: история разговора.
        profile: форма разговора.
        step: ведущий шаг, на котором остановились.
        attempt: какая это попытка по счёту.
        history_limit: сколько последних сообщений оставить.

    Returns:
        Список сообщений: одно системное и хвост истории.
    """
    _ = script
    _ = profile
    lines = [
        f"{settings.agent_name}, к клиенту на «Вы», живой тон. Одна-две коротких фразы.",
        "Человек молчит несколько секунд после твоей реплики. "
        "Реплика строится по тому, о чём только что говорили: "
        "предложи пояснить сказанное, спроси, что интересует, "
        "что осталось непонятным.",
        "Молчание не означает ни согласия, ни отказа. Из него нельзя "
        "делать выводов, нельзя ничего решать за человека, бронировать, "
        "записывать, закреплять и объявлять договорённости. "
        "Реплика по молчанию только возвращает человека в разговор.",
    ]
    if int(attempt) <= 1:
        lines.append(
            "Первая попытка: мягко верни к теме разговора — к тому, о чём говорили только что."
        )
    else:
        lines.append(
            "Повторный заход: зайди с другой стороны, не повторяй форму "
            "предыдущей попытки — она видна в хвосте диалога. "
            "Сформулировать иначе — повторять то же самое нельзя."
        )
    lines.append(
        "Запрещены дежурные оклики: «алло», «вы тут?», «вы меня слышите» "
        "и любые замечания о том, что человек молчит."
    )
    lines.append(
        "Не начинать заново, не здороваться, не пересказывать уже сказанное, "
        "не сыпать новыми фактами — сейчас задача вернуть внимание, "
        "а не рассказать больше."
    )
    lines.append(_HARD_FACT_BAN)
    if step is not None:
        name = getattr(step, "name", None) or getattr(step, "goal", None) or step.id
        lines.append(f"Ведущий шаг: {step.id} — {name}.")

    limit = max(1, int(history_limit))
    tail = list(messages)[-limit:]
    if not tail:
        tail = [HumanMessage(content="(клиент молчит)")]
    return [SystemMessage(content="\n".join(lines)), *tail]


def build_waiting_messages(
    script: CompiledScript,
    *,
    messages: Sequence[BaseMessage],
    profile: Mapping[str, str],
    pending_fields: Sequence[str],
    step: AnyStep | None,
    history_limit: int,
    turn_kind: str = "client",
    context_text: str = "",
) -> list[BaseMessage]:
    """Собирает укороченный запрос к генератору для реплики ожидания.

    Реплика короткая и нужна быстро, поэтому в промпт не идут факты
    и шапка шагов: только правила речи, уже добытый контекст, ведущий
    шаг одной строкой, что уточняется, и хвост диалога.

    Args:
        script: скомпилированный скрипт.
        messages: история разговора.
        profile: профиль клиента.
        pending_fields: поля, которые сейчас разбираются.
        step: ведущий шаг, чтобы не терять нить.
        history_limit: сколько последних сообщений оставить.
        turn_kind: ``client`` или ``continuation``.
        context_text: уже добытый контекст (статика и динамика).

    Returns:
        Список сообщений: одно системное и обрезанный хвост истории.
    """
    lines = [
        f"Роль: {settings.agent_name}, {settings.agent_role} «{settings.agent_company}».",
        "К клиенту только на «Вы». Тон живой, без канцелярита.",
        "Реплика ожидания — одно короткое предложение, не длиннее восьми слов.",
        "Никаких объяснений, зачем уточняешь, и никаких хвостов вроде "
        "«чтобы было удобно», «чтобы всё было понятно», «максимально точно» — "
        "они канцелярские и в живой речи не встречаются. "
        "Запрещены придаточные цели и обороты про удобство клиента.",
        "Задача: сказать своими словами, что сейчас готовишь, и удержать "
        "разговор. Предмет назвать конкретно: из хвоста диалога и из "
        "текущего шага пойми, что именно готовится, и скажи это своими "
        "словами — лучше «сейчас подберу …», чем голое «сейчас уточню "
        "информацию».",
        "Не выдумывать данные, не называть цифры и адреса, не начинать заново, не здороваться.",
        _HARD_FACT_BAN,
        "Реплика опирается на историю разговора. Не повторять сказанное, "
        "не задавать вопрос, ответ на который уже прозвучал.",
    ]
    cont = continuation_block(turn_kind=turn_kind)
    if cont:
        lines.append(cont)
    ctx = context_block(context_text)
    if ctx:
        lines.append(ctx)
    if step is not None:
        name = getattr(step, "name", None) or getattr(step, "goal", None) or step.id
        lines.append(f"Ведущий шаг: {step.id} — {name}.")
    form = profile_block(script, profile, pending_fields=pending_fields)
    lines.append(form)

    limit = max(1, int(history_limit))
    tail = list(messages)[-limit:]
    if not tail:
        tail = [HumanMessage(content="(клиент молчит)")]
    return [SystemMessage(content="\n".join(lines)), *tail]


def build_turn_messages(
    *,
    script: CompiledScript,
    steps: Sequence[AnyStep] | None = None,
    step: AnyStep | None = None,
    profile: Mapping[str, str],
    facts: Mapping[str, Any],
    history: Sequence[BaseMessage],
    asides_done: Sequence[str],
    next_step: AnyStep | None = None,
    context_text: str = "",
    dynamic_status: str = "",
    pending_fields: Sequence[str] = (),
    turn_kind: str = "client",
    closed_steps: Sequence[AnyStep] = (),
) -> list[BaseMessage]:
    """Собирает сообщения запроса к генератору.

    Порядок: персона + естественность + unknown → статика контекста →
    профиль → факты хода → шапка → возражения → инструкция схемы →
    история целиком. Ветка ожидания сюда не входит.

    Args:
        script: скомпилированный скрипт.
        steps: шапка шагов; если None — собирается из step/next_step.
        step: ведущий шаг (совместимость).
        profile: собранный профиль.
        facts: факты хода.
        history: история звонка без системных сообщений; уходит целиком.
        asides_done: отработанные возражения.
        next_step: следующий шаг; только источник завершающего вопроса,
            в шапку висящих не входит.
        context_text: документ контекста.
        dynamic_status: статус динамики контекста (``готово`` / ``не нашлось``).
        pending_fields: поля профиля, которые сейчас разбираются.
        turn_kind: ``client`` или ``continuation``.
        closed_steps: шаги, уже закрытые к этому ходу.

    Returns:
        Список сообщений: одно системное и полная история.
    """
    head: list[AnyStep]
    if steps is not None:
        head = list(steps)
    elif step is not None:
        # next_step в шапку не кладём — только отдельно как источник вопроса.
        head = [step]
    else:
        head = []

    ask_for_move = bool(head)
    blocks: list[str] = [
        persona_block(),
        naturalness_block(ask_for_move=ask_for_move, pending_only=False),
        unknown_block(script),
    ]
    closed = closed_steps_block(closed_steps)
    if closed:
        blocks.append(closed)
    cont = continuation_block(turn_kind=turn_kind)
    if cont:
        blocks.append(cont)
    status_note = dynamic_status_block(status=dynamic_status)
    if status_note:
        blocks.append(status_note)
    ctx_for_steps = ""
    if context_text:
        ctx = context_block(context_text)
        if ctx:
            blocks.append(ctx)
        ctx_for_steps = context_text
    blocks.append(profile_block(script, profile, pending_fields=pending_fields))

    facts_text = facts_block(facts)
    if facts_text:
        blocks.append(facts_text)

    blocks.append(
        steps_block(
            head,
            profile,
            facts,
            context_text=ctx_for_steps,
            next_step=next_step,
        )
    )
    if not script.is_sales:
        blocks.append(aside_block(script, asides_done))
    blocks.append(
        "Вернуть ответ строго по схеме. В поле reply — только то, что звучит "
        "вслух: живой разговорный русский, без списков и канцелярита."
    )

    tail = list(history)
    if not tail:
        tail = [HumanMessage(content="(клиент молчит)")]
    return [SystemMessage(content="\n\n".join(blocks)), *tail]
