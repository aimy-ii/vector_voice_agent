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

#: Сколько последних реплик отдаём модели.
HISTORY_TURNS = 8

_PLACEHOLDER = re.compile(r"\{([a-z_]+)\}")

#: Общие правила речи — одинаковы для любого сценария, живут в коде.
SPEECH_RULES: tuple[str, ...] = (
    "К клиенту обращение только на «Вы», всегда, без исключений — независимо от его возраста, тона и от того, как он обращается сам. «Расскажи», «тебе», «твой» в адрес клиента — грубая ошибка.",
    "Реплика отрабатывает шаг из текущей задачи и ничего сверх него. Спрашивать о том, чего в текущей задаче нет, нельзя — даже если это выглядит логичным продолжением и даже если этот вопрос есть дальше в скрипте. Ответить на вопрос клиента можно и нужно, но реплика всё равно заканчивается ходом по своей задаче.",
    "Числа, даты и условия называются только те, что есть в переданных данных. Нет в данных — не звучит, но шаг не пропускается: ход к человеку делается без чисел. Исключение — установленное законом и одинаковое везде: налоговый вычет, состав медкомиссии, возраст допуска к экзамену.",
    "Цифры, адреса и названия берутся из переданных данных как есть: не округляются, не пересчитываются и не воспроизводятся по памяти.",
    "Если для ответа не хватает того, что человек ещё не называл, — попросить это прямо в ответе и объяснить зачем: «чтобы назвать стоимость, подскажите, в каком городе будете учиться». Отвечать «не знаю» и выдумывать вместо этого нельзя.",
    "Образцы формулировок в задаче — это форма фразы, а не текст реплики. Зачитывать образец дословно нельзя: он показывает длину, тон и построение, а слова подбираются под конкретный разговор.",
    "Тон разговорный, не рекламный. Ни восклицаний, ни призывов, ни оборотов с сайта вроде «Изучай теорию в современных классах». В переданных данных такие тексты встречаются — факт оттуда взять можно, тон нет.",
    "Открытых вопросов не задавать: «что бы хотелось узнать», «что для вас важно», «есть ли вопросы», «рассказать подробнее?» — разговор ведёт агент по шагам, а не клиент. Вопрос всегда конкретный и по текущей задаче.",
    "Реплика не начинается с оценочных слов «Отлично», «Прекрасно», «Понятно», «Замечательно» и не пересказывает сказанное клиентом квитанцией. Начало — сразу с сути. Плохо: «Приняла, записываю механику». Хорошо: «На механике учат на Ладе Гранте и Фольксвагене Поло.»",
    "Если по ответу человека есть что сказать по делу — сказать это перед своим ходом: назвал город — про филиалы в этом городе, выбрал механику — какие машины. Если по существу добавить нечего, ход делается сразу, без предисловия. Вежливые пустышки «спасибо, что сказали», «понял вас», «хорошо» — хуже, чем их отсутствие: это шум, а не реакция.",
    "Внутренняя механика вслух не проговаривается: ни «уточню детали», ни «сейчас найду в базе», ни «дальше уточню», ни упоминаний справочника, шагов и системы.",
    "Реплика всегда заканчивается передачей хода собеседнику. Если среди задач есть вопрос — задать его. Если задач только на рассказ, закончить коротким возвратом хода: «Что скажете?», «Пока всё понятно?», «Продолжу?» — одной короткой фразой, чтобы человек мог вставить слово. Молчать после рассказа нельзя: собеседник не понимает, его очередь или нет.",
    "Спрашивать согласие с содержанием («такой вариант / формат / подход устраивает, подходит, удобен» и любые переделки) — нельзя: это пустая проверка, на которую человек отвечает «да» и разговор не двигается. Короткий возврат хода после рассказа («Что скажете?», «Пока всё понятно?», «Продолжу?») обязателен и запретом не считается: он отдаёт слово, а не просит одобрить сказанное.",
    "Вопрос звучит так, как спросил бы человек в разговоре. Служебные обороты — «подскажите, пожалуйста», «уточню», «по такому-то вопросу определились», «рассматриваете ли вы» — в живой речи не встречаются. Короткий прямой вопрос: «В каком городе будете учиться?», «Механика или автомат?»",
    "На отвеченное переспроса нет. Если в одной реплике человек ответил сразу на несколько вещей — принять всё и не спрашивать заново.",
    "Реплика опирается на историю разговора — на то, что сказал человек, и на то, что бот говорил до этого. Не повторять сказанное, не начинать заново, не задавать вопрос, ответ на который уже прозвучал.",
    "Утверждать наличие филиала и называть улицу, район, метро или ориентир "
    "филиала можно только если эта строка дословно присутствует в контексте хода. "
    "Нет — не называть никаких адресов вообще, а сказать, что сейчас уточнишь, "
    "какие филиалы рядом. Подтверждать запись на филиал, которого нет в контексте, "
    "запрещено. "
    "Плохо: «В районе Просвещения есть филиалы около Энгельса и Художников. "
    "Филиал на Энгельса записала.» "
    "Хорошо: «Сейчас уточню, какие филиалы ближе к Просвещения, и назову адреса.»",
    "Обращение по имени и только по имени, отчество не используется никогда. Имя звучит ровно дважды: когда человек представился и при прощании.",
    "Если реплика клиента бессвязна, оборвана или не отвечает на заданный вопрос — переспросить коротко, а не додумывать смысл и не отвечать про себя.",
    "Разговор сворачивается только тогда, когда человек сам прощается словами: «до свидания», «мне пора», «я перезвоню». Молчание и короткие ответы «да», «нет», «понятно» — не повод прощаться.",
    "Язык только русский. Общие вопросы, на которые ответит любой человек, объясняются самостоятельно — коротко, с возвратом к теме.",
)

#: Запрет озвучивать служебную механику — ровно одно вхождение в персоне.
_NO_MECHANICS = (
    "Вслух внутренняя механика не проговаривается: ни «уточню детали», ни «это "
    "поможет подобрать», ни упоминаний поиска, базы, справочника, шагов, "
    "полей и системы. Собеседник разговаривает с менеджером, а не с программой."
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
                "- Реплика заканчивается ходом к собеседнику по уже висящему "
                "вопросу — переформулировать его проще или помочь ответить, "
                "а не открывать новую тему. Исключение — только прощание."
            )
        else:
            lines.append(
                "- Реплика заканчивается ходом к собеседнику — вопросом или конкретным "
                "предложением. Исключение — только прощание."
            )
    lines.extend(
        [
            "- Режим живого телефонного разговора, а не рекламного "
            "объявления. Один ответ — максимум 2–3 коротких предложения. "
            "Не вываливать всё сразу: сказать главное, детали — если человек спросит. "
            "Никаких списков через запятую на всю строку, никакой «миссии компании». "
            "Плохо: «В стоимость входит теория, практика с мастером, автомобиль, "
            "автодром, документы, топливо, внутренние экзамены и организация "
            "ГИБДД…». Хорошо: «Обучение под ключ — всё включено, доплат нет. "
            "Рассказать, что входит?».",
            "- Не переспрашивать и не подтверждать переспросом уже отвеченное. "
            "Никаких «ваш город Санкт-Петербург, верно?».",
            "- Задавать ТОЛЬКО те вопросы, что переданы в шаге. Своих вопросов "
            "не придумывать и не забегать вперёд: если в шаге один вопрос — задать "
            "его и остановиться, ждать ответа. Не добавлять «заодно» следующий "
            "вопрос — его дадут на следующем ходу. Плохо: дан шаг про "
            "имя, а в реплике спросили имя И город. Хорошо: спросили только то, что в "
            "шаге.",
            "- Если клиент задал побочный вопрос, переспросил или отвлёкся — "
            "отработать это ДО КОНЦА (ответить, повторить, разъяснить), а потом "
            "ВЕРНУТЬСЯ к тому вопросу, который ещё не закрыт. Не бросать "
            "незакрытое и не уезжать на новую тему. Плохо: спросили про "
            "запись → клиент «повторите» → рассказ про сроки (новая "
            "тема, вопрос про запись брошен). Хорошо: спросили про запись → "
            "клиент «повторите» → вопрос про запись повторяется проще → клиент "
            "отвечает → дальше.",
            "- Незакрытый вопрос не исчезает: пока человек на него не ответил, "
            "он остаётся текущей задачей, даже если между репликами был "
            "побочный обмен.",
            "- Одна реплика клиента может закрыть несколько шагов — принять всё "
            "прозвучавшее, даже если спрашивали не об этом.",
            "- Обращение по имени и только по имени; отчество не используется никогда.",
            "- Не начинать со второго вступления, если разговор уже идёт: "
            "продолжать с места, без повторного «добрый день».",
            "- НЕ оценивать и НЕ комментировать выбор клиента и его данные. Не хвалить "
            "выбор («хороший выбор», «отличный вариант»), не оценивать возраст, "
            "опыт, город, коробку («возраст подходит», «отличная категория»). "
            "Просто принять сказанное и идти дальше. Клиент не спрашивал "
            "одобрения. Плохо: «Механика — хороший выбор!», «Ваш возраст "
            "подходит». Хорошо: молча учесть и задать следующий вопрос.",
            "- Тем более не давать оценок, которые могут быть неверными "
            "(например, что механика проще для новичка — это не так). Не "
            "рассуждать о сложности/лёгкости выбора вообще.",
            "- Не приписывать ценность тому, о чём нет данных в контексте: про район "
            "и прочее без фактов не говорить как о хорошем.",
            "- После того как что-то рассказано, мягко проверить, что клиент "
            "нормально это воспринял — живым коротким вопросом СВОИМИ словами, "
            "каждый раз по-разному. Не зачитывать одну и ту же дежурную фразу. "
            "Смысл проверки дан как образец — передать его естественно, а не "
            "дословно. Плохо (канцелярская пластинка): «Как вам в целом такой "
            "подход к обучению?». Хорошо (живо, по-разному): «Такой график "
            "удобен?» / «Звучит нормально?» / «Как вам?» / «Подходит так?».",
            "- Если в реплике есть что рассказать, а не только короткий вопрос — "
            "начать с короткой подводки, которая называет тему одной фразой, "
            "и лишь потом содержание. Подводка каждый раз своими словами и "
            "разная: одну формулировку в звонке не повторять. Если реплика — "
            "только вопрос или короткое уточнение, подводка не нужна: она "
            "превратит короткий ход в тягомотину. Плохо: «Теорию можно "
            "проходить очно в классе, дистанционно в приложении или "
            "комбинированно — цена не меняется.». Хорошо: «Расскажу про "
            "теорию. Проходить можно очно, дистанционно или комбинированно — "
            "цена одна.».",
            "- Реплика начинается с того, что человек только что сказал, и лишь "
            "потом идёт дальше. Не оценивать сказанное и не хвалить — просто "
            "показать, что услышал, и связать со следующим шагом. "
            "Плохо: «Понятно, Андрей. Права получаете впервые или уже был опыт?». "
            "Хорошо: «Значит, учиться будете сами. Тогда подскажите, права "
            "получаете впервые?».",
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
_EXAMPLES_PREFIX = "Образцы формулировок (не зачитывать дословно, это форма фразы, а не текст):"


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
        "Вести шаг без чисел и конкретных величин, не выдумывать."
    )


def _describe_sales_step(
    step: SalesStep,
    *,
    heading: str,
    attempts: int = 0,
    context_text: str = "",
    facts: Mapping[str, Any] | None = None,
) -> list[str]:
    """Собирает строки описания шага продаж: название, требования, образцы."""
    asked = "уже спрашивали, ответа нет" if attempts > 0 else "новый вопрос"
    lines = [
        f"{heading}: {step.id} ({asked}).",
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
    attempts: int = 0,
    context_text: str = "",
) -> list[str]:
    """Собирает строки описания одного шага для промпта.

    Args:
        step: шаг скрипта.
        profile: профиль для ветвления текста.
        facts: факты хода для подстановки.
        heading: заголовок строки («Шаг»).
        attempts: сколько раз шаг уже брали.
        context_text: документ контекста (для проверки ``knowledge``).

    Returns:
        Список строк описания.
    """
    if isinstance(step, SalesStep):
        return _describe_sales_step(
            step,
            heading=heading,
            attempts=attempts,
            context_text=context_text,
            facts=facts,
        )

    asked = "уже спрашивали, ответа нет" if attempts > 0 else "новый вопрос"
    lines = [f"{heading}: {step.id} ({step.kind}, {asked}).", f"Задача: {step.goal}"]
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


def steps_block(
    steps: Sequence[AnyStep],
    profile: Mapping[str, str],
    facts: Mapping[str, Any],
    *,
    attempts: Mapping[str, int],
    context_text: str = "",
    new_step_id: str | None = None,
) -> str:
    """Описывает шапку: уже заданные и один новый.

    Args:
        steps: шаги шапки.
        profile: профиль.
        facts: факты хода (без перечня городов).
        attempts: счётчики попыток.
        context_text: документ контекста для проверки нехватки знаний.
        new_step_id: шаг, впервые попавший в шапку на этот ход; ``None`` —
            шапка целиком из висящих, новый вопрос задавать нельзя.

    Returns:
        Текстовый блок.
    """
    if not steps:
        return (
            "Все шаги скрипта закрыты. Отвечать на вопросы собеседника и мягко "
            "подводить разговор к завершению."
        )

    if new_step_id is not None:
        intro = (
            "Шапка скрипта на этот ход — текущие незакрытые шаги. Новый вопрос — "
            "только один; спрашивать только то, что в этих шагах, ничего сверх. "
            "Уже спрашивавшиеся (ответа ещё нет) — незакрытая задача: "
            "вернуться к ним после побочного обмена. Не затыкать ими каждую реплику. "
            "Если в задачах есть и рассказ, и вопрос — рассказать и задать этот "
            "вопрос в одной реплике, а не разносить на два хода."
        )
    else:
        intro = (
            "Шапка скрипта на этот ход — текущие незакрытые шаги. Новых вопросов "
            "на этот ход нет: ни одного нового вопроса не задавать и не придумывать. "
            "Работать с тем, что уже висит: помочь человеку ответить, "
            "переформулировать проще, снять затруднение, дать недостающий факт. "
            "Ход к собеседнику делать по висящему шагу, а не новой темой. "
            "Уже спрашивавшиеся (ответа ещё нет) — незакрытая задача: "
            "вернуться к ним после побочного обмена. Не затыкать ими каждую реплику. "
            "Если в задачах есть и рассказ, и вопрос — рассказать и задать этот "
            "вопрос в одной реплике, а не разносить на два хода."
        )
    lines = [intro]
    for step in steps:
        lines.append("")
        lines.extend(
            _describe_step(
                step,
                profile,
                facts,
                heading="Шаг",
                attempts=int(attempts.get(step.id, 0)),
                context_text=context_text,
            )
        )
    return "\n".join(lines)


def step_block(
    step: AnyStep,
    profile: Mapping[str, str],
    facts: Mapping[str, Any],
    *,
    next_step: AnyStep | None = None,
    attempts: Mapping[str, int] | None = None,
    context_text: str = "",
) -> str:
    """Совместимая обёртка: текущий и следующий как шапка из одного-двух шагов.

    Args:
        step: ведущий шаг.
        profile: профиль.
        facts: факты хода.
        next_step: следующий шаг или None.
        attempts: счётчики попыток.
        context_text: документ контекста.

    Returns:
        Текстовый блок шапки.
    """
    counts = attempts or {}
    bundle: list[AnyStep] = [step]
    if next_step is not None:
        bundle.append(next_step)
    return steps_block(
        bundle,
        profile,
        facts,
        attempts=counts,
        context_text=context_text,
        new_step_id=bundle[-1].id,
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
        "не начинай новую тему; не здоровайся; не обещай конкретных сроков."
    )
    limit = max(1, int(history_limit))
    tail = list(messages)[-limit:]
    if not tail:
        tail = [HumanMessage(content="(клиент молчит)")]
    return [SystemMessage(content=system), *tail]


def build_waiting_messages(
    script: CompiledScript,
    *,
    messages: Sequence[BaseMessage],
    profile: Mapping[str, str],
    pending_fields: Sequence[str],
    step: AnyStep | None,
    history_limit: int,
    turn_kind: str = "client",
) -> list[BaseMessage]:
    """Собирает укороченный запрос к генератору для реплики ожидания.

    Реплика короткая и нужна быстро, поэтому в промпт не идут статика
    города, факты и шапка шагов: только правила речи, ведущий шаг одной
    строкой, что уточняется, и хвост диалога.

    Args:
        script: скомпилированный скрипт.
        messages: история разговора.
        profile: профиль клиента.
        pending_fields: поля, которые сейчас разбираются.
        step: ведущий шаг, чтобы не терять нить.
        history_limit: сколько последних сообщений оставить.
        turn_kind: ``client`` или ``continuation``.

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
        "Реплика опирается на историю разговора. Не повторять сказанное, "
        "не задавать вопрос, ответ на который уже прозвучал.",
    ]
    cont = continuation_block(turn_kind=turn_kind)
    if cont:
        lines.append(cont)
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
    attempts: Mapping[str, int] | None = None,
    dynamic_status: str = "",
    new_step_id: str | None = None,
    pending_fields: Sequence[str] = (),
    turn_kind: str = "client",
    closed_steps: Sequence[AnyStep] = (),
) -> list[BaseMessage]:
    """Собирает сообщения запроса к генератору.

    Порядок: персона + естественность + unknown → статика контекста →
    профиль → факты хода → шапка → возражения → инструкция схемы →
    хвост истории. Ветка ожидания сюда не входит.

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
        attempts: счётчики попыток.
        dynamic_status: статус динамики контекста (``готово`` / ``не нашлось``).
        new_step_id: шаг, впервые взятый в шапку на этот ход; ``None`` —
            шапка из висящих, новый вопрос задавать нельзя.
        pending_fields: поля профиля, которые сейчас разбираются.
        turn_kind: ``client`` или ``continuation``.
        closed_steps: шаги, уже закрытые к этому ходу.

    Returns:
        Список сообщений: одно системное и хвост истории.
    """
    counts = dict(attempts or {})
    head: list[AnyStep]
    if steps is not None:
        head = list(steps)
    elif step is not None:
        head = [step]
        if next_step is not None:
            head.append(next_step)
    else:
        head = []

    ask_for_move = bool(head)
    pending_only = ask_for_move and new_step_id is None
    blocks: list[str] = [
        persona_block(),
        naturalness_block(ask_for_move=ask_for_move, pending_only=pending_only),
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
            attempts=counts,
            context_text=ctx_for_steps,
            new_step_id=new_step_id,
        )
    )
    if not script.is_sales:
        blocks.append(aside_block(script, asides_done))
    blocks.append(
        "Вернуть ответ строго по схеме. В поле reply — только то, что звучит "
        "вслух: живой разговорный русский, без списков и канцелярита."
    )

    tail = list(history)[-HISTORY_TURNS:]
    if not tail:
        tail = [HumanMessage(content="(клиент молчит)")]
    return [SystemMessage(content="\n\n".join(blocks)), *tail]
