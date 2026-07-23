"""Разбор истории звонка.

Всё здесь — чистые функции над списком сообщений: ни сети, ни модели.
Логика диалога тестируется без звука и без провайдеров именно поэтому.

Отдельно про системные сообщения. Бот кладёт свои инструкции в историю как
сообщение с ролью `system` (в `livekit-agents 1.6.6` — с идентификатором из
константы `INSTRUCTIONS_MESSAGE_ID`), а плагин честно превращает их в
`SystemMessage` и подаёт графу. Свой промпт граф собирает сам, поэтому
входящие системные сообщения отбрасываются: два промпта в одном запросе
дерутся между собой.
"""

from __future__ import annotations

import re
from typing import Iterable, Sequence

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

#: Короткие подтверждения, после которых спрашивать нечего — можно сразу
#: выталкивать следующий блок скрипта, не тратя ход на модель.
_ACKS: frozenset[str] = frozenset(
    {
        "да",
        "ага",
        "угу",
        "конечно",
        "хорошо",
        "ладно",
        "окей",
        "ок",
        "давайте",
        "давай",
        "понятно",
        "понял",
        "поняла",
        "ясно",
        "слушаю",
        "верно",
        "точно",
        "именно",
        "правильно",
        "продолжайте",
        "рассказывайте",
        "интересно",
        "супер",
        "отлично",
        "спасибо",
    }
)

#: Сколько слов ещё считается коротким подтверждением.
_ACK_MAX_WORDS = 3


def strip_system(messages: Iterable[BaseMessage]) -> list[BaseMessage]:
    """Убирает системные сообщения, пришедшие от бота.

    Приведение словарей к ``BaseMessage`` — в редьюсере ``replace_messages``:
    сюда история должна приходить уже объектами.

    Args:
        messages: история звонка (объекты сообщений).

    Returns:
        История без системных сообщений.
    """
    return [m for m in messages if m.type not in ("system", "developer")]


def text_of(message: BaseMessage) -> str:
    """Достаёт текст сообщения строкой.

    Args:
        message: сообщение истории.

    Returns:
        Текст без окружающих пробелов; пустая строка, если текста нет.
    """
    content = message.content
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = [part for part in content if isinstance(part, str)]
        return " ".join(parts).strip()
    return ""


def last_user_text(messages: Sequence[BaseMessage]) -> str:
    """Возвращает последнюю реплику клиента.

    Args:
        messages: история звонка.

    Returns:
        Текст последней реплики клиента; пустая строка, если её нет.
    """
    for message in reversed(list(messages)):
        if isinstance(message, HumanMessage):
            return text_of(message)
    return ""


def last_agent_text(messages: Sequence[BaseMessage]) -> str:
    """Возвращает последнюю фактически произнесённую реплику бота.

    В историю попадает то, что бот успел сказать: при перебивании там
    обрезанный текст. Это и есть правда о произнесённом — по ней и считаем,
    что клиент услышал.

    Args:
        messages: история звонка.

    Returns:
        Текст последней реплики бота; пустая строка, если её нет.
    """
    for message in reversed(list(messages)):
        if isinstance(message, AIMessage):
            return text_of(message)
    return ""


def is_first_turn(messages: Sequence[BaseMessage]) -> bool:
    """Первый ли это содержательный ход бота в звонке.

    Во всех входящих звонках первым говорит клиент, и говорит по делу.
    Скрипт подключается со второго хода.

    Args:
        messages: история звонка.

    Returns:
        True, если бот ещё ничего не отвечал по существу.
    """
    return not any(isinstance(m, AIMessage) for m in messages)


def normalize(text: str) -> str:
    """Приводит реплику к виду для сравнения: нижний регистр, «ё» → «е», без знаков.

    Пробелы схлопываются: знак препинания превращается в пробел, и без
    схлопывания «Всё, понятно» дало бы двойной пробел — признак справки
    вроде «под ключ» после этого перестал бы находиться.

    Args:
        text: исходный текст.

    Returns:
        Нормализованная строка.
    """
    lowered = text.strip().lower().replace("ё", "е")
    without_marks = re.sub(r"[^\w\s]+", " ", lowered, flags=re.UNICODE)
    return re.sub(r"\s+", " ", without_marks).strip()


def is_acknowledgement(text: str) -> bool:
    """Короткое подтверждение вроде «да, конечно» или «понятно».

    Такие реплики не несут ни ответа на вопрос шага, ни постороннего вопроса,
    поэтому модель для них не нужна: следующий блок скрипта выталкивается
    напрямую. Это самый частый случай в презентации, и он же самый дешёвый.

    Args:
        text: реплика клиента.

    Returns:
        True, если реплика — только подтверждение.
    """
    words = normalize(text).split()
    if not words or len(words) > _ACK_MAX_WORDS:
        return False
    return all(word in _ACKS for word in words)


def matches_triggers(text: str, triggers: Sequence[str]) -> bool:
    """Есть ли в реплике признак срабатывания справки или возражения.

    Быстрый путь по подстрокам. Когда список справок заменят поиском,
    поменяется только эта проверка — форма записи справки останется прежней.

    Args:
        text: реплика клиента.
        triggers: признаки срабатывания из данных скрипта.

    Returns:
        True, если сработал хотя бы один признак.
    """
    haystack = normalize(text)
    return any(normalize(trigger) in haystack for trigger in triggers if trigger)


def find_aside(text: str, catalogue: dict[str, Sequence[str]]) -> str | None:
    """Ищет справку или возражение по признакам срабатывания.

    Args:
        text: реплика клиента.
        catalogue: идентификатор → признаки срабатывания.

    Returns:
        Идентификатор первого совпадения или None.
    """
    for aside_id, triggers in catalogue.items():
        if matches_triggers(text, triggers):
            return aside_id
    return None
