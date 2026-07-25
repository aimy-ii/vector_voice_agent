"""Контекстер: наполняет динамику контекста и ставит ей статус.

Единственный писатель динамической части. Статику не трогает — её пишет
``lookup_node``. Реальный поиск подключается позже через ``ContexterTools``;
интерфейс наружу при этом не меняется.
"""

from __future__ import annotations

import re
from typing import Protocol

from graph.context import (
    DYN_NONE,
    DYN_READY,
    DYN_SEARCHING,
    ConversationContext,
)

#: Слаг ситуации по умолчанию, пока точный разбор не подключён.
DEFAULT_SITUATION = "default"

#: Признаки вопроса / запроса факта в реплике.
_QUESTION_MARKERS = re.compile(
    r"(?i)(\?|сколько|как\s|где\s|какой|какая|какие|что\s|есть\sли|"
    r"подскаж|расскаж|уточн|интересует|нужн)"
)

#: Слова короче этого не считаем значимыми для пересечения со статикой.
_MIN_TOKEN = 4


class ContexterTools(Protocol):
    """Инструменты контекстера.

    Точка замены: сегодня заглушки/справочник, завтра вектор, полнотекст,
    карты. Интерфейс стабилен.
    """

    async def search(self, query: str) -> str | None:
        """Ищет ответ по запросу; ``None`` — ничего не найдено."""
        ...


class NullContexterTools:
    """Боевая заглушка: поиска нет, всегда ``None``."""

    async def search(self, query: str) -> str | None:
        """Поиск не подключён — всегда промах."""
        return None


def _looks_like_fact_request(reply: str) -> bool:
    """Грубая эвристика: реплика похожа на запрос факта."""
    text = (reply or "").strip()
    if not text:
        return False
    return bool(_QUESTION_MARKERS.search(text))


def _tokens(text: str) -> set[str]:
    """Значимые токены для пересечения со статикой."""
    return {t for t in re.findall(r"[а-яёa-z0-9]+", text.lower()) if len(t) >= _MIN_TOKEN}


def _answer_already_in_context(reply: str, context: ConversationContext) -> bool:
    """Ответ на реплику уже есть в статике или накопленной динамике."""
    haystack = f"{context.static_text}\n{context.dynamic_text}".strip().lower()
    if not haystack:
        return False
    reply_tokens = _tokens(reply)
    if not reply_tokens:
        return False
    known = _tokens(haystack)
    return bool(reply_tokens & known)


async def run_contexter(
    context: ConversationContext,
    *,
    reply: str,
    tools: ContexterTools,
) -> ConversationContext:
    """Наполняет динамику контекста и ставит ей статус.

    Единственный писатель динамической части. Решает: нужного нет — статус «в
    поиске» + слаг ситуации; ответ уже в статике/накопленном — «готово» сразу,
    без вызовов; не нашлось — «не нашлось». Реальные инструменты поиска
    подключаются позже, интерфейс наружу при этом не меняется.

    Args:
        context: текущий контекст разговора.
        reply: реплика клиента на этот ход.
        tools: набор инструментов поиска.

    Returns:
        Контекст с обновлённой динамикой и статусом; статика без изменений.
    """
    updated = context.model_copy(deep=True)

    if not _looks_like_fact_request(reply):
        updated.dynamic_status = DYN_NONE
        updated.situation_slug = None
        return updated

    if _answer_already_in_context(reply, updated):
        updated.dynamic_status = DYN_READY
        updated.situation_slug = None
        return updated

    # Каркас: поиск не подключён. Ставим «в поиске» + общий слаг, чтобы
    # генератор мог отдать заглушку. Точный разбор ситуаций — следующий этап.
    # Если tools.search когда-нибудь вернёт текст — допишем динамику и READY.
    found = await tools.search(reply.strip())
    if found:
        dynamic = (updated.dynamic_text + "\n" + found).strip()
        updated.dynamic_text = dynamic
        updated.dynamic_status = DYN_READY
        updated.situation_slug = None
        updated.filler_spoken = False
    else:
        # Поиска нет / промах: для каркаса оставляем «в поиске» с default,
        # чтобы проверить заглушку; «не нашлось» — когда заглушка не нужна.
        updated.dynamic_status = DYN_SEARCHING
        updated.situation_slug = DEFAULT_SITUATION
        # Новый заход поиска — флаг заглушки сбрасываем.
        updated.filler_spoken = False

    return updated
