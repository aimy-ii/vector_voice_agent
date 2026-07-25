"""Контекстер: наполняет динамику контекста и ставит ей статус.

Единственный писатель динамической части. Статику не трогает — её пишет
``lookup_node``. Источники ответа — реестр ``ContextTool`` (сегодня справки
из скрипта, завтра вектор/карты). Интерфейс наружу не меняется.
"""

from __future__ import annotations

import re
from typing import Mapping, Sequence

from graph.context import (
    DYN_MISSING,
    DYN_NONE,
    DYN_READY,
    ConversationContext,
)
from graph.history import find_aside
from graph.tools_registry import ContextTool
from script.models import Objection

#: Признаки вопроса / запроса факта в реплике.
_QUESTION_MARKERS = re.compile(
    r"(?i)(\?|сколько|как\s|где\s|какой|какая|какие|что\s|есть\sли|"
    r"подскаж|расскаж|уточн|интересует|нужн)"
)

#: Слова короче этого не считаем значимыми для пересечения со статикой.
_MIN_TOKEN = 4


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


def _triggers_catalogue(
    items: Mapping[str, Objection],
) -> dict[str, Sequence[str]]:
    """Идентификатор → признаки срабатывания."""
    return {item_id: item.triggers for item_id, item in items.items()}


def _append_dynamic(context: ConversationContext, text: str) -> None:
    """Дописывает текст в динамику, без пустых дублей в хвосте."""
    chunk = text.strip()
    if not chunk:
        return
    if chunk in context.dynamic_text:
        return
    dynamic = (context.dynamic_text + "\n" + chunk).strip()
    context.dynamic_text = dynamic


def _mark_ready(context: ConversationContext) -> None:
    """Статус «готово»: генератор может опираться на контекст."""
    context.dynamic_status = DYN_READY
    context.situation_slug = None


def _mark_missing(context: ConversationContext) -> None:
    """Статус «не нашлось»: вопрос был, ответа нет."""
    context.dynamic_status = DYN_MISSING
    context.situation_slug = None
    context.filler_spoken = False


def _mark_none(context: ConversationContext) -> None:
    """Статус «не требуется»: вопрос не справочный / возражение."""
    context.dynamic_status = DYN_NONE
    context.situation_slug = None


async def run_contexter(
    context: ConversationContext,
    *,
    reply: str,
    tools: Sequence[ContextTool],
    objections: Mapping[str, Objection] | None = None,
) -> ConversationContext:
    """Наполняет динамику контекста и ставит ей статус.

    Единственный писатель динамической части. Перебирает реестр
    инструментов по порядку: первый подходящий отвечает. Возражения не
    трогает — это тактика генератора. Решает: нужного нет — «не нашлось»;
    ответ уже в статике/накопленном — «готово» сразу.

    Args:
        context: текущий контекст разговора.
        reply: реплика клиента на этот ход.
        tools: реестр инструментов (порядок = приоритет).
        objections: возражения скрипта; при совпадении статус «не требуется».

    Returns:
        Контекст с обновлённой динамикой и статусом; статика без изменений.
    """
    updated = context.model_copy(deep=True)

    # Возражения — тактика разговора в скрипте, контекстер не обрабатывает.
    if objections and find_aside(reply, _triggers_catalogue(objections)):
        _mark_none(updated)
        return updated

    # Реестр: первый инструмент с не-None даёт ответ (или пусто → «не нашлось»).
    for tool in tools:
        found = await tool.try_answer(reply, updated)
        if found is None:
            continue
        if found.strip():
            _append_dynamic(updated, found)
            _mark_ready(updated)
            updated.filler_spoken = False
        else:
            _mark_missing(updated)
        return updated

    if not _looks_like_fact_request(reply):
        _mark_none(updated)
        return updated

    if _answer_already_in_context(reply, updated):
        _mark_ready(updated)
        return updated

    # Справочный вопрос есть, но никто из реестра не ответил.
    _mark_missing(updated)
    return updated
