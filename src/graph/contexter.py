"""Контекстер: наполняет динамику контекста и ставит ей статус.

Единственный писатель динамической части. Статику не трогает — её пишет
``lookup_node``. Справки берёт из ``helps`` скрипта; реальный поиск
подключается позже через ``ContexterTools``. Интерфейс наружу не меняется.
"""

from __future__ import annotations

import re
from typing import Mapping, Protocol, Sequence

from graph.context import (
    DYN_MISSING,
    DYN_NONE,
    DYN_READY,
    ConversationContext,
)
from graph.history import find_aside
from script.models import Help, Objection

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


def _triggers_catalogue(
    items: Mapping[str, Help] | Mapping[str, Objection],
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


async def run_contexter(
    context: ConversationContext,
    *,
    reply: str,
    tools: ContexterTools,
    helps: Mapping[str, Help] | None = None,
    objections: Mapping[str, Objection] | None = None,
) -> ConversationContext:
    """Наполняет динамику контекста и ставит ей статус.

    Единственный писатель динамической части. Справки — из ``helps`` скрипта
    (по ``find_aside`` / триггерам). Возражения не трогает — это тактика
    генератора. Решает: нужного нет — «не нашлось»; ответ уже в
    статике/накопленном — «готово» сразу; нашлось через поиск — «готово».

    Args:
        context: текущий контекст разговора.
        reply: реплика клиента на этот ход.
        tools: набор инструментов поиска.
        helps: справки скрипта; ``None`` — не сверять.
        objections: возражения скрипта; при совпадении статус «не требуется».

    Returns:
        Контекст с обновлённой динамикой и статусом; статика без изменений.
    """
    updated = context.model_copy(deep=True)

    # Возражения — тактика разговора в скрипте, контекстер не обрабатывает.
    if objections and find_aside(reply, _triggers_catalogue(objections)):
        updated.dynamic_status = DYN_NONE
        updated.situation_slug = None
        return updated

    # Справки из статичного списка helps — источник до появления реального поиска.
    if helps:
        help_id = find_aside(reply, _triggers_catalogue(helps))
        if help_id is not None:
            item = helps.get(help_id)
            text = (item.text if item else "").strip()
            if text:
                _append_dynamic(updated, text)
                updated.dynamic_status = DYN_READY
                updated.situation_slug = None
                updated.filler_spoken = False
            else:
                updated.dynamic_status = DYN_MISSING
                updated.situation_slug = None
            return updated

    if not _looks_like_fact_request(reply):
        updated.dynamic_status = DYN_NONE
        updated.situation_slug = None
        return updated

    if _answer_already_in_context(reply, updated):
        updated.dynamic_status = DYN_READY
        updated.situation_slug = None
        return updated

    found = await tools.search(reply.strip())
    if found:
        _append_dynamic(updated, found)
        updated.dynamic_status = DYN_READY
        updated.situation_slug = None
        updated.filler_spoken = False
    else:
        # Справочный вопрос есть, но ни в helps, ни в поиске — «не нашлось».
        updated.dynamic_status = DYN_MISSING
        updated.situation_slug = None
        updated.filler_spoken = False

    return updated
