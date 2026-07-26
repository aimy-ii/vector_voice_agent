"""Контекстер: наполняет динамику контекста и ставит ей статус.

Единственный писатель динамической части. Статику не трогает — её пишет
``lookup_node``. Источники ответа — реестр ``ContextTool``. Интерфейс
наружу не меняется.

Прогрев под предстоящий шаг (мета города, филиалы, цена) идёт отдельно
по ``needs_of`` / ``есть_в_базе``; контекстер отвечает на побочный вопрос
клиента. В формате продаж справок в скрипте нет — только FAQ справочника.
"""

from __future__ import annotations

from typing import Mapping, Sequence

from graph.context import (
    DYN_MISSING,
    DYN_NEED_CITY,
    DYN_NONE,
    DYN_READY,
    ConversationContext,
)
from graph.history import find_aside
from graph.tools_registry import NEED_CITY_SIGNAL, ContextTool
from script.models import Objection


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
    """Статус «не нашлось»: инструмент сходил и ничего не нашёл."""
    context.dynamic_status = DYN_MISSING
    context.situation_slug = None
    context.filler_spoken = False


def _mark_need_city(context: ConversationContext) -> None:
    """Статус «нужен город»: без города клиента ответить нельзя."""
    context.dynamic_status = DYN_NEED_CITY
    context.situation_slug = None
    context.filler_spoken = False


def _mark_none(context: ConversationContext) -> None:
    """Статус «не требуется»: реестр ничего не дал / возражение."""
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
    трогает — это тактика генератора. Если никто не ответил — «не
    требуется» (генератор опирается на статику). «Не нашлось» — только
    когда инструмент подошёл и вернул пустую строку.

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

    # Реестр: первый инструмент с не-None даёт ответ (или пусто / нужен город).
    for tool in tools:
        found = await tool.try_answer(reply, updated)
        if found is None:
            continue
        if found == NEED_CITY_SIGNAL:
            _mark_need_city(updated)
            return updated
        if found.strip():
            _append_dynamic(updated, found)
            _mark_ready(updated)
            updated.filler_spoken = False
        else:
            _mark_missing(updated)
        return updated

    # Ни один инструмент не подошёл — статика уже в промпте, статус не нужен.
    _mark_none(updated)
    return updated
