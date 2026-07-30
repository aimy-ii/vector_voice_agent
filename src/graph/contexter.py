"""Контекстер: агент с инструментами, единственный писатель динамики.

Решает, нужен ли контекст под реплику, выбирает инструмент, отдаёт предмет
для ситуативной заглушки и по результату выставляет статус динамики.
Статику не трогает — её пишут ``lookup_node`` и прогрев лайв-канала.
Возражения остаются тактикой скрипта: при совпадении статус «не требуется».

Работает в лайв-канале: инструмент всегда дожидаемся, спешить некуда.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Mapping, Sequence

from graph.context import (
    DYN_MISSING,
    DYN_NONE,
    DYN_READY,
    DYN_SEARCHING,
    ConversationContext,
)
from graph.context_agent import ContextAgent, decide_context
from graph.history import find_aside
from graph.log_fmt import format_contexter_done
from graph.progress import stage
from graph.tools_registry import ContextTool
from script.models import Objection

log = logging.getLogger(__name__)


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


def _clear_subject_unless_searching(context: ConversationContext) -> None:
    """Чистит предмет, если статус не «в поиске»."""
    if context.dynamic_status != DYN_SEARCHING:
        context.situation_slug = None


def _mark_ready(context: ConversationContext) -> None:
    """Статус «готово»: генератор может опираться на контекст."""
    context.dynamic_status = DYN_READY
    _clear_subject_unless_searching(context)


def _mark_missing(context: ConversationContext) -> None:
    """Статус «не нашлось»: инструмент сходил и ничего не нашёл."""
    context.dynamic_status = DYN_MISSING
    _clear_subject_unless_searching(context)
    context.filler_spoken = False


def _mark_searching(context: ConversationContext, subject: str) -> None:
    """Статус «в поиске»: заглушка в эфир, результат в этот ход не ждём.

    Остаётся в контракте на случай, когда инструмент действительно долгий.
    """
    context.dynamic_status = DYN_SEARCHING
    context.situation_slug = (subject or "").strip() or None
    context.filler_spoken = False


def _mark_none(context: ConversationContext) -> None:
    """Статус «не требуется»: контекст не нужен / возражение."""
    context.dynamic_status = DYN_NONE
    _clear_subject_unless_searching(context)


def _tool_by_name(tools: Sequence[ContextTool], name: str | None) -> ContextTool | None:
    """Находит инструмент по имени в реестре."""
    if not name:
        return None
    for tool in tools:
        if tool.name == name:
            return tool
    return None


def _finish(
    updated: ConversationContext,
    *,
    reply: str,
    tool: str | None,
    subject: str,
    started: float,
    needed: bool,
    branch_slugs: Sequence[str] = (),
) -> ConversationContext:
    """Выставляет ``dynamic_reply``, пишет лог и возвращает контекст."""
    updated.dynamic_reply = reply
    elapsed_ms = int((time.perf_counter() - started) * 1000)
    stage(
        "contexter",
        format_contexter_done(
            tool=tool,
            subject=subject,
            status=updated.dynamic_status,
            elapsed_ms=elapsed_ms,
            needed=needed,
            branch_slugs_count=len(branch_slugs) if tool == "branches" else None,
        ),
        "done",
    )
    return updated


async def run_contexter(
    context: ConversationContext,
    *,
    reply: str,
    tools: Sequence[ContextTool],
    objections: Mapping[str, Objection] | None = None,
    agent: ContextAgent | None = None,
    branches: Sequence[Mapping[str, Any]] = (),
) -> ConversationContext:
    """Наполняет динамику контекста и ставит ей статус.

    Единственный писатель динамической части. Агент решает, нужен ли
    поход и каким инструментом. Возражения не трогает — это тактика
    генератора. Инструмент всегда дожидаемся — контекстер в лайв-канале.

    Args:
        context: текущий контекст разговора.
        reply: реплика клиента на этот ход.
        tools: реестр инструментов для агента.
        objections: возражения скрипта; при совпадении статус «не требуется».
        agent: подмена агента для офлайн-тестов.
        branches: филиалы города для отбора слагов агентом.

    Returns:
        Контекст с обновлённой динамикой и статусом; статика без изменений.
    """
    started = time.perf_counter()
    updated = context.model_copy(deep=True)

    # Возражения — тактика разговора в скрипте, контекстер не обрабатывает.
    if objections and find_aside(reply, _triggers_catalogue(objections)):
        _mark_none(updated)
        return _finish(
            updated,
            reply=reply,
            tool=None,
            subject="",
            started=started,
            needed=False,
        )

    decision = await decide_context(reply, updated, tools, agent=agent, branches=branches)
    if not decision.need:
        _mark_none(updated)
        return _finish(
            updated,
            reply=reply,
            tool=None,
            subject=decision.subject,
            started=started,
            needed=False,
            branch_slugs=decision.branch_slugs,
        )

    tool = _tool_by_name(tools, decision.tool)
    if tool is None:
        _mark_none(updated)
        return _finish(
            updated,
            reply=reply,
            tool=decision.tool,
            subject=decision.subject,
            started=started,
            needed=False,
            branch_slugs=decision.branch_slugs,
        )

    try:
        found = await tool.run(decision.query, updated, slugs=decision.branch_slugs)
    except Exception as exc:  # noqa: BLE001
        log.warning("Инструмент контекстера не ответил: %s", exc)
        found = ""

    if (found or "").strip():
        _append_dynamic(updated, found)
        _mark_ready(updated)
        updated.filler_spoken = False
    else:
        _mark_missing(updated)

    return _finish(
        updated,
        reply=reply,
        tool=decision.tool,
        subject=decision.subject,
        started=started,
        needed=True,
        branch_slugs=decision.branch_slugs,
    )
