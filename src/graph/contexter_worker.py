"""Фоновый воркер контекстера: исполняет разбор реплики до конца.

Служебный проход живёт со стратегией interrupt: новая реплика человека
отменяет незавершённый проход. Поэтому сам проход контекстер не гоняет:
он ставит задачу сюда, в очередь enqueue. Воркер берёт задачи по порядку,
исполняет существующий ``run_contexter`` (агент, инструменты, запись
динамики — без изменений) и сливает результат в кеш звонка, откуда его
читают ход и следующий проход. Статус воркер доводит до итога сам:
зависшего «в поиске» после завершения задачи не остаётся.
"""

from __future__ import annotations

import logging
import time
from typing import Any, TypedDict

from langgraph.graph import StateGraph

from core.config import settings
from graph.context import DYN_MISSING, DYN_READY, DYN_SEARCHING, ConversationContext
from graph.context_store import (
    CONTEXT_FIELDS_DYNAMIC,
    CONTEXT_FIELDS_STATIC,
    context_store,
    merge_context_fields,
)
from graph.contexter import reply_hash, run_contexter
from graph.progress import stage
from graph.tools_registry import build_context_tools
from script.source import registry

log = logging.getLogger(__name__)


class ContexterTaskState(TypedDict, total=False):
    """Одна задача воркеру: разобрать реплику и добыть контекст.

    Attributes:
        call_id: идентификатор звонка; ключ контекста в кеше.
        reply: реплика клиента.
        needs: потребности справочника по шапке хода.
        step_needs: строки знаний ведущего шага для агента.
        profile: слитый профиль разговора.
        script_id: идентификатор скрипта.
        script_version: версия скрипта.
    """

    call_id: str
    reply: str
    needs: list[str]
    step_needs: list[str]
    profile: dict[str, str]
    script_id: str
    script_version: str


def _keep_concurrent_dynamic(base: ConversationContext, overlay: ConversationContext) -> None:
    """Не даёт локальной динамике затереть текст, появившийся в кеше параллельно.

    Args:
        base: свежий слепок кеша перед записью.
        overlay: локальный результат разбора; ``dynamic_text`` дополняется
            чужим текстом на месте.
    """
    concurrent = (base.dynamic_text or "").strip()
    local = (overlay.dynamic_text or "").strip()
    if concurrent and concurrent not in (overlay.dynamic_text or ""):
        overlay.dynamic_text = f"{concurrent}\n{local}".strip() if local else concurrent


async def contexter_task_node(state: ContexterTaskState) -> dict[str, Any]:
    """Исполняет один разбор реплики до конца и пишет результат в кеш.

    Свежесть проверяется на входе: реплика уже разобрана (совпал
    ``last_reply_hash``) или разговор завершён — выход без работы.
    Итоговый статус ставится всегда: «готово» при добытых данных,
    «не нашлось» при пустом походе; без похода статус не трогается.

    Args:
        state: задача с идентификатором звонка и материалом реплики.

    Returns:
        Пустой патч: результат живёт в кеше контекста, не в состоянии графа.
    """
    started = time.perf_counter()
    call_id = str(state.get("call_id") or "").strip()
    reply = str(state.get("reply") or "")
    if not call_id or not reply.strip():
        log.warning("Воркер контекстера: пустая задача, call_id=%r", call_id)
        return {}

    cached = await context_store.load(call_id)
    context = cached if cached is not None else ConversationContext()
    if context.conversation_ended:
        stage("contexter-worker", f"звонок {call_id}: разговор завершён, пропуск", "done")
        return {}
    if reply_hash(reply) == (context.last_reply_hash or ""):
        stage("contexter-worker", f"звонок {call_id}: реплика уже разобрана, пропуск", "done")
        return {}

    try:
        script = registry.get(
            str(state.get("script_id") or settings.script_id),
            str(state.get("script_version") or settings.script_version) or None,
        )
    except Exception as exc:  # noqa: BLE001
        log.warning("Воркер контекстера: скрипт не загрузился: %s", exc)
        return {}

    updated = await run_contexter(
        context,
        reply=reply,
        tools=build_context_tools(script),
        needs=list(state.get("needs") or []),
        step_needs=list(state.get("step_needs") or []),
        profile=dict(state.get("profile") or {}),
        objections=script.objections,
    )

    if updated.dynamic_status == DYN_SEARCHING:
        # Разбор завершён, а статус остался поисковым — доводим до итога,
        # чтобы генератор не ждал то, что уже разобрано.
        updated.dynamic_status = DYN_READY if updated.dynamic_text.strip() else DYN_MISSING
        updated.situation_slug = None

    base = await context_store.load(call_id)
    if base is not None:
        if base.conversation_ended:
            stage("contexter-worker", f"звонок {call_id}: завершён во время разбора", "done")
            return {}
        _keep_concurrent_dynamic(base, updated)
    merged = merge_context_fields(
        base if base is not None else updated,
        updated,
        CONTEXT_FIELDS_STATIC | CONTEXT_FIELDS_DYNAMIC,
    )
    await context_store.save(call_id, merged)

    elapsed_ms = int((time.perf_counter() - started) * 1000)
    stage(
        "contexter-worker",
        f"звонок {call_id}: статус {merged.dynamic_status}, {elapsed_ms} мс",
        "done",
    )
    return {}


def build_contexter_worker_graph() -> StateGraph:
    """Собирает граф фонового контекстера: один узел."""
    builder: StateGraph = StateGraph(ContexterTaskState)
    builder.add_node("contexter_task", contexter_task_node)
    builder.set_entry_point("contexter_task")
    builder.set_finish_point("contexter_task")
    return builder


graph = build_contexter_worker_graph().compile(name="vector_contexter")
