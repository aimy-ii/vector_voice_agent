"""Фоновый граф добычи данных города.

Служебный проход живёт со стратегией interrupt: новая реплика человека
отменяет незавершённый проход вместе с походом в справочник. Этот граф
запускается отдельно со стратегией enqueue: задачи копятся в очереди и
доводятся до конца, отмена прохода их не касается.

Результат кладётся сразу в кеш контекста звонка (`context_store`), а не в
состояние графа: оттуда его читают и служебный проход, и основной ход.
"""

from __future__ import annotations

import logging
import time
from typing import Any, TypedDict

from langgraph.graph import StateGraph

from core.config import settings
from graph.context import ConversationContext, record_empty_needs
from graph.context_store import (
    CONTEXT_FIELDS_DYNAMIC,
    CONTEXT_FIELDS_STATIC,
    context_store,
    merge_context_fields,
)
from graph.progress import stage
from graph.tools_registry import build_context_tools
from script.build import CompiledScript
from script.source import registry

log = logging.getLogger(__name__)


class CityTaskState(TypedDict, total=False):
    """Состояние одной задачи на данные города.

    Attributes:
        call_id: идентификатор звонка; ключ контекста в кеше.
        probe: строка с названием города (из анкеты или реплики).
    """

    call_id: str
    probe: str


async def _load_script() -> CompiledScript | None:
    """Читает скомпилированный скрипт; ошибка — в лог, задача не падает."""
    try:
        return registry.get(settings.script_id, settings.script_version)
    except Exception as exc:  # noqa: BLE001
        log.warning("Фоновый город: скрипт не загрузился: %s", exc)
        return None


def _keep_concurrent_dynamic(base: ConversationContext, overlay: ConversationContext) -> None:
    """Не даёт локальной динамике затереть текст, появившийся в кеше параллельно.

    Args:
        base: свежий слепок кеша перед записью.
        overlay: локальный результат похода; ``dynamic_text`` дополняется
            чужим текстом на месте.
    """
    concurrent = (base.dynamic_text or "").strip()
    local = (overlay.dynamic_text or "").strip()
    if concurrent and concurrent not in (overlay.dynamic_text or ""):
        overlay.dynamic_text = f"{concurrent}\n{local}".strip() if local else concurrent


async def city_task_node(state: CityTaskState) -> dict[str, Any]:
    """Один поход за данными города до конца, с записью в кеш.

    Свежесть проверяется по кешу на входе: город уже добыт другим путём —
    выходим без похода. Успех пишет слаг и статику города; пустой ответ
    увеличивает счётчик попыток через ``record_empty_needs``.

    Args:
        state: задача с идентификатором звонка и строкой города.

    Returns:
        Пустой патч: результат живёт в кеше контекста, не в состоянии графа.
    """
    started = time.perf_counter()
    call_id = str(state.get("call_id") or "").strip()
    probe = str(state.get("probe") or "").strip()
    if not call_id or not probe:
        log.warning("Фоновый город: пустая задача, call_id=%r", call_id)
        return {}

    cached = await context_store.load(call_id)
    context = cached if cached is not None else ConversationContext()
    if (context.city_slug or "").strip():
        stage("city-worker", f"звонок {call_id}: город уже добыт, пропуск", "done")
        return {}
    if "city_choices" in {str(n).strip() for n in (context.empty_needs or [])}:
        stage("city-worker", f"звонок {call_id}: попытки исчерпаны, пропуск", "done")
        return {}

    script = await _load_script()
    if script is None:
        return {}
    tools = build_context_tools(script)
    city_tool = next((tool for tool in tools if tool.name == "city"), None)
    if city_tool is None:
        return {}

    updated = context.model_copy(deep=True)
    found = ""
    try:
        found = await city_tool.run(probe, updated, slugs=(), reply=probe)  # type: ignore[call-arg]
    except Exception as exc:  # noqa: BLE001
        log.warning("Фоновый город: инструмент не ответил: %s", exc)

    got = bool((found or "").strip()) or bool((updated.city_slug or "").strip())
    record_empty_needs(updated, ["city_choices"], found=got)
    if got and (found or "").strip() and found.strip() not in updated.dynamic_text:
        updated.dynamic_text = (updated.dynamic_text + "\n" + found.strip()).strip()

    base = await context_store.load(call_id)
    if base is not None:
        _keep_concurrent_dynamic(base, updated)
    merged = merge_context_fields(
        base if base is not None else updated,
        updated,
        CONTEXT_FIELDS_STATIC | CONTEXT_FIELDS_DYNAMIC,
    )
    await context_store.save(call_id, merged)

    elapsed_ms = int((time.perf_counter() - started) * 1000)
    stage(
        "city-worker",
        f"звонок {call_id}: город={'добыт' if got else 'пусто'}, {elapsed_ms} мс",
        "done",
    )
    return {}


def build_city_worker_graph() -> StateGraph:
    """Собирает граф фоновой добычи города: один узел."""
    builder: StateGraph = StateGraph(CityTaskState)
    builder.add_node("city_task", city_task_node)
    builder.set_entry_point("city_task")
    builder.set_finish_point("city_task")
    return builder


graph = build_city_worker_graph().compile(name="vector_city_worker")
