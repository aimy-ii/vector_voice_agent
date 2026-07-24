"""Граф одного хода разговора.

::

    __start__
        │
     ingest            принять историю, поднять скрипт, сверить произнесённое
        │
      check            чекер: закрыть шаги по счётчику и диалогу
        │
      plan             шапка шагов, счётчик при взятии, маршрут
        ├──────────────────────────────► verbatim ─┐
        ├──► lookup ──► respond ──┐                │
        │                         │                │
        └──► respond ─────────────┤                │
                                  ├──► verbatim ───┤
                                  └────────────────┴──► commit → __end__

Граф компилируется без чекпоинтера: чекпоинтер серверный.
``interrupt()`` не используется. Режим потока ``custom`` строкой.
"""

from __future__ import annotations

from typing import Literal

from langgraph.graph import StateGraph

from graph.nodes import (
    ROUTE_LOOKUP,
    ROUTE_VERBATIM,
    check_node,
    commit_node,
    ingest_node,
    lookup_node,
    plan_node,
    respond_node,
    verbatim_node,
)
from graph.state import CallContext, CallState
from script.source import registry


def _after_plan(state: CallState) -> Literal["lookup", "respond", "verbatim"]:
    """Куда идти после планирования."""
    route = state.get("route")
    if route == ROUTE_LOOKUP:
        return "lookup"
    if route == ROUTE_VERBATIM:
        return "verbatim"
    return "respond"


def _after_lookup(state: CallState) -> Literal["respond", "verbatim"]:
    """После похода за данными: к модели или сразу в эфир."""
    return "verbatim" if state.get("skip_model") else "respond"


def _after_respond(state: CallState) -> Literal["verbatim", "commit"]:
    """Нужно ли после реплики модели дочитать дословный блок шага."""
    step_id = state.get("current_step")
    if not step_id:
        return "commit"

    script = registry.get(
        state.get("script_id") or "",
        state.get("script_version"),
    )
    step = script.steps.get(step_id)
    if step is None or not step.verbatim:
        return "commit"

    result = state.get("turn_result") or {}
    if result and not result.get("resume_step", True):
        return "commit"
    return "verbatim"


def build_graph() -> StateGraph:
    """Собирает граф хода.

    Returns:
        Незакомпилированный граф.
    """
    builder: StateGraph = StateGraph(CallState, context_schema=CallContext)

    builder.add_node("ingest", ingest_node)
    builder.add_node("check", check_node)
    builder.add_node("plan", plan_node)
    builder.add_node("lookup", lookup_node)
    builder.add_node("respond", respond_node)
    builder.add_node("verbatim", verbatim_node)
    builder.add_node("commit", commit_node)

    builder.set_entry_point("ingest")
    builder.add_edge("ingest", "check")
    builder.add_edge("check", "plan")
    builder.add_conditional_edges(
        "plan",
        _after_plan,
        {"lookup": "lookup", "respond": "respond", "verbatim": "verbatim"},
    )
    builder.add_conditional_edges(
        "lookup",
        _after_lookup,
        {"respond": "respond", "verbatim": "verbatim"},
    )
    builder.add_conditional_edges(
        "respond",
        _after_respond,
        {"verbatim": "verbatim", "commit": "commit"},
    )
    builder.add_edge("verbatim", "commit")
    builder.set_finish_point("commit")
    return builder


#: Точка входа для LangGraph Server (см. `langgraph.json`).
graph = build_graph().compile(name="vector_agent")
