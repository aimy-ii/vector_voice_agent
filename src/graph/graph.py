"""Граф одного хода разговора.

::

    __start__
        │
     ingest            принять историю, поднять скрипт, сверить произнесённое
        │
     lookup            резолвер города/филиала и факты справочника
        │
      plan             шапка из кеша (закрытия лайв-канала), счётчик при взятии
        │
     respond           генератор
        │
      commit → __end__

Чекер живёт только в лайв-канале (``vector_checker``): основной ход его не
зовёт и не ждёт. Plan читает то, что лайв успел закрыть в Redis.

Граф компилируется без чекпоинтера: чекпоинтер серверный.
``interrupt()`` не используется. Режим потока ``custom`` строкой.
"""

from __future__ import annotations

from langgraph.graph import StateGraph

from graph.nodes import (
    commit_node,
    ingest_node,
    lookup_node,
    plan_node,
    respond_node,
)
from graph.state import CallContext, CallState


def build_graph() -> StateGraph:
    """Собирает граф хода.

    Returns:
        Незакомпилированный граф.
    """
    builder: StateGraph = StateGraph(CallState, context_schema=CallContext)

    builder.add_node("ingest", ingest_node)
    builder.add_node("lookup", lookup_node)
    builder.add_node("plan", plan_node)
    builder.add_node("respond", respond_node)
    builder.add_node("commit", commit_node)

    builder.set_entry_point("ingest")
    builder.add_edge("ingest", "lookup")
    builder.add_edge("lookup", "plan")
    builder.add_edge("plan", "respond")
    builder.add_edge("respond", "commit")
    builder.set_finish_point("commit")
    return builder


#: Точка входа для LangGraph Server (см. `langgraph.json`).
graph = build_graph().compile(name="vector_agent")
