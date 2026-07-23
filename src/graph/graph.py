"""Граф одного хода разговора.

::

    __start__
        │
     ingest            принять историю, поднять скрипт, сверить произнесённое
        │
      plan             КОД выбирает шаг и маршрут; модель здесь не участвует
        ├──────────────────────────────► verbatim ─┐   (клиент поддакнул,
        │                                          │    блок без модели)
        ├──► lookup ──► respond ──┐                │
        │                         │                │
        └──► respond ─────────────┤                │
                                  ├──► verbatim ───┤   (ответили на вопрос,
                                  │                │    затем дословный блок)
                                  └────────────────┴──► commit
                                                          │
                                                       __end__

Ключевое:

* у каждого условного перехода **явная карта назначений** — иначе Studio не
  нарисует рёбра, а роутер может уехать в несуществующий узел;
* `lookup` всегда стоит перед `respond`, а не после: справочник опрашивается
  до модели, поэтому фактам неоткуда взяться выдуманными;
* `verbatim` достижим двумя путями — напрямую из `plan`, когда разбирать
  нечего, и после `respond`, когда сначала надо ответить на посторонний
  вопрос и вернуться на место;
* `commit` — единственный выход, поэтому состояние обновляется ровно один раз
  за ход, каким бы путём тот ни прошёл.

Граф компилируется без чекпоинтера: чекпоинтер серверный, в postgres.
`interrupt()` не используется — граф отрабатывает ход и умирает.
"""

from __future__ import annotations

from typing import Literal

from langgraph.graph import StateGraph

from graph.nodes import (
    ROUTE_LOOKUP,
    ROUTE_VERBATIM,
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
    """Куда идти после планирования: за данными, к модели или сразу в эфир."""
    route = state.get("route")
    if route == ROUTE_LOOKUP:
        return "lookup"
    if route == ROUTE_VERBATIM:
        return "verbatim"
    return "respond"


def _after_lookup(state: CallState) -> Literal["respond", "verbatim"]:
    """После похода за данными: к модели или сразу выталкивать дословный блок.

    Данные шагу могут понадобиться и тогда, когда модель не нужна вовсе —
    например, дословный блок о цене подставляет сумму из справочника. Поэтому
    решает признак `skip_model`, а не маршрут.
    """
    return "verbatim" if state.get("skip_model") else "respond"


def _after_respond(state: CallState) -> Literal["verbatim", "commit"]:
    """Нужно ли после реплики модели дочитать дословный блок шага.

    Дословный блок произносится в том же ходу после ответа на посторонний
    вопрос — справки возвращают разговор на место. Если модель сочла возврат
    неуместным, блок ждёт следующего хода.
    """
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
        Незакомпилированный граф — на случай, если его понадобится встроить.
    """
    builder: StateGraph = StateGraph(CallState, context_schema=CallContext)

    builder.add_node("ingest", ingest_node)
    builder.add_node("plan", plan_node)
    builder.add_node("lookup", lookup_node)
    builder.add_node("respond", respond_node)
    builder.add_node("verbatim", verbatim_node)
    builder.add_node("commit", commit_node)

    builder.set_entry_point("ingest")
    builder.add_edge("ingest", "plan")
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
