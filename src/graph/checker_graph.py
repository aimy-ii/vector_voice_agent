r"""Служебный граф чекера в реальном времени.

Вторая точка входа на том же состоянии и треде, что основной ход.
Принимает накопленный распознанный текст (``partial_reply``), гоняет
``check_pass``, пишет обратно только статусы/счётчики закрытых шагов.
В ``messages`` не пишет, реплик в эфир не выдаёт.

Политика запусков (на стороне клиента SDK, см. настройки)::

    vector_checker  → multitask_strategy="interrupt"
        новый служебный проход отменяет предыдущий незавершённый;

    vector_agent    → multitask_strategy="enqueue"
        основной ход не ждёт служебный: перед стартом клиент отменяет
        идущий ``vector_checker`` (или стартует с interrupt), иначе при
        enqueue сервер поставит основной ход в очередь за служебным.
"""

from __future__ import annotations

from typing import Any

from langgraph.graph import StateGraph
from langgraph.runtime import Runtime

from core.config import settings
from graph.checker import check_pass
from graph.log_fmt import format_check_done
from graph.nodes import _checker_client, _load_progress, _save_progress
from graph.progress import stage
from graph.state import CallContext, CallState


def growth_below_threshold(reply: str, previous: str, *, min_growth: int) -> bool:
    """Прирост текста меньше порога и это не первый проход.

    Args:
        reply: накопленный текст текущего прохода.
        previous: текст прошлого отработанного прохода.
        min_growth: порог прироста в символах.

    Returns:
        True — модель звать не нужно.
    """
    if not previous:
        return False
    return len(reply) - len(previous) < min_growth


async def live_check_node(state: CallState, runtime: Runtime[CallContext]) -> dict[str, Any]:
    """Один служебный проход чекера по ``partial_reply``.

    При приросте меньше порога (и не первый проход) тихо выходит без
    вызова модели. Иначе зовёт ``check_pass`` и сохраняет прогресс.
    """
    reply = str(state.get("partial_reply") or "")
    previous = str(state.get("last_checked_partial") or "")
    if growth_below_threshold(
        reply,
        previous,
        min_growth=settings.checker_min_growth_chars,
    ):
        stage("live_check", "прирост ниже порога — пропуск", "done")
        return {}

    progress = await _load_progress(state)
    progress, closures = await check_pass(
        state,
        reply=reply,
        judge=_checker_client,
        progress=progress,
    )
    patch = await _save_progress(progress)
    patch["last_checked_partial"] = reply
    stage("live_check", format_check_done(closures), "done")
    return patch


def build_checker_graph() -> StateGraph:
    """Собирает служебный граф чекера.

    Returns:
        Незакомпилированный граф из одного узла.
    """
    builder: StateGraph = StateGraph(CallState, context_schema=CallContext)
    builder.add_node("live_check", live_check_node)
    builder.set_entry_point("live_check")
    builder.set_finish_point("live_check")
    return builder


#: Точка входа для LangGraph Server (см. `langgraph.json`).
graph = build_checker_graph().compile(name="vector_checker")
