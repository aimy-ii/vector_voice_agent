r"""Служебный граф чекера в реальном времени.

Лайв-канал: закрывает шаги, дозаполняет базовый профиль, греет контекст
под предстоящий шаг. В ``messages`` не пишет, реплик в эфир не выдаёт.
Прогрев не на пути хода генератора — ошибка только в лог.

Политика запусков (на стороне клиента SDK, см. настройки)::

    vector_checker  → multitask_strategy="interrupt"
        новый служебный проход отменяет предыдущий незавершённый;

    vector_agent    → multitask_strategy="enqueue"
        основной ход не ждёт служебный: перед стартом клиент отменяет
        идущий ``vector_checker`` (или стартует с interrupt), иначе при
        enqueue сервер поставит основной ход в очередь за служебным.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from langgraph.graph import StateGraph
from langgraph.runtime import Runtime

from core.config import settings
from graph.checker import check_pass
from graph.context import context_from_state, merge_static
from graph.contexter import run_contexter
from graph.facts import needs_of
from graph.nodes import (
    _checker_client,
    _load_progress,
    _merge_profile,
    _save_progress,
)
from graph.profile_fill import fill_basic_profile
from graph.progress import stage
from graph.state import CallContext, CallState
from graph.tools_registry import build_context_tools
from kb.client import vector_kb
from script.planner import peek_next_step, pick_step
from script.price import price_line, price_line_from_kb
from script.source import registry
from script.store import PROGRESS_FIELDS_CHECKER

log = logging.getLogger(__name__)


def growth_below_threshold(reply: str, previous: str, *, min_growth: int) -> bool:
    """Прирост текста меньше порога внутри текущей реплики.

    Точку отсчёта при новой реплике сбрасывают снаружи по
    ``partial_utterance_id``; сюда доходит только прирост внутри одной
    реплики. Пустой ``previous`` — первый проход, порог не применяется.

    Args:
        reply: накопленный текст текущего прохода.
        previous: текст прошлого отработанного прохода в этой же реплике.
        min_growth: порог прироста в символах.

    Returns:
        True — модель звать не нужно.
    """
    if not previous:
        return False
    growth = len(reply) - len(previous)
    # Укорочение текста (переразметка ASR) — не пропуск по порогу.
    if growth < 0:
        return False
    return growth < min_growth


def is_new_utterance(
    utterance_id: str,
    last_utterance_id: str,
) -> bool:
    """Новая ли реплика по идентификатору от бота.

    Args:
        utterance_id: ``partial_utterance_id`` из полезной нагрузки.
        last_utterance_id: идентификатор, к которому относится точка отсчёта.

    Returns:
        True — точка отсчёта прироста должна быть сброшена.
    """
    if not utterance_id:
        return False
    return utterance_id != last_utterance_id


def _script_of(state: CallState):
    """Скомпилированный скрипт из реестра по полям состояния."""
    return registry.get(
        state.get("script_id") or settings.script_id,
        state.get("script_version") or settings.script_version,
    )


async def _warmup_next_step(
    state: CallState,
    *,
    progress,
    profile: dict[str, str],
    ctx,
    asks_inform: bool,
) -> Any:
    """Прогревает мету города / филиалы / цену под предстоящий шаг.

    Ошибки только в лог — ход лайв-канала не роняют.
    """
    script = _script_of(state)
    current_id = state.get("current_step")
    current = script.steps.get(current_id) if current_id else None
    try:
        if current is None:
            nxt = pick_step(
                script,
                status=progress.status,
                profile=profile,
                attempts=progress.attempts,
                inform_reason=asks_inform,
                pending_soft_cap=settings.pending_steps_soft_cap,
            )
        else:
            nxt = peek_next_step(
                script,
                current=current,
                status=progress.status,
                profile=profile,
                attempts=progress.attempts,
                inform_reason=asks_inform,
                pending_soft_cap=settings.pending_steps_soft_cap,
            )
    except Exception as exc:  # noqa: BLE001
        log.warning("Прогрев: peek_next_step не удался: %s", exc)
        return ctx
    if nxt is None:
        return ctx

    needs = set(needs_of(nxt))
    city_slug = state.get("city_slug") or ctx.city_slug or None

    try:
        if "city_choices" in needs:
            await vector_kb.list_cities()
        # Город из профиля ещё без слага — мета не греется, это ок.
        if not city_slug:
            return ctx

        want_city = ("city_meta" in needs or "price" in needs) and not ctx.city_slug
        want_price = "price" in needs
        want_branches = "branches" in needs
        if want_city or want_price or not ctx.city_faq:
            city_meta = await vector_kb.get_city(city_slug)
            if city_meta:
                city_name = (
                    state.get("city_name") or profile.get("city") or ctx.city_name or city_slug
                )
                price_phrase = None
                if want_price and city_meta.get("price") is not None:
                    if script.is_sales:
                        price_phrase = price_line_from_kb(city_meta.get("price"))
                    else:
                        price_phrase = price_line(city_meta.get("price"), script.params.price)
                ctx = merge_static(
                    ctx,
                    city_slug=city_slug,
                    city_name=str(city_name),
                    city_meta=city_meta,
                    price_line=price_phrase,
                )
        if want_branches:
            await vector_kb.list_branches(city_slug)
        if "branch_meta" in needs:
            branch_slug = state.get("branch_slug") or ctx.branch_slug
            if branch_slug:
                await vector_kb.get_branch(branch_slug)
    except Exception as exc:  # noqa: BLE001
        log.warning("Прогрев под шаг %s не удался: %s", nxt.id, exc)
    return ctx


async def live_check_node(state: CallState, runtime: Runtime[CallContext]) -> dict[str, Any]:
    """Один служебный проход чекера и контекстера по ``partial_reply``.

    Точка отсчёта сбрасывается при смене ``partial_utterance_id`` от бота —
    не по знаку прироста длины. Порог ``checker_min_growth_chars``
    применяется только к приросту внутри одной реплики относительно
    ``last_checked_partial``. Первый проход новой реплики порогом не режется.
    Иначе дозаполняет профиль, зовёт ``check_pass``, ``run_contexter``
    и прогрев под предстоящий шаг.
    """
    started = time.perf_counter()
    reply = str(state.get("partial_reply") or "")
    utterance_id = str(state.get("partial_utterance_id") or "")
    last_utterance_id = str(state.get("last_checked_utterance_id") or "")
    previous = str(state.get("last_checked_partial") or "")
    min_growth = settings.checker_min_growth_chars
    new_utterance = is_new_utterance(utterance_id, last_utterance_id)
    if new_utterance:
        previous = ""
        stage(
            "live-check",
            f"накоплено {len(reply)} симв., utterance {utterance_id} — "
            f"новая реплика, сброс точки отсчёта",
            "start",
        )
    growth = len(reply) - len(previous)
    if not new_utterance:
        stage(
            "live-check",
            f"накоплено {len(reply)} симв., прирост с прошлого прохода {growth} симв.",
            "start",
        )
    if growth_below_threshold(reply, previous, min_growth=min_growth):
        stage(
            "live-check",
            f"прирост {growth} < порога {min_growth}, пропуск",
            "skip",
        )
        return {}

    progress = await _load_progress(state)
    profile = _merge_profile(state, progress)

    # Фон: базовые поля из уже сказанного — до check_pass, чтобы fills закрылись.
    filled = fill_basic_profile(reply, profile)
    if filled:
        profile = {**profile, **filled}
        progress.profile = {**progress.profile, **filled}

    state_for_check: dict[str, Any] = {**state, "profile": profile}
    progress, closures, asks_inform = await check_pass(
        state_for_check,
        reply=reply,
        judge=_checker_client,
        progress=progress,
    )
    patch = await _save_progress(progress, fields=PROGRESS_FIELDS_CHECKER)
    patch["last_checked_partial"] = reply
    if utterance_id:
        patch["last_checked_utterance_id"] = utterance_id
    patch["client_asks_inform"] = asks_inform
    patch["profile"] = profile

    # Контекстер печёт вперёд, пока клиент говорит: справка/статус к ходу.
    script = _script_of(state)
    ctx = context_from_state(state.get("conversation_context"))
    ctx = await run_contexter(
        ctx,
        reply=reply,
        tools=build_context_tools(script),
        objections=script.objections,
    )

    # Прогрев под предстоящий шаг — не на пути хода, ошибки только в лог.
    ctx = await _warmup_next_step(
        state,
        progress=progress,
        profile=profile,
        ctx=ctx,
        asks_inform=asks_inform,
    )
    patch["conversation_context"] = ctx.model_dump()

    if closures:
        checker_text = "закрыл шаги " + ",".join(step_id for step_id, _ in closures)
    else:
        checker_text = "ничего"
    elapsed_ms = int((time.perf_counter() - started) * 1000)
    stage(
        "live-check",
        f"чекер: {checker_text}; контекстер: статус {ctx.dynamic_status}; {elapsed_ms} мс",
        "done",
    )
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
