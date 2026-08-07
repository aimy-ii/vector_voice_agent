r"""Служебный граф чекера в реальном времени.

Лайв-канал: закрывает шаги, зовёт контекстер за данными, разбирает профиль,
решает конец разговора, греет контекст под предстоящий шаг. В ``messages``
не пишет, реплик в эфир не выдаёт. Ошибка только в лог — ход генератора
не роняет.

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
from graph.context import DYN_NONE, DYN_SEARCHING, DYN_WORKING, merge_static
from graph.context_store import CONTEXT_FIELDS_DYNAMIC, CONTEXT_FIELDS_STATIC
from graph.contexter import reply_hash, run_contexter
from graph.facts import knowledge_of, needs_of
from graph.farewell_agent import decide_farewell
from graph.log_fmt import format_check_done, format_live_check_state
from graph.nearby import (
    apply_result,
    format_searching,
    is_searching,
    lookup_nearby,
    should_refresh,
)
from graph.nodes import (
    _call_id,
    _checker_client,
    _lead_from_progress,
    _load_context,
    _load_progress,
    _merge_profile,
    _save_context,
    _save_progress,
)
from graph.profile_agent import guess_profile, profile_fields_of
from graph.profile_form import rewritable_keys
from graph.progress import stage
from graph.state import CallContext, CallState
from graph.tools_registry import build_context_tools
from graph.transcript import to_messages
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


def _script_of_state(state: CallState):
    """Скомпилированный скрипт из реестра по полям состояния."""
    return registry.get(
        state.get("script_id") or settings.script_id,
        state.get("script_version") or settings.script_version,
    )


def _head_needs(
    state: CallState,
    *,
    progress,
    profile: dict[str, str],
) -> list[str]:
    """Потребности справочника по шапке хода (``needs_of``).

    Args:
        state: состояние звонка.
        progress: прогресс из кеша.
        profile: слитый профиль.

    Returns:
        Уникальный список потребностей в порядке шагов шапки.
    """
    head, _step = _lead_from_progress(state, progress=progress, profile=profile)
    needs: list[str] = []
    for head_step in head:
        for need in needs_of(head_step):
            if need not in needs:
                needs.append(need)
    return needs


def _lead_knowledge(
    state: CallState,
    *,
    progress,
    profile: dict[str, str],
) -> list[str]:
    """Потребности ведущего шага для агента контекста (``knowledge``).

    Тот же разбор ведущего шага, что у прогрева через
    ``_lead_from_progress``: агенту нужны строки скрипта, а не ключи
    справочника.

    Args:
        state: состояние звонка.
        progress: прогресс из кеша.
        profile: слитый профиль.

    Returns:
        Строки ``knowledge`` ведущего шага; пусто — шага нет или список пуст.
    """
    _head, lead = _lead_from_progress(state, progress=progress, profile=profile)
    return knowledge_of(lead)


async def _warmup_next_step(
    state: CallState,
    *,
    progress,
    profile: dict[str, str],
    ctx,
    asks_inform: bool,
) -> Any:
    """Прогревает мету города / филиалы / цену под предстоящий шаг.

    В лайв-треде ``current_step`` обычно нет — ведущий шаг берём из
    прогресса через ``_lead_from_progress``, чтобы греть *следующий*
    шаг, а не тот, что бот уже произносит. Ошибки только в лог —
    ход лайв-канала не роняют.
    """
    script = _script_of_state(state)
    current_id = state.get("current_step")
    current = script.steps.get(current_id) if current_id else None
    try:
        if current is None:
            _head, current = _lead_from_progress(state, progress=progress, profile=profile)
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
        if not city_slug:
            return ctx

        want_city = ("city_meta" in needs or "price" in needs) and not ctx.city_slug
        want_price = "price" in needs
        want_branches = "branches" in needs
        if (
            want_city
            or (want_price and not ctx.city_slug)
            or (not ctx.city_faq and not ctx.city_slug)
        ):
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
    """Один служебный проход: контекстер → чекер → профиль → прощание → прогрев.

    Первым делом ставит статус «в работе» с хешем реплики — ход видит,
    что фон уже взял реплику, ещё до профиля и контекстера. Контекстер
    дальше сменит статус на «в поиске» / итог. Любой выход, в том числе
    по исключению, не оставляет «в работе»: иначе ход тянет заглушки
    до конца звонка.

    Точка отсчёта сбрасывается при смене ``partial_utterance_id``;
    порог прироста — только внутри одной реплики.
    """
    started = time.perf_counter()
    reply = str(state.get("partial_reply") or "")
    utterance_id = str(state.get("partial_utterance_id") or "")
    last_utterance_id = str(state.get("last_checked_utterance_id") or "")
    previous = str(state.get("last_checked_partial") or "")
    is_final = bool(state.get("partial_is_final"))
    min_growth = settings.checker_min_growth_chars
    call_id = _call_id()
    new_utterance = is_new_utterance(utterance_id, last_utterance_id)
    if new_utterance:
        previous = ""
        stage(
            "live-check",
            f"накоплено {len(reply)} симв., utterance {utterance_id} — "
            f"новая реплика, сброс точки отсчёта, звонок {call_id}",
            "start",
        )
    growth = len(reply) - len(previous)
    if not new_utterance:
        kind = "финал" if is_final else "прирост"
        stage(
            "live-check",
            f"накоплено {len(reply)} симв., {kind} с прошлого прохода {growth} симв., "
            f"звонок {call_id}",
            "start",
        )
    if not is_final and growth_below_threshold(reply, previous, min_growth=min_growth):
        stage(
            "live-check",
            f"прирост {growth} < порога {min_growth}, пропуск",
            "skip",
        )
        return {}

    turn = int(state.get("turn") or 0)
    digest = reply_hash(reply) if reply else ""
    ctx = await _load_context(state)
    prior_status = ctx.dynamic_status or DYN_NONE
    if prior_status in (DYN_WORKING, DYN_SEARCHING):
        prior_status = DYN_NONE
    ctx = ctx.model_copy(
        update={
            "dynamic_status": DYN_WORKING,
            "dynamic_reply_hash": digest,
            "dynamic_turn": turn,
            "situation_slug": None,
            "filler_spoken": False,
        }
    )
    await _save_context(ctx, fields=CONTEXT_FIELDS_DYNAMIC)

    patch: dict[str, Any] = {}
    farewell_note = "не вызывался"
    try:
        progress = await _load_progress(state)
        profile = _merge_profile(state, progress)
        stage(
            "live-check",
            format_live_check_state(
                attempts=progress.attempts,
                status=progress.status,
                profile=profile,
            ),
            "state",
        )

        script = _script_of_state(state)
        needs = _head_needs(state, progress=progress, profile=profile)
        step_needs = _lead_knowledge(state, progress=progress, profile=profile)

        # Контекстер до check_pass: сам выставит статус и сходит за данными.
        ctx = await run_contexter(
            ctx,
            reply=reply,
            tools=build_context_tools(script),
            needs=needs,
            step_needs=step_needs,
            profile=profile,
            objections=script.objections,
        )

        # История из кеша: у фона своего снимка нет. После контекстера
        # подтягиваем свежую — основной ход мог дописать реплики.
        ctx = ctx.model_copy(update={"transcript": (await _load_context(state)).transcript})
        history = to_messages(ctx.transcript) or list(state.get("messages") or [])
        state_for_check: dict[str, Any] = {
            **state,
            "profile": profile,
            "messages": history,
        }
        progress, closures, asks_inform = await check_pass(
            state_for_check,
            reply=reply,
            judge=_checker_client,
            progress=progress,
        )
        patch = {
            "last_checked_partial": reply,
            "client_asks_inform": asks_inform,
        }
        if utterance_id:
            patch["last_checked_utterance_id"] = utterance_id

        # Слаги из статики контекстера — в патч состояния (не в форму профиля).
        if ctx.city_slug and not state.get("city_slug"):
            patch["city_slug"] = ctx.city_slug
            if ctx.city_name:
                patch["city_name"] = ctx.city_name
        if ctx.branch_slug and not state.get("branch_slug"):
            patch["branch_slug"] = ctx.branch_slug

        fields = profile_fields_of(script)
        rewritable = rewritable_keys()
        guess = await guess_profile(
            reply,
            history=history,
            known=profile,
            fields=fields,
            rewritable=rewritable,
        )
        for item in guess.values:
            key = item.key
            value = item.value
            if not value:
                continue
            current = str(profile.get(key) or "").strip()
            if current and (key not in rewritable or current == value.strip()):
                continue
            profile[key] = value
        progress.profile = dict(profile)
        progress_patch = await _save_progress(progress, fields=PROGRESS_FIELDS_CHECKER)
        patch.update(progress_patch)
        patch["profile"] = profile

        turn_kind = str(state.get("turn_kind") or "client").strip().lower()
        no_client_reply = turn_kind in {"continuation", "silence"}
        if no_client_reply:
            farewell_note = "пропуск: нет реплики человека"
        elif len(history) < settings.farewell_min_messages:
            farewell_note = (
                f"пропуск: реплик {len(history)} < порога {settings.farewell_min_messages}"
            )
        elif not reply.strip():
            farewell_note = "пропуск: пустая реплика"
        else:
            decision = await decide_farewell(reply, history=history)
            if decision is None:
                farewell_note = "ошибка агента, флаг не тронут"
            else:
                ended = bool(decision.conversation_ended)
                ctx = ctx.model_copy(update={"conversation_ended": ended})
                farewell_note = f"закончен={ended}"

        ctx = await _warmup_next_step(
            state,
            progress=progress,
            profile=profile,
            ctx=ctx,
            asks_inform=asks_inform,
        )

        # Подбор ближайших филиалов: решение принимает код по изменению поля
        # формы, а не агент. Модель поставляет только само место словами.
        nearby_place = str(profile.get("location_hint") or "").strip()
        nearby_city = (ctx.city_slug or str(state.get("city_slug") or "")).strip()
        nearby_key_new = should_refresh(
            city_slug=nearby_city or None,
            place=nearby_place,
            current_key=ctx.nearby_key,
        )
        if nearby_key_new:
            previous_nearby_text = ctx.nearby_text
            previous_nearby_found = ctx.nearby_found
            ctx.nearby_key = nearby_key_new
            ctx.nearby_text = format_searching(nearby_place)
            # Отдельной записью до похода: ход идёт параллельно и должен
            # увидеть, что подбор начат, а не пустоту.
            await _save_context(ctx, fields=CONTEXT_FIELDS_DYNAMIC)
            nearby_result = await lookup_nearby(
                vector_kb,
                city_slug=nearby_city,
                place=nearby_place,
                key=nearby_key_new,
            )
            ctx.nearby_text, ctx.nearby_found = apply_result(
                previous_text=previous_nearby_text,
                previous_found=previous_nearby_found,
                result=nearby_result,
            )
            if nearby_result.branch_slugs:
                ctx.branch_candidates = nearby_result.branch_slugs

        ctx_patch = await _save_context(ctx, fields=CONTEXT_FIELDS_STATIC | CONTEXT_FIELDS_DYNAMIC)
        patch.update(ctx_patch)

        if closures:
            checker_text = format_check_done(closures)
        else:
            checker_text = "ничего"
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        subject = (ctx.situation_slug or "").strip()
        subject_part = f", предмет «{subject}»" if subject else ""
        stage(
            "live-check",
            f"чекер: {checker_text}; контекстер: статус {ctx.dynamic_status}"
            f"{subject_part}; прощание: {farewell_note}; {elapsed_ms} мс",
            "done",
        )
        return patch
    finally:
        final_ctx = await _load_context(state)
        updates: dict[str, Any] = {}
        if final_ctx.dynamic_status == DYN_WORKING:
            updates["dynamic_status"] = prior_status if prior_status != DYN_WORKING else DYN_NONE
        # Проход оборвался на исключении посреди подбора: строку о подборе
        # снимаем, ключ чистим — следующая реплика пересчитает то же место.
        if is_searching(final_ctx.nearby_text):
            updates["nearby_text"] = ""
            updates["nearby_key"] = ""
        if updates:
            final_ctx = final_ctx.model_copy(update=updates)
            fixed = await _save_context(final_ctx, fields=CONTEXT_FIELDS_DYNAMIC)
            patch.update(fixed)


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
