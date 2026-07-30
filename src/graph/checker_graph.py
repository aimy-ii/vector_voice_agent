r"""Служебный граф чекера в реальном времени.

Лайв-канал: закрывает шаги, разбирает город и филиал, собирает факты,
греет контекст под предстоящий шаг. В ``messages`` не пишет, реплик в эфир
не выдаёт. Ошибка только в лог — ход генератора не роняет.

Политика запусков (на стороне клиента SDK, см. настройки)::

    vector_checker  → multitask_strategy="interrupt"
        новый служебный проход отменяет предыдущий незавершённый;

    vector_agent    → multitask_strategy="enqueue"
        основной ход не ждёт служебный: перед стартом клиент отменяет
        идущий ``vector_checker`` (или стартует с interrupt), иначе при
        enqueue сервер поставит основной ход в очередь за служебным.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from typing import Any

from langgraph.graph import StateGraph
from langgraph.runtime import Runtime

from core.config import settings
from graph.checker import check_pass
from graph.context import (
    DYN_MISSING,
    DYN_NONE,
    DYN_READY,
    DYN_SEARCHING,
    ConversationContext,
    merge_static,
)
from graph.context_store import CONTEXT_FIELDS_DYNAMIC, CONTEXT_FIELDS_STATIC
from graph.contexter import run_contexter
from graph.facts import collect_facts, needs_of
from graph.log_fmt import format_live_check_state, format_lookup_done
from graph.nodes import (
    _branch_resolver,
    _call_id,
    _checker_client,
    _city_resolver,
    _field_step_attempts,
    _lead_from_progress,
    _load_context,
    _load_progress,
    _merge_profile,
    _price_phrase_for,
    _save_context,
    _save_progress,
    _script_of,
    _step_fills_branch,
    _step_fills_city,
    _step_needs_lookup,
)
from graph.progress import stage
from graph.resolvers import resolve_branch, resolve_city
from graph.state import CallContext, CallState
from graph.tools_registry import build_context_tools
from kb.client import vector_kb
from script.planner import peek_next_step, pick_step
from script.price import price_line, price_line_from_kb
from script.source import registry
from script.store import PROGRESS_FIELDS_CHECKER, ScriptProgress

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


def _facts_to_dynamic(facts: dict[str, Any]) -> str:
    """Сериализует факты хода в текст для динамики контекста.

    Args:
        facts: факты, собранные из справочника.

    Returns:
        Текст для ``dynamic_text`` или пустая строка.
    """
    payload = {
        key: value
        for key, value in facts.items()
        if value not in (None, "", [], {}) and key not in {"city_choices", "branches_total"}
    }
    if not payload:
        return ""
    return "Факты справочника:\n" + json.dumps(payload, ensure_ascii=False, indent=1)


@dataclass(frozen=True)
class _LiveLookupIntent:
    """План разбора справочника на проходе лайв-канала."""

    will_search: bool
    pending: tuple[str, ...]
    needs_lookup: bool
    need_city: bool
    need_branch: bool
    needs: tuple[str, ...]
    search_text: str | None
    existing_slug: str | None
    fills_city: bool
    fills_branch: bool
    city_asked: bool
    branch_asked: bool


def _plan_live_lookup(
    state: CallState,
    *,
    reply: str,
    progress: ScriptProgress,
    profile: dict[str, str],
    ctx: ConversationContext,
) -> _LiveLookupIntent:
    """Решает, предстоит ли разбор города, филиала или фактов.

    Те же условия, что использует ``_lookup_in_live``: ранняя пометка
    ``DYN_SEARCHING`` и сам разбор не должны разъехаться.

    Args:
        state: состояние звонка.
        reply: накопленная реплика клиента.
        progress: прогресс из кеша.
        profile: слитый профиль.
        ctx: текущий контекст.

    Returns:
        План: флаг похода, pending-поля и промежуточные признаки.
    """
    script = _script_of(state)
    head, step = _lead_from_progress(state, progress=progress, profile=profile)
    profile_city = str(profile.get("city") or "").strip()

    def _asked(field: str) -> bool:
        """Шаг, заполняющий поле, уже задавался или взят в шапку этого хода."""
        if _field_step_attempts(script, progress, field) > 0:
            return True
        if field == "city":
            return any(_step_fills_city(s) for s in head)
        if field == "branch":
            return any(_step_fills_branch(s) for s in head)
        return False

    city_asked = _asked("city")
    branch_asked = _asked("branch")
    needs_lookup = _step_needs_lookup(
        head, state, city_asked=city_asked, branch_asked=branch_asked
    ) or bool(city_asked and profile_city and not (state.get("city_slug") or ctx.city_slug))

    needs: list[str] = []
    for head_step in head:
        for need in needs_of(head_step):
            if need not in needs:
                needs.append(need)

    fills_city = _step_fills_city(step)
    if city_asked and fills_city and reply:
        search_text: str | None = reply
    elif city_asked and profile_city:
        search_text = profile_city
    else:
        search_text = None
    existing_slug = state.get("city_slug") or ctx.city_slug
    need_city = bool(not existing_slug and search_text)

    fills_branch = any(_step_fills_branch(s) for s in head)
    need_branch = bool(
        branch_asked
        and existing_slug
        and not (state.get("branch_slug") or ctx.branch_slug)
        and fills_branch
        and reply
    )

    pending: list[str] = []
    if need_city:
        pending.append("city")
    if need_branch:
        pending.append("branch")
    # Факты/цена тоже требуют похода — статус «в поиске», даже без города/филиала.
    will_search = needs_lookup and (
        need_city or need_branch or bool(needs) or bool(city_asked and profile_city)
    )
    return _LiveLookupIntent(
        will_search=bool(will_search or need_city or need_branch),
        pending=tuple(pending),
        needs_lookup=needs_lookup,
        need_city=need_city,
        need_branch=need_branch,
        needs=tuple(needs),
        search_text=search_text,
        existing_slug=existing_slug,
        fills_city=fills_city,
        fills_branch=fills_branch,
        city_asked=city_asked,
        branch_asked=branch_asked,
    )


async def _lookup_in_live(
    state: CallState,
    *,
    reply: str,
    progress,
    profile: dict[str, str],
    ctx: ConversationContext,
) -> tuple[ConversationContext, dict[str, Any], dict[str, str]]:
    """Разбор города, филиала и фактов — бывший ``_lookup_body`` основного хода.

    Пометку ``DYN_SEARCHING`` ставит ``live_check_node`` до вызова модели;
    здесь по итогу — ``DYN_READY`` или ``DYN_MISSING``. Ошибки справочника
    не роняют узел.

    Args:
        state: состояние звонка.
        reply: накопленная реплика клиента.
        progress: прогресс из кеша.
        profile: слитый профиль.
        ctx: текущий контекст.

    Returns:
        Обновлённый контекст, патч состояния (слаги, кандидаты, журнал) и профиль.
    """
    script = _script_of(state)
    turn = int(state.get("turn") or 0)
    patch: dict[str, Any] = {}
    journal: list[dict[str, Any]] = list(state.get("tool_log") or [])
    turn_calls: list[dict[str, Any]] = []
    facts: dict[str, Any] = {}

    intent = _plan_live_lookup(state, reply=reply, progress=progress, profile=profile, ctx=ctx)
    needs_lookup = intent.needs_lookup
    need_city = intent.need_city
    need_branch = intent.need_branch
    needs = list(intent.needs)
    search_text = intent.search_text
    existing_slug = intent.existing_slug
    fills_city = intent.fills_city
    fills_branch = intent.fills_branch
    pending = list(intent.pending)
    branch_asked = intent.branch_asked

    if not needs_lookup:
        stage("lookup", "нечего искать, пропуск", "done")
        if pending:
            ctx = ctx.model_copy(update={"pending_fields": [], "dynamic_status": DYN_READY})
            await _save_context(ctx, fields=CONTEXT_FIELDS_DYNAMIC)
        return ctx, patch, profile

    def _note(entry: dict[str, Any]) -> None:
        journal.append(entry)
        turn_calls.append(entry)

    try:
        if not existing_slug and search_text:
            cities = await vector_kb.list_cities()
            _note({"call": "list_cities", "found": len(cities)})
            resolution = await resolve_city(search_text, cities, resolver=_city_resolver)
            _note(
                {
                    "call": "resolve_city",
                    "slug": resolution.slug,
                    "is_district": resolution.is_district,
                }
            )
            if resolution.is_district:
                facts["city_note"] = (
                    "Клиент назвал район внутри города, а не город сети. "
                    "Уточни город обучения, район городом не записывай."
                )
                stage("city", "слаг —, имя —, район=True", "done")
            elif resolution.slug and resolution.name:
                patch["city_slug"] = resolution.slug
                patch["city_name"] = resolution.name
                profile["city"] = resolution.name
                city_meta = await vector_kb.get_city(resolution.slug)
                _note({"call": "get_city", "slug": resolution.slug, "ok": city_meta is not None})
                price_phrase = None
                if city_meta and city_meta.get("price") is not None:
                    price_phrase = _price_phrase_for(script, city_meta.get("price"))
                if city_meta:
                    ctx = merge_static(
                        ctx,
                        city_slug=resolution.slug,
                        city_name=resolution.name,
                        city_meta=city_meta,
                        price_line=price_phrase,
                    )
                    if price_phrase:
                        facts["price_line"] = price_phrase
                stage(
                    "city",
                    f"слаг {resolution.slug}, имя {resolution.name}, район=False",
                    "done",
                )
            else:
                stage("city", "не распознан", "miss")
        elif existing_slug and search_text and fills_city:
            log.warning(
                "Резолвер города: слаг уже есть (%s), вызов пропущен — такого быть не должно",
                existing_slug,
            )

        city_slug = patch.get("city_slug") or state.get("city_slug") or ctx.city_slug

        if (
            branch_asked
            and city_slug
            and not (state.get("branch_slug") or ctx.branch_slug)
            and fills_branch
            and reply
        ):
            branches = await vector_kb.list_branches(city_slug)
            _note({"call": "list_branches", "slug": city_slug, "found": len(branches)})
            resolution = await resolve_branch(reply, branches, resolver=_branch_resolver)
            _note(
                {
                    "call": "resolve_branch",
                    "slugs": resolution.slugs,
                    "selected": resolution.selected,
                }
            )
            by_slug = {str(b.get("slug")): b for b in branches if b.get("slug")}
            candidates = [by_slug[s] for s in resolution.slugs if s in by_slug]
            if candidates:
                facts["branch_options"] = [
                    {
                        "slug": c.get("slug"),
                        "address": c.get("address"),
                        **({"landmark": c["landmark"]} if c.get("landmark") else {}),
                    }
                    for c in candidates
                ]
                patch["branch_candidates"] = [str(c.get("slug")) for c in candidates]
            if resolution.selected and resolution.selected in by_slug:
                branch_meta = await vector_kb.get_branch(resolution.selected)
                _note(
                    {
                        "call": "get_branch",
                        "slug": resolution.selected,
                        "ok": branch_meta is not None,
                    }
                )
                if branch_meta:
                    patch["branch_slug"] = resolution.selected
                    profile["branch"] = resolution.selected
                    ctx = merge_static(
                        ctx,
                        branch_slug=resolution.selected,
                        branch_meta=branch_meta,
                    )

        extra_needs = [n for n in needs if n != "branches" or "branch_options" not in facts]

        if city_slug and extra_needs:
            more, more_journal = await collect_facts(
                vector_kb,
                script=script,
                needs=extra_needs,
                city_slug=city_slug,
                branch_slug=patch.get("branch_slug") or state.get("branch_slug"),
                want_city_choices=False,
            )
            facts.update(more)
            journal.extend(more_journal)
            turn_calls.extend(more_journal)
            if city_slug and not ctx.city_slug and more.get("city"):
                city_name = state.get("city_name") or profile.get("city") or city_slug
                raw_meta = await vector_kb.get_city(city_slug)
                if raw_meta:
                    ctx = merge_static(
                        ctx,
                        city_slug=city_slug,
                        city_name=str(city_name),
                        city_meta=raw_meta,
                        price_line=more.get("price_line"),
                    )

        facts_text = _facts_to_dynamic(facts)
        if facts_text:
            dynamic = (ctx.dynamic_text + "\n" + facts_text).strip()
            ctx = ctx.model_copy(
                update={
                    "dynamic_text": dynamic,
                    "dynamic_status": DYN_READY,
                    "pending_fields": [],
                    "dynamic_turn": turn,
                }
            )
        elif turn_calls:
            # Ходили в справочник, но полезных фактов нет.
            ctx = ctx.model_copy(
                update={
                    "dynamic_status": DYN_MISSING if (need_city or need_branch) else DYN_READY,
                    "pending_fields": [],
                    "dynamic_turn": turn,
                }
            )
        else:
            ctx = ctx.model_copy(
                update={
                    "dynamic_status": DYN_READY,
                    "pending_fields": [],
                    "dynamic_turn": turn,
                }
            )

        # Если искали город/филиал и ничего не зафиксировали — «не нашлось».
        if need_city and not (patch.get("city_slug") or ctx.city_slug):
            ctx = ctx.model_copy(update={"dynamic_status": DYN_MISSING, "pending_fields": []})
        if need_branch and not (patch.get("branch_slug") or ctx.branch_slug):
            if "branch_options" not in facts:
                ctx = ctx.model_copy(update={"dynamic_status": DYN_MISSING, "pending_fields": []})

        await _save_context(ctx, fields=CONTEXT_FIELDS_STATIC | CONTEXT_FIELDS_DYNAMIC)
        stage("lookup", format_lookup_done(turn_calls), "done", calls=turn_calls)
        patch.update({"tool_log": journal, "profile": profile})
        return ctx, patch, profile

    except Exception as exc:  # noqa: BLE001
        log.warning("Лайв-канал: разбор справочника не удался: %s", exc)
        ctx = ctx.model_copy(
            update={
                "dynamic_status": DYN_MISSING,
                "pending_fields": [],
                "dynamic_turn": turn,
            }
        )
        await _save_context(ctx, fields=CONTEXT_FIELDS_DYNAMIC)
        stage("lookup", f"ошибка, пропуск: {exc}", "done")
        return ctx, patch, profile


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
    script = _script_of_state(state)
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
    """Один служебный проход чекера, разбора справочника и контекстера.

    Точка отсчёта сбрасывается при смене ``partial_utterance_id`` от бота —
    не по знаку прироста длины. Порог ``checker_min_growth_chars``
    применяется только к промежуточным кускам внутри одной реплики
    относительно ``last_checked_partial``. Финальная реплика
    (``partial_is_final``) разбирается всегда, в том числе при нулевом
    приросте. Первый проход новой реплики порогом не режется.
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

    # Пометка «в поиске» — до check_pass, иначе основной ход не успеет её увидеть.
    ctx = await _load_context(state)
    intent = _plan_live_lookup(state, reply=reply, progress=progress, profile=profile, ctx=ctx)
    if intent.will_search:
        turn = int(state.get("turn") or 0)
        ctx = ctx.model_copy(
            update={
                "dynamic_status": DYN_SEARCHING,
                "dynamic_turn": turn,
                "pending_fields": list(intent.pending),
            }
        )
        await _save_context(ctx, fields=CONTEXT_FIELDS_DYNAMIC)

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

    # Разбор города/филиала/фактов — раньше был на пути основного хода.
    ctx, lookup_patch, profile = await _lookup_in_live(
        state,
        reply=reply,
        progress=progress,
        profile=profile,
        ctx=ctx,
    )
    patch.update(lookup_patch)
    patch["profile"] = profile

    # Контекстер печёт справку/статус по реплике.
    script = _script_of_state(state)
    branches: list[Any] = []
    if ctx.city_slug and not ctx.branch_slug:
        try:
            branches = await vector_kb.list_branches(ctx.city_slug)
        except Exception as exc:  # noqa: BLE001
            log.warning("Лайв-канал: филиалы города не загрузились: %s", exc)
            branches = []
    # Статус разбора справочника не должен затираться решением «справка не нужна».
    lookup_status = ctx.dynamic_status
    lookup_pending = list(ctx.pending_fields or [])
    lookup_turn = int(ctx.dynamic_turn or 0)
    lookup_text = ctx.dynamic_text
    ctx = await run_contexter(
        ctx,
        reply=reply,
        tools=build_context_tools(script),
        objections=script.objections,
        branches=branches,
    )
    if lookup_status in {DYN_READY, DYN_MISSING, DYN_SEARCHING} and ctx.dynamic_status == DYN_NONE:
        # DYN_SEARCHING не подменяем на «готово»: данные ещё не принесены.
        ctx = ctx.model_copy(
            update={
                "dynamic_status": lookup_status,
                "pending_fields": lookup_pending if lookup_status == DYN_SEARCHING else [],
                "dynamic_turn": lookup_turn or ctx.dynamic_turn,
                "dynamic_text": ctx.dynamic_text or lookup_text,
            }
        )
    ctx_patch = await _save_context(ctx, fields=CONTEXT_FIELDS_DYNAMIC)
    patch.update(ctx_patch)

    ctx = await _warmup_next_step(
        state,
        progress=progress,
        profile=profile,
        ctx=ctx,
        asks_inform=asks_inform,
    )
    static_patch = await _save_context(ctx, fields=CONTEXT_FIELDS_STATIC)
    patch.update(static_patch)

    if closures:
        checker_text = "закрыл шаги " + ",".join(step_id for step_id, _ in closures)
    else:
        checker_text = "ничего"
    elapsed_ms = int((time.perf_counter() - started) * 1000)
    subject = (ctx.situation_slug or "").strip()
    subject_part = f", предмет «{subject}»" if subject else ""
    stage(
        "live-check",
        f"чекер: {checker_text}; контекстер: статус {ctx.dynamic_status}"
        f"{subject_part}; {elapsed_ms} мс",
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
