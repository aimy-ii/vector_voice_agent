"""Узлы графа: один ход разговора.

::

    ingest → check → plan ─┬─► lookup ──► respond ──► commit
                           └─► respond ─────────────► commit

Чекер — единственная точка закрытия шагов; генератор плюсует счётчик ведущему
шагу в момент взятия. Прогресс скрипта пишется в Redis; в конце звонка слепок — в тред.
"""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.messages import AIMessage
from langgraph.config import get_config
from langgraph.runtime import Runtime

from core.config import settings
from graph.checker import CheckerClient, check_pass, close_delivered_inform
from graph.context import (
    DYN_SEARCHING,
    context_from_state,
    merge_static,
)
from graph.contexter import run_contexter
from graph.facts import collect_facts, needs_of
from graph.fillers import branch_filler, city_filler, cost_filler
from graph.history import (
    last_agent_text,
    last_user_text,
    strip_system,
)
from graph.log_fmt import (
    format_check_done,
    format_lookup_done,
    format_plan_done,
    format_spoken_preview,
)
from graph.names import given_name
from graph.progress import say, stage
from graph.prompts import build_turn_messages
from graph.reconcile import count_agent_messages, delivery_patch
from graph.resolvers import BranchResolver, CityResolver, resolve_branch, resolve_city
from graph.schemas import TURN_SCHEMA_NAME, TurnResult
from graph.situations import pick_filler
from graph.state import CallContext, CallState, new_state_defaults
from graph.summary import build_summary
from graph.tools_registry import build_context_tools
from kb.client import vector_kb
from script.build import CompiledScript
from script.models import Step
from script.planner import (
    answered_inform_check,
    client_asks_inform,
    peek_next_step,
    script_head,
)
from script.price import price_line
from script.source import registry
from script.store import (
    ScriptProgress,
    progress_from_state,
    progress_to_state,
    script_store,
)
from utils.llm_gen import LLMTurnFailed, astream_structured, get_llm, response_format_from

log = logging.getLogger(__name__)

ROUTE_LOOKUP = "lookup"
ROUTE_RESPOND = "respond"

#: Подмены для офлайн-тестов.
_checker_client: CheckerClient | None = None
_city_resolver: CityResolver | None = None
_branch_resolver: BranchResolver | None = None


def _thread_id() -> str:
    """Идентификатор треда из конфига LangGraph; без него — local."""
    try:
        config = get_config()
        configurable = config.get("configurable") or {}
        return str(configurable.get("thread_id") or "local")
    except Exception:  # noqa: BLE001
        return "local"


def _call_id() -> str:
    """Идентификатор звонка для ключа скрипта в кеше.

    Берёт непустой ``call_id`` из конфига; иначе ``thread_id`` без суффикса
    лайв-треда. Без конфига — ``local``.
    """
    try:
        config = get_config()
        configurable = config.get("configurable") or {}
        call_id = configurable.get("call_id")
        if call_id:
            return str(call_id)
        thread_id = str(configurable.get("thread_id") or "local")
        suffix = settings.live_thread_suffix
        if suffix and thread_id.endswith(suffix):
            return thread_id[: -len(suffix)]
        return thread_id
    except Exception:  # noqa: BLE001
        return "local"


def _script_of(state: CallState) -> CompiledScript:
    """Достаёт скомпилированный скрипт звонка из реестра."""
    return registry.get(
        state.get("script_id") or settings.script_id,
        state.get("script_version") or settings.script_version,
    )


def _current_step(state: CallState) -> Step | None:
    """Возвращает ведущий шаг хода."""
    step_id = state.get("current_step")
    if not step_id:
        return None
    return _script_of(state).steps.get(step_id)


def _head_steps(state: CallState) -> list[Step]:
    """Шапка шагов этого хода."""
    script = _script_of(state)
    ids = state.get("head_steps") or []
    if ids:
        return [script.steps[i] for i in ids if i in script.steps]
    step = _current_step(state)
    return [step] if step is not None else []


async def _load_progress(state: CallState) -> ScriptProgress:
    """Читает прогресс из Redis; при промахе — из состояния треда."""
    stored = await script_store.load(_call_id())
    if stored is not None:
        return stored
    return progress_from_state(state)


async def _save_progress(progress: ScriptProgress, *, persist_state: bool = True) -> dict[str, Any]:
    """Пишет прогресс в Redis и возвращает правки состояния."""
    await script_store.save(_call_id(), progress)
    return progress_to_state(progress) if persist_state else {}


def call_summary(state: CallState) -> dict[str, Any]:
    """Саммари звонка в любой момент разговора.

    Args:
        state: состояние звонка.

    Returns:
        Плоская структура шаг → значение.
    """
    script = _script_of(state)
    ctx = context_from_state(state.get("conversation_context"))
    return build_summary(
        script=script,
        step_status=state.get("step_status") or {},
        profile=state.get("profile") or {},
        city_slug=state.get("city_slug") or ctx.city_slug,
        city_name=state.get("city_name") or ctx.city_name,
        branch_slug=state.get("branch_slug") or ctx.branch_slug,
    )


async def ingest_node(state: CallState, runtime: Runtime[CallContext]) -> dict[str, Any]:
    """Принимает ход: чистит историю, поднимает скрипт, сверяет произнесённое."""
    ctx: CallContext = runtime.context or {}
    patch: dict[str, Any] = {
        key: value for key, value in new_state_defaults().items() if key not in state
    }

    script = registry.get(
        state.get("script_id") or ctx.get("script_id") or settings.script_id,
        state.get("script_version") or ctx.get("script_version") or settings.script_version,
    )
    patch["script_id"] = script.id
    patch["script_version"] = script.version

    messages = strip_system(state.get("messages") or [])
    patch["messages"] = messages
    patch["turn"] = int(state.get("turn") or 0) + 1
    patch["facts"] = {}
    patch["spoken"] = []
    patch["spoken_filler"] = None
    patch["turn_result"] = {}
    patch["last_error"] = None
    patch["branch_candidates"] = []

    # Внешний контекст запуска или уже запечённый conversation_context.
    conv = context_from_state(state.get("conversation_context"))
    external_slug = ctx.get("city_slug") or conv.city_slug
    if not state.get("city_slug") and external_slug:
        patch["city_slug"] = external_slug
        profile = dict(state.get("profile") or {})
        if conv.city_name:
            profile.setdefault("city", conv.city_name)
            patch["city_name"] = state.get("city_name") or conv.city_name
        else:
            profile.setdefault("city", external_slug)
        patch["profile"] = profile

    delivery = delivery_patch(
        state=state,
        messages=messages,
        last_spoken=last_agent_text(messages),
    )
    patch.update(delivery)
    stage("ingest", f"ход {patch['turn']}, скрипт {script.id} v{script.version}", "start")
    return patch


async def check_node(state: CallState, runtime: Runtime[CallContext]) -> dict[str, Any]:
    """Синхронный чекер в начале хода: закрывает шаги по счётчику и диалогу."""
    script = _script_of(state)
    progress = await _load_progress(state)
    closures: list[tuple[str, str]] = []

    delivered_step = state.get("delivered_step")
    if delivered_step and state.get("last_delivered", True):
        before = dict(progress.status)
        progress = close_delivered_inform(
            script=script,
            progress=progress,
            pending_step=delivered_step,
            delivered=True,
        )
        if (
            progress.status.get(delivered_step) == "closed"
            and before.get(delivered_step) != "closed"
        ):
            closures.append((delivered_step, "доставка"))

    progress, checked = await check_pass(
        state,
        reply=last_user_text(state.get("messages") or []),
        judge=_checker_client,
        progress=progress,
    )
    closures.extend(checked)
    patch = await _save_progress(progress)
    stage("check", format_check_done(closures), "done")
    return patch


async def plan_node(state: CallState, runtime: Runtime[CallContext]) -> dict[str, Any]:
    """Берёт шапку, плюсует счётчик ведущему шагу, выбирает маршрут."""
    script = _script_of(state)
    progress = await _load_progress(state)
    profile = dict(state.get("profile") or {})
    turn = int(state.get("turn") or 0)
    user_text = last_user_text(state.get("messages") or [])

    inform_reason = client_asks_inform(user_text) or answered_inform_check(
        script,
        status=progress.status,
        pending_step=state.get("pending_step") or state.get("delivered_step"),
    )

    head = script_head(
        script,
        status=progress.status,
        attempts=progress.attempts,
        profile=profile,
        inform_reason=inform_reason,
        pending_soft_cap=settings.pending_steps_soft_cap,
    )

    step = head[0] if head else None
    if state.get("resume_step") and state["resume_step"] in script.steps:
        resume = script.step(state["resume_step"])
        if any(s.id == resume.id for s in head):
            step = resume

    # Счётчик — только ведущему шагу хода. Висящие в шапке попытку не тратят.
    if step is not None:
        prev = int(progress.attempts.get(step.id, 0))
        progress.attempts[step.id] = prev + 1
        if step.id not in progress.taken_turn:
            progress.taken_turn[step.id] = turn
        progress.status.setdefault(step.id, "pending")

    progress_patch = await _save_progress(progress)

    if step is None:
        route = ROUTE_RESPOND
    elif step.needs or (step.fills and "city" in step.fills and not state.get("city_slug")):
        route = ROUTE_LOOKUP
    elif (
        step.fills
        and "branch" in step.fills
        and state.get("city_slug")
        and not state.get("branch_slug")
    ):
        route = ROUTE_LOOKUP
    else:
        route = ROUTE_RESPOND

    nxt = (
        peek_next_step(
            script,
            current=step,
            status=progress.status,
            profile=profile,
            attempts=progress.attempts,
            inform_reason=inform_reason,
            pending_soft_cap=settings.pending_steps_soft_cap,
        )
        if step is not None
        else None
    )
    stage(
        "plan",
        format_plan_done(
            step_id=step.id if step else None,
            route=route,
            head=[(s.id, int(progress.attempts.get(s.id, 0))) for s in head],
            city_slug=state.get("city_slug")
            or context_from_state(state.get("conversation_context")).city_slug,
            branch_slug=state.get("branch_slug")
            or context_from_state(state.get("conversation_context")).branch_slug,
        ),
        "done",
        step=step.id if step else None,
        route=route,
    )
    return {
        **progress_patch,
        "current_step": step.id if step is not None else None,
        "next_step": nxt.id if nxt is not None else None,
        "head_steps": [s.id for s in head],
        "route": route,
    }


async def lookup_node(state: CallState, runtime: Runtime[CallContext]) -> dict[str, Any]:
    """Резолверы города/филиала и факты справочника; заглушки без модели."""
    script = _script_of(state)
    step = _current_step(state)
    needs = needs_of(step)
    profile = dict(state.get("profile") or {})
    user_text = last_user_text(state.get("messages") or [])
    facts: dict[str, Any] = {}
    journal: list[dict[str, Any]] = list(state.get("tool_log") or [])
    turn_calls: list[dict[str, Any]] = []
    patch: dict[str, Any] = {}
    fillers_used = list(state.get("fillers_used") or [])
    spoken_filler: str | None = None
    turn = int(state.get("turn") or 0)
    last_filler_turn = int(state.get("last_filler_turn") or 0)
    # Заглушка не звучит два хода подряд (нулевой ход — «ещё не было»).
    allow_filler = settings.lookup_fillers_enabled and not (
        last_filler_turn > 0 and last_filler_turn == turn - 1
    )
    ctx = context_from_state(state.get("conversation_context"))

    def _speak_filler(phrase: str | None) -> None:
        nonlocal spoken_filler
        if not phrase or spoken_filler:
            return
        spoken_filler = phrase
        say(phrase + " ")
        fillers_used.append(phrase)

    def _note(entry: dict[str, Any]) -> None:
        journal.append(entry)
        turn_calls.append(entry)

    # Город: резолвер только при пустом слаге и поводе — ведущий шаг
    # собирает city (ищем в реплике) либо в профиле уже есть имя (ищем по нему).
    fills_city = bool(step and step.fills and "city" in step.fills)
    profile_city = str(profile.get("city") or "").strip()
    if fills_city and user_text:
        search_text: str | None = user_text
    elif profile_city:
        search_text = profile_city
    else:
        search_text = None

    existing_slug = state.get("city_slug") or ctx.city_slug
    if not existing_slug and search_text:
        if allow_filler:
            _speak_filler(city_filler(script.params.city_fillers, used=fillers_used))

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
                price_phrase = price_line(city_meta.get("price"), script.params.price)
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
        # Ведущий шаг city при уже известном слаге — резолвер не зовём.
        log.warning(
            "Резолвер города: слаг уже есть (%s), вызов пропущен — такого быть не должно",
            existing_slug,
        )

    city_slug = patch.get("city_slug") or state.get("city_slug") or ctx.city_slug

    # Филиал: отбор до трёх; мета только после выбора.
    if (
        city_slug
        and not (state.get("branch_slug") or ctx.branch_slug)
        and step
        and "branch" in (step.fills or [])
        and user_text
    ):
        if allow_filler and not spoken_filler:
            _speak_filler(branch_filler(script.params.branch_fillers, used=fillers_used))

        branches = await vector_kb.list_branches(city_slug)
        _note({"call": "list_branches", "slug": city_slug, "found": len(branches)})
        resolution = await resolve_branch(user_text, branches, resolver=_branch_resolver)
        _note(
            {"call": "resolve_branch", "slugs": resolution.slugs, "selected": resolution.selected}
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

    # Прочие needs шага — без city_choices в промпт.
    extra_needs = [n for n in needs if n != "branches" or "branch_options" not in facts]
    need_price_lookup = (
        "price" in extra_needs
        and city_slug
        and not (ctx.static_text and "Стоимость" in ctx.static_text)
        and "price_line" not in facts
    )
    if need_price_lookup and allow_filler and not spoken_filler:
        _speak_filler(cost_filler(script.params.fillers, used=fillers_used))

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
        # Если статика города ещё пуста, а мета уже есть — запечь.
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

    # Общая заглушка без предмета не звучит: предмет обязателен.

    # Контекстер и на маршруте lookup: справка может прийти вместе с городом.
    ctx = await run_contexter(
        ctx,
        reply=user_text,
        tools=build_context_tools(script),
        objections=script.objections,
    )

    stage("lookup", format_lookup_done(turn_calls), "done", calls=turn_calls)
    patch.update(
        {
            "facts": facts,
            "tool_log": journal,
            "profile": profile,
            "conversation_context": ctx.model_dump(),
            "spoken_filler": spoken_filler,
            "fillers_used": fillers_used,
            "spoken": list(state.get("spoken") or []) + ([spoken_filler] if spoken_filler else []),
        }
    )
    if spoken_filler:
        patch["last_filler_turn"] = turn
    return patch


async def respond_node(state: CallState, runtime: Runtime[CallContext]) -> dict[str, Any]:
    """Единственный вызов генератора за ход."""
    script = _script_of(state)
    head = _head_steps(state)
    facts = dict(state.get("facts") or {})
    # Перечень городов генератору не отдаём никогда.
    facts.pop("city_choices", None)
    spoken: list[str] = []
    ctx = context_from_state(state.get("conversation_context"))
    user_text = last_user_text(state.get("messages") or [])
    # Контекстер в ходу: реестр инструментов → динамика до генерации.
    ctx = await run_contexter(
        ctx,
        reply=user_text,
        tools=build_context_tools(script),
        objections=script.objections,
    )
    fillers_used = list(state.get("fillers_used") or [])
    spoken_filler: str | None = state.get("spoken_filler")
    # Повторный «в поиске» после заглушки — на контекст не опираемся.
    searching_retry = ctx.dynamic_status == DYN_SEARCHING and ctx.filler_spoken
    if ctx.dynamic_status == DYN_SEARCHING and not ctx.filler_spoken:
        phrase = pick_filler(ctx.situation_slug, spoken=fillers_used)
        say(phrase + " ")
        spoken.append(phrase)
        fillers_used.append(phrase)
        spoken_filler = phrase
        ctx = ctx.model_copy(update={"filler_spoken": True})

    messages = build_turn_messages(
        script=script,
        steps=head,
        profile=dict(state.get("profile") or {}),
        facts=facts,
        history=state.get("messages") or [],
        asides_done=list(state.get("asides_done") or []),
        context_text="" if searching_retry else ctx.render(),
        spoken_filler=spoken_filler,
        attempts=state.get("step_attempts") or {},
        dynamic_status=ctx.dynamic_status,
        searching_retry=searching_retry,
    )
    system_len = len(messages[0].content) if messages else 0
    stage("prompt", f"системное сообщение {system_len} символов", "done", chars=system_len)
    schema = response_format_from(TurnResult, name=TURN_SCHEMA_NAME)

    def _on_delta(delta: str) -> None:
        spoken.append(delta)
        say(delta)

    try:
        async with get_llm() as llm:
            raw = await astream_structured(
                llm,
                messages,
                schema=schema,
                text_field="reply",
                on_delta=_on_delta,
                purpose="генератор",
            )
    except LLMTurnFailed as exc:
        log.warning("Ход отдан в заглушку: %s", exc)
        if not spoken:
            say(script.params.fallback)
            spoken.append(script.params.fallback)
        stage("respond", "модель не ответила, отдана аварийная реплика", "done")
        return {
            "turn_result": {},
            "spoken": list(state.get("spoken") or []) + spoken,
            "last_error": str(exc),
            "conversation_context": ctx.model_dump(),
            "spoken_filler": spoken_filler,
            "fillers_used": fillers_used,
        }

    result = _safe_result(raw)
    stage("respond", f"разбор: вопрос {result.aside_id or '—'}", "done", aside_id=result.aside_id)
    return {
        "turn_result": result.model_dump(),
        "spoken": list(state.get("spoken") or []) + spoken,
        "conversation_context": ctx.model_dump(),
        "spoken_filler": spoken_filler,
        "fillers_used": fillers_used,
    }


def _safe_result(raw: dict[str, Any]) -> TurnResult:
    """Валидирует ответ модели, не роняя ход на кривом JSON."""
    try:
        return TurnResult.model_validate(raw)
    except Exception as exc:  # noqa: BLE001
        log.warning("Ответ модели не прошёл валидацию: %s", exc)
        reply = raw.get("reply") if isinstance(raw, dict) else ""
        return TurnResult(reply=reply if isinstance(reply, str) else "")


async def commit_node(state: CallState, runtime: Runtime[CallContext]) -> dict[str, Any]:
    """Раскладывает разобранное по состоянию; в конце звонка — слепок в тред."""
    script = _script_of(state)
    step = _current_step(state)
    result = dict(state.get("turn_result") or {})
    progress = await _load_progress(state)

    profile = dict(state.get("profile") or {})
    asides_done = list(state.get("asides_done") or [])
    patch: dict[str, Any] = {}

    for item in result.get("understood") or []:
        key = str(item.get("key", "")).strip()
        value = str(item.get("value", "")).strip()
        if key in script.profile_fields and value:
            if key in {"caller_name", "student_name"}:
                value = given_name(value) or value
            profile[key] = value

    # Город/филиал из understood не подтверждаем слагом в обход резолвера:
    # если фиксации ещё нет — значение профиля без слага не держим как city.
    if profile.get("city") and not (state.get("city_slug") or patch.get("city_slug")):
        # Оставляем читаемое имя только если слаг уже есть в состоянии.
        if not state.get("city_name"):
            pass

    aside_id = result.get("aside_id")
    if aside_id and aside_id not in asides_done:
        record = script.aside(str(aside_id))
        if record is not None:
            asides_done.append(str(aside_id))
            for key, value in getattr(record, "sets", {}).items():
                if key in script.profile_fields:
                    profile[key] = value

    spoken_text = "".join(state.get("spoken") or []).strip()
    resume: str | None = None
    if step is not None and aside_id and result.get("resume_step", True):
        if progress.status.get(step.id) != "closed":
            resume = step.id

    messages = list(state.get("messages") or [])
    if spoken_text:
        messages = messages + [AIMessage(content=spoken_text)]

    # Слепок прогресса в тред; в конце звонка — на постоянку.
    all_closed = all(
        progress.status.get(step_id) == "closed" or step_id not in progress.status
        for step_id in script.step_order
    )
    # Реалистичнее: нет доступных шагов в шапке.
    remaining = script_head(
        script,
        status=progress.status,
        attempts=progress.attempts,
        profile=profile,
        pending_soft_cap=settings.pending_steps_soft_cap,
    )
    finished = not remaining and (bool(profile.get("outcome")) or all_closed or step is None)

    progress_patch = await _save_progress(progress)
    if finished:
        progress_patch["script_progress"] = progress.to_dict()
        progress_patch["call_finished"] = True

    patch.update(progress_patch)
    # Пустой эфир (например пустая подстановка) — pending не ставим,
    # иначе was_delivered(planned_len=0) ложно закроет inform.
    pending_step = step.id if step is not None and spoken_text else None
    patch.update(
        {
            "profile": profile,
            "asides_done": asides_done,
            "resume_step": resume,
            "outcome": profile.get("outcome") or state.get("outcome"),
            "messages": messages,
            "pending_step": pending_step,
            "pending_len": len(spoken_text),
            "pending_ai_count": count_agent_messages(state.get("messages") or []),
            "city_slug": state.get("city_slug"),
            "city_name": state.get("city_name"),
            "branch_slug": state.get("branch_slug"),
            "conversation_context": state.get("conversation_context") or {},
        }
    )
    stage(
        "commit",
        (
            f"произнесено {len(spoken_text)} симв.: «{format_spoken_preview(spoken_text)}»"
            if spoken_text
            else f"шаг {step.id if step else '—'}, в эфир ничего не ушло"
        ),
        "done",
        profile_keys=sorted(profile),
    )
    return patch
