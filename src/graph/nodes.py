"""Узлы графа: один ход разговора.

::

    ingest → lookup → plan → respond → commit

Модельный чекер живёт только в лайв-канале. Plan читает закрытия из
Redis-кеша и не ждёт лайв. Генератор плюсует счётчик всем шагам шапки
в момент взятия. Прогресс скрипта пишется в Redis; в конце звонка
слепок — в тред.
"""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.messages import AIMessage
from langgraph.config import get_config
from langgraph.runtime import Runtime

from core.config import settings
from graph.checker import CheckerClient, close_delivered_inform
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
    format_lookup_done,
    format_plan_done,
    format_spoken_preview,
)
from graph.names import given_name
from graph.profile_fill import fill_basic_profile
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
from script.build import AnyStep, CompiledScript
from script.models import SalesStep
from script.planner import (
    answered_inform_check,
    peek_next_step,
    script_head,
)
from script.price import price_line, price_line_from_kb
from script.source import registry
from script.store import (
    PROGRESS_FIELDS_CHECKER,
    PROGRESS_FIELDS_GENERATOR,
    ScriptProgress,
    merge_progress_fields,
    progress_from_state,
    progress_to_state,
    script_store,
)
from utils.llm_gen import (
    LLMTurnFailed,
    astream_structured,
    get_llm,
    response_format_from,
)

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


def _current_step(state: CallState) -> AnyStep | None:
    """Возвращает ведущий шаг хода."""
    step_id = state.get("current_step")
    if not step_id:
        return None
    return _script_of(state).steps.get(step_id)


def _lead_from_progress(
    state: CallState,
    *,
    progress: ScriptProgress,
    profile: dict[str, str],
) -> tuple[list[AnyStep], AnyStep | None]:
    """Считает шапку и ведущий шаг по прогрессу — без инкремента счётчика.

    Args:
        state: состояние хода.
        progress: прогресс из кеша или зеркала.
        profile: слитый профиль.

    Returns:
        Шапка и ведущий шаг (или None).
    """
    script = _script_of(state)
    inform_reason = bool(state.get("client_asks_inform")) or answered_inform_check(
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
    return head, step


def _field_step_attempts(
    script: CompiledScript,
    progress: ScriptProgress,
    field: str,
) -> int:
    """Счётчик попыток шага, который заполняет поле профиля.

    Args:
        script: скомпилированный скрипт.
        progress: прогресс звонка.
        field: ключ поля профиля (``city``, ``branch``).

    Returns:
        Число попыток; ноль — шаг ещё ни разу не задавался.
    """
    step_id = script.filled_by.get(field)
    if not step_id and script.is_sales and field in {"city", "branch"}:
        step_id = field
    if not step_id:
        return 0
    return int(progress.attempts.get(step_id, 0))


def _step_fills_city(step: AnyStep | None) -> bool:
    """Шаг собирает город: по fills (старый) или по id/знанию (продажи)."""
    if step is None:
        return False
    if isinstance(step, SalesStep):
        if step.id == "city":
            return True
        return "перечень городов сети" in step.knowledge
    return bool(step.fills and "city" in step.fills)


def _step_fills_branch(step: AnyStep | None) -> bool:
    """Шаг собирает филиал: по fills (старый) или по id/знанию (продажи)."""
    if step is None:
        return False
    if isinstance(step, SalesStep):
        if step.id == "branch":
            return True
        return "филиалы города с адресами" in step.knowledge
    return bool(step.fills and "branch" in step.fills)


def _price_phrase_for(script: CompiledScript, price: Any) -> str | None:
    """Готовая фраза о цене: из шаблонов скрипта или из базы (продажи)."""
    if price is None:
        return None
    if script.is_sales:
        return price_line_from_kb(price)
    return price_line(price, script.params.price)


def _step_needs_lookup(
    step: AnyStep | None,
    state: CallState,
    *,
    city_asked: bool = True,
    branch_asked: bool = True,
) -> bool:
    """Нужен ли справочник на этом ведущем шаге.

    Город и филиал ищем только если шаг, который их заполняет, уже
    задавался (счётчик попыток > 0).

    Args:
        step: ведущий шаг или None.
        state: состояние хода (слаги города/филиала).
        city_asked: шаг city уже был в шапке.
        branch_asked: шаг branch уже был в шапке.

    Returns:
        True — резолвер/факты имеют смысл; иначе узел может выйти сразу.
    """
    if step is None:
        return False
    if needs_of(step):
        return True
    if city_asked and _step_fills_city(step) and not state.get("city_slug"):
        return True
    if (
        branch_asked
        and _step_fills_branch(step)
        and state.get("city_slug")
        and not state.get("branch_slug")
    ):
        return True
    return False


def _head_steps(state: CallState) -> list[AnyStep]:
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


async def _save_progress(
    progress: ScriptProgress,
    *,
    persist_state: bool = True,
    fields: frozenset[str] | None = None,
) -> dict[str, Any]:
    """Пишет прогресс в Redis и возвращает правки состояния.

    Args:
        progress: локальный прогресс канала.
        persist_state: класть ли зеркало прогресса в правки состояния.
        fields: набор полей для точечной записи. ``None`` — прежнее
            поведение: сохранить объект целиком без слияния.

    Returns:
        Правки ``CallState`` с зеркалом прогресса (если ``persist_state``).
    """
    to_save = progress
    if fields is not None:
        cached = await script_store.load(_call_id())
        # Промах кеша: базой берём локальный прогресс (из state), иначе
        # точечная запись на пустой объект затрёт чужие поля.
        base = cached if cached is not None else progress
        to_save = merge_progress_fields(base, progress, fields)
    await script_store.save(_call_id(), to_save)
    return progress_to_state(to_save) if persist_state else {}


def _merge_profile(state: CallState, progress: ScriptProgress) -> dict[str, str]:
    """Сливает профиль состояния с профилем из кеша прогресса."""
    merged = dict(state.get("profile") or {})
    for key, value in progress.profile.items():
        text = str(value).strip()
        if text and key not in merged:
            merged[key] = text
    return merged


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


async def plan_node(state: CallState, runtime: Runtime[CallContext]) -> dict[str, Any]:
    """Берёт шапку из кеша, плюсует счётчик всем её шагам, выбирает маршрут.

    Закрытия шагов делает только лайв-канал; здесь читаем уже записанный
    прогресс и закрываем ``inform`` по факту доставки прошлой реплики.
    Ждать лайв-канал нельзя: что не успел — увидим следующим ходом.

    Счётчик растёт у каждого шага шапки: генератор мог отработать любой
    из них, и для чекера шаг считается заданным. Побочный эффект — висящий
    шаг тратит попытку каждый ход, пока висит; при потолке в две попытки
    он уходит из шапки быстрее. Если на прогоне шаги вылетают слишком рано,
    поднимают ``STEP_ATTEMPT_LIMIT`` настройкой, без правки кода.
    """
    script = _script_of(state)
    progress = await _load_progress(state)
    profile = _merge_profile(state, progress)
    turn = int(state.get("turn") or 0)

    # Inform закрывается кодом по доставке — это не модельный чекер.
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
            await _save_progress(progress, fields=PROGRESS_FIELDS_CHECKER)

    inform_reason = bool(state.get("client_asks_inform")) or answered_inform_check(
        script,
        status=progress.status,
        pending_step=state.get("pending_step") or state.get("delivered_step"),
    )

    head, step = _lead_from_progress(state, progress=progress, profile=profile)

    # Шаг попал в шапку — генератор мог его отработать, для чекера задан.
    for head_step in head:
        prev = int(progress.attempts.get(head_step.id, 0))
        progress.attempts[head_step.id] = prev + 1
        if head_step.id not in progress.taken_turn:
            progress.taken_turn[head_step.id] = turn
        progress.status.setdefault(head_step.id, "pending")

    progress_patch = await _save_progress(progress, fields=PROGRESS_FIELDS_GENERATOR)

    if _step_needs_lookup(step, state):
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
            call_id=_call_id(),
        ),
        "done",
        step=step.id if step else None,
        route=route,
    )
    return {
        **progress_patch,
        "profile": profile,
        "current_step": step.id if step is not None else None,
        "next_step": nxt.id if nxt is not None else None,
        "head_steps": [s.id for s in head],
        "route": route,
    }


async def lookup_node(state: CallState, runtime: Runtime[CallContext]) -> dict[str, Any]:
    """Резолверы города/филиала и факты справочника; заглушки без модели.

    Идёт параллельно с чекером: ведущий шаг считает сам по прогрессу.
    Если искать нечего — сразу пустой патч. Ошибка не роняет ход.
    """
    try:
        return await _lookup_body(state, runtime)
    except Exception as exc:  # noqa: BLE001
        log.warning("lookup не удался: %s", exc)
        stage("lookup", f"ошибка, пропуск: {exc}", "done")
        return {}


async def _lookup_body(state: CallState, runtime: Runtime[CallContext]) -> dict[str, Any]:
    """Тело резолвера; исключения ловит ``lookup_node``."""
    script = _script_of(state)
    progress = await _load_progress(state)
    profile = _merge_profile(state, progress)
    # Фон: базовые поля из реплики — чтобы city из текста резолвился
    # даже когда ведущий шаг ещё не city (параллельно с чекером).
    user_text = last_user_text(state.get("messages") or [])
    filled = fill_basic_profile(user_text, profile)
    if filled:
        profile = {**profile, **filled}

    _head, step = _lead_from_progress(state, progress=progress, profile=profile)
    ctx = context_from_state(state.get("conversation_context"))
    profile_city = str(profile.get("city") or "").strip()
    city_asked = _field_step_attempts(script, progress, "city") > 0
    branch_asked = _field_step_attempts(script, progress, "branch") > 0
    # Резолвер нужен по шагу либо когда в профиле уже есть город без слага
    # и шаг city уже задавался (иначе вхолостую не ищем).
    needs_lookup = _step_needs_lookup(
        step, state, city_asked=city_asked, branch_asked=branch_asked
    ) or bool(city_asked and profile_city and not (state.get("city_slug") or ctx.city_slug))
    if not needs_lookup:
        # Нечего искать — не ждём справочник и не зовём контекстер.
        stage("lookup", "нечего искать, пропуск", "done")
        return {}

    needs = needs_of(step)
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

    # Город: только если шаг city уже задавался и слага ещё нет —
    # ведущий шаг собирает city (ищем в реплике) либо имя уже в профиле.
    fills_city = _step_fills_city(step)
    if city_asked and fills_city and user_text:
        search_text: str | None = user_text
    elif city_asked and profile_city:
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
        # Ведущий шаг city при уже известном слаге — резолвер не зовём.
        log.warning(
            "Резолвер города: слаг уже есть (%s), вызов пропущен — такого быть не должно",
            existing_slug,
        )

    city_slug = patch.get("city_slug") or state.get("city_slug") or ctx.city_slug

    # Филиал: только если шаг branch уже задавался; отбор до трёх.
    if (
        branch_asked
        and city_slug
        and not (state.get("branch_slug") or ctx.branch_slug)
        and step
        and _step_fills_branch(step)
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
        reason = str(exc) or "неизвестная причина"
        log.warning("Подстановка фолбэка: %s", reason)
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
        # В формате продаж форма не объявлена — пишем любые имена полей.
        allowed = script.is_sales or key in script.profile_fields
        if allowed and value:
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
                if script.is_sales or key in script.profile_fields:
                    profile[key] = value

    spoken_text = "".join(list(state.get("spoken") or [])).strip()
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

    progress_patch = await _save_progress(progress, fields=PROGRESS_FIELDS_GENERATOR)
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
