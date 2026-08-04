"""Узлы графа: один ход разговора.

::

    ingest → plan → respond → commit

Модельный чекер и разбор справочника живут только в лайв-канале.
Plan читает закрытия из Redis-кеша и не ждёт лайв. Генератор плюсует
счётчик всем шагам шапки в момент взятия. Прогресс скрипта пишется
в Redis; в конце звонка слепок — в тред.

Ход только читает кеш, собирает промпт и генерирует реплику: в справочник
и к контекстеру не ходит.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Sequence
from typing import Any

from langchain_core.messages import AIMessage
from langgraph.config import get_config
from langgraph.runtime import Runtime

from core.config import settings
from graph.checker import CheckerClient, close_delivered_inform
from graph.context import (
    DYN_NONE,
    DYN_SEARCHING,
    DYN_WORKING,
    ConversationContext,
    context_from_state,
    missing_needs,
)
from graph.context_store import (
    CONTEXT_FIELDS_TURN,
    context_store,
    merge_context_fields,
)
from graph.contexter import reply_hash
from graph.facts import needs_of
from graph.history import (
    last_agent_text,
    last_user_text,
    normalize,
    strip_system,
    text_of,
)
from graph.log_fmt import (
    format_plan_done,
    format_reply_integrity,
    format_spoken_preview,
)
from graph.progress import say, stage
from graph.prompts import (
    TurnMode,
    build_filler_messages,
    build_turn_messages,
    build_waiting_messages,
)
from graph.reconcile import count_agent_messages, delivery_patch
from graph.resolvers import (
    BranchResolver,
    CityResolver,
)
from graph.schemas import TURN_SCHEMA_NAME, TurnResult
from graph.state import CallContext, CallState, new_state_defaults
from graph.summary import build_summary
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

#: Подмены для офлайн-тестов (лайв-канал).
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


def _configurable() -> dict[str, Any]:
    """Словарь ``configurable`` из конфига LangGraph или пустой."""
    try:
        config = get_config()
        return dict(config.get("configurable") or {})
    except Exception:  # noqa: BLE001
        return {}


def _call_id() -> str:
    """Идентификатор звонка для ключа скрипта в кеше.

    Берёт непустой ``call_id`` из конфига; иначе ``thread_id`` без суффикса
    лайв-треда. Без конфига — ``local``.
    """
    try:
        configurable = _configurable()
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


def _turn_kind() -> str:
    """Признак запуска хода: ``client``, ``continuation``, ``silence`` или ``pull``.

    Читает ``turn_kind`` из ``configurable`` рядом с ``call_id``.
    Отсутствует или пусто — считать ``client``.
    """
    kind = str(_configurable().get("turn_kind") or "").strip().lower()
    if kind in {"continuation", "silence", "pull"}:
        return kind
    return "client"


def _no_client_reply(turn_kind: str) -> bool:
    """Ход без реплики клиента: продолжение, возврат из молчания или вытаскивание."""
    return turn_kind in {"continuation", "silence", "pull"}


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
    steps: Sequence[AnyStep],
    state: CallState,
    *,
    city_asked: bool = True,
    branch_asked: bool = True,
) -> bool:
    """Нужен ли справочник хотя бы одному шагу шапки.

    Город и филиал ищем только если шаг, который их заполняет, уже
    задавался или взят в шапку этого хода.

    Args:
        steps: шапка хода (может быть пустой).
        state: состояние хода (слаги города/филиала).
        city_asked: шаг city уже был в шапке.
        branch_asked: шаг branch уже был в шапке.

    Returns:
        True — резолвер/факты имеют смысл; иначе можно выйти сразу.
    """
    for step in steps:
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
        base = cached if cached is not None else progress
        to_save = merge_progress_fields(base, progress, fields)
    await script_store.save(_call_id(), to_save)
    return progress_to_state(to_save) if persist_state else {}


async def _load_context(state: CallState) -> ConversationContext:
    """Читает контекст из кеша; при промахе — из состояния треда.

    Args:
        state: состояние звонка.

    Returns:
        Контекст разговора.
    """
    stored = await context_store.load(_call_id())
    if stored is not None:
        return stored
    return context_from_state(state.get("conversation_context"))


async def _save_context(
    context: ConversationContext,
    *,
    fields: frozenset[str],
) -> dict[str, Any]:
    """Сливает выбранные поля в кеш и возвращает зеркало для состояния.

    Args:
        context: локальный контекст канала.
        fields: набор полей для точечной записи (статика или динамика).

    Returns:
        Правки ``{"conversation_context": ...}`` для ``CallState``.
    """
    cached = await context_store.load(_call_id())
    base = cached if cached is not None else context
    to_save = merge_context_fields(base, context, fields)
    await context_store.save(_call_id(), to_save)
    return {"conversation_context": to_save.model_dump()}


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


def _tail_has_agent_reply(messages: Sequence[Any], reply: str) -> bool:
    """Есть ли в хвосте истории последняя реплика бота.

    Смотрит последнюю AI-реплику (хвостовой human не мешает). Сравнение
    через ``normalize``: снимок бота и запись мозга могут отличаться
    пробелами и знаками.
    """
    target = normalize(reply)
    if not target:
        return True
    for message in reversed(list(messages)):
        if isinstance(message, AIMessage):
            return normalize(text_of(message)) == target
    return False


def _restore_last_agent_reply(messages: list[Any], reply: str) -> list[Any]:
    """Дописывает реплику бота в хвост, перед хвостовым human если он есть."""
    agent = AIMessage(content=reply)
    if messages and getattr(messages[-1], "type", None) == "human":
        return list(messages[:-1]) + [agent, messages[-1]]
    return list(messages) + [agent]


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
    # Снимок бота может затереть произнесённое: commit кладёт его в
    # last_agent_reply, здесь возвращаем в хвост, если пропало.
    last_reply = ((await _load_context(state)).last_agent_reply or "").strip()
    if last_reply and not _tail_has_agent_reply(messages, last_reply):
        messages = _restore_last_agent_reply(messages, last_reply)
    patch["messages"] = messages
    patch["turn"] = int(state.get("turn") or 0) + 1
    patch["facts"] = {}
    patch["spoken"] = []
    patch["turn_result"] = {}
    patch["last_error"] = None
    patch["branch_candidates"] = []
    patch["expect_continuation"] = False
    patch["turn_kind"] = _turn_kind()

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
    """Берёт шапку из кеша и помечает её шаги взятыми в работу.

    Закрытия шагов делает только лайв-канал; здесь читаем уже записанный
    прогресс и закрываем ``inform`` по факту доставки прошлой реплики.
    Ждать лайв-канал нельзя: что не успел — увидим следующим ходом.

    Шаги шапки помечаются ``in_work``; счётчик ``attempts`` растёт для
    совместимости формата, но решения по нему не принимаются. На ходе без
    реплики клиента (``continuation`` / ``silence``) пометки и счётчики
    не трогаем.

    Если на ходе с репликой клиента ведущий снова совпал с шагом прошлого
    хода — берём следующий открытый из шапки (шапка сама не меняется).
    Так не ведём повторно шаг, который судья ещё не успел закрыть.

    Счётчик ``lead_repeat`` считает подряд идущие ходы с одним ведущим
    шагом; на сдвиге сбрасывается. Сами шаги не закрываются и не открываются.
    """
    script = _script_of(state)
    progress = await _load_progress(state)
    profile = _merge_profile(state, progress)
    turn = int(state.get("turn") or 0)
    turn_kind = str(state.get("turn_kind") or "client")
    no_client_reply = _no_client_reply(turn_kind)

    # Inform закрывается кодом по доставке — это не модельный чекер.
    # Без реплики клиента закрывать нечего.
    delivered_step = state.get("delivered_step")
    if not no_client_reply and delivered_step and state.get("last_delivered", True):
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

    # Повтор ведущего прошлого хода на реплике клиента — сдвиг по шапке.
    prev_step_id = state.get("current_step")
    shifted_from: str | None = None
    lead_repeat = int(state.get("lead_repeat") or 0)
    lead_repeat = lead_repeat + 1 if step is not None and step.id == prev_step_id else 1
    if (
        not no_client_reply
        and step is not None
        and prev_step_id
        and step.id == prev_step_id
        and len(head) > 1
    ):
        shifted_from = step.id
        step = head[1]
        lead_repeat = 1

    # Новый шаг хода — только ведущий шапки, если взят впервые.
    new_step_id: str | None = None
    if head and head[0].id not in progress.in_work:
        new_step_id = head[0].id

    # Без реплики клиента шаги не закрываем и в работу не берём.
    if not no_client_reply:
        for head_step in head:
            prev = int(progress.attempts.get(head_step.id, 0))
            progress.attempts[head_step.id] = prev + 1
            if head_step.id not in progress.taken_turn:
                progress.taken_turn[head_step.id] = turn
            if head_step.id not in progress.in_work:
                progress.in_work.append(head_step.id)
            progress.status.setdefault(head_step.id, "pending")

    progress_patch = await _save_progress(progress, fields=PROGRESS_FIELDS_GENERATOR)

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
            route="respond",
            head=[(s.id, int(progress.attempts.get(s.id, 0))) for s in head],
            city_slug=state.get("city_slug")
            or context_from_state(state.get("conversation_context")).city_slug,
            branch_slug=state.get("branch_slug")
            or context_from_state(state.get("conversation_context")).branch_slug,
            call_id=_call_id(),
            shifted_from=shifted_from,
        ),
        "done",
        step=step.id if step else None,
    )
    return {
        **progress_patch,
        "profile": profile,
        "current_step": step.id if step is not None else None,
        "lead_repeat": lead_repeat,
        "next_step": nxt.id if nxt is not None else None,
        "head_steps": [s.id for s in head],
        "head_new_step": new_step_id if not no_client_reply else None,
    }


def _build_respond_messages(
    *,
    prompt_kind: str,
    script: CompiledScript,
    state: CallState,
    history: Sequence[Any],
    profile: dict[str, str],
    facts: dict[str, Any],
    lead: AnyStep | None,
    head: Sequence[AnyStep],
    context_text: str,
    dynamic_status: str,
    pending_fields: list[str],
    turn_kind: str,
) -> list[Any]:
    """Собирает сообщения генератора для одной ступени."""
    if prompt_kind == "waiting":
        return build_waiting_messages(
            script,
            messages=history,
            profile=profile,
            pending_fields=pending_fields,
            step=lead,
            history_limit=settings.waiting_history_limit,
            turn_kind=turn_kind,
            context_text=context_text,
        )
    if prompt_kind == "filler":
        return build_filler_messages(
            script,
            messages=history,
            history_limit=settings.filler_history_limit,
        )
    closed_steps = [
        script.steps[step_id]
        for step_id, status in (state.get("step_status") or {}).items()
        if status == "closed" and step_id in script.steps
    ]
    next_step_id = state.get("next_step")
    next_step = (
        script.steps[next_step_id] if next_step_id and next_step_id in script.steps else None
    )
    turn_mode: TurnMode = "normal"
    if turn_kind == "pull":
        turn_mode = "pull"
    elif (
        settings.lead_repeat_threshold > 0
        and int(state.get("lead_repeat") or 0) >= settings.lead_repeat_threshold
    ):
        turn_mode = "repeat"
    return build_turn_messages(
        script=script,
        steps=list(head),
        profile=profile,
        facts=facts,
        history=history,
        asides_done=list(state.get("asides_done") or []),
        next_step=next_step,
        context_text=context_text,
        dynamic_status=dynamic_status,
        pending_fields=pending_fields,
        turn_kind=turn_kind,
        closed_steps=closed_steps,
        mode=turn_mode,
    )


def _ladder_prompt_kind(
    *,
    status: str,
    same_reply: bool,
    stubs_spoken: int,
    force_full: bool,
) -> str:
    """Выбирает сборку ступени лестницы по статусу и лимиту заглушек."""
    if force_full or stubs_spoken >= 2:
        return "full"
    if same_reply and status == DYN_WORKING:
        return "filler"
    if same_reply and status == DYN_SEARCHING:
        return "waiting"
    return "full"


async def respond_node(state: CallState, runtime: Runtime[CallContext]) -> dict[str, Any]:
    """Генератор хода: штатная реплика или лестница ожидания данных.

    Лестница включается только когда ведущему шагу нужны данные, которых
    ещё нет в контексте. Ступени — отдельные генерации (filler → waiting →
    full) с перечитыванием статуса из кеша; произнесённое дописывается в
    локальную историю узла. Одна и та же сборка подряд не повторяется:
    без смены статуса лестница уходит в штатную генерацию. Флаг
    продолжения на лестнице не ставится.
    """
    script = _script_of(state)
    head = _head_steps(state)
    facts = dict(state.get("facts") or {})
    facts.pop("city_choices", None)
    spoken: list[str] = []
    ctx = await _load_context(state)
    user_text = last_user_text(state.get("messages") or [])
    turn_kind = str(state.get("turn_kind") or "client")
    profile = dict(state.get("profile") or {})
    is_continuation = turn_kind == "continuation"
    is_silence = turn_kind == "silence"

    digest = reply_hash(user_text) if user_text else ""
    lead = head[0] if head else None
    lead_missing = missing_needs(ctx, needs_of(lead), profile) if lead else []
    use_ladder = bool(lead_missing) and not is_continuation and not is_silence

    if is_silence:
        # Молчание собирается полным промптом, как обычный ход: короткая
        # сборка предписывала, что говорить, и выдавала «может, что-то
        # пояснить» по кругу. Факт «реплики не было» подаёт continuation_block.
        prompt_kind = "full"
        prompt_reason = "молчание"
        expect_continuation = False
    elif is_continuation:
        prompt_kind = "full"
        prompt_reason = "продолжение"
        expect_continuation = False
    elif use_ladder:
        prompt_kind = "ladder"
        prompt_reason = f"недостающие данные ведущего шага: {', '.join(lead_missing)}"
        expect_continuation = False
    else:
        prompt_kind = "full"
        prompt_reason = "данных достаточно"
        expect_continuation = False

    schema = response_format_from(TurnResult, name=TURN_SCHEMA_NAME)
    local_history: list[Any] = list(state.get("messages") or [])
    last_result: TurnResult = TurnResult(reply="")
    last_ctx = ctx
    last_error: str | None = None

    async def _generate(kind: str, reason: str, *, step: int | None = None) -> str:
        """Одна генерация; дописывает в ``spoken`` и возвращает текст реплики."""
        nonlocal last_result, last_ctx, last_error
        context_text = last_ctx.render()
        dynamic_status = last_ctx.dynamic_status or DYN_NONE
        pending_fields = list(last_ctx.pending_fields or [])
        messages = _build_respond_messages(
            prompt_kind=kind,
            script=script,
            state=state,
            history=local_history,
            profile=profile,
            facts=facts,
            lead=lead,
            head=head,
            context_text=context_text,
            dynamic_status=dynamic_status,
            pending_fields=pending_fields,
            turn_kind=turn_kind,
        )
        system_len = len(messages[0].content) if messages else 0
        step_prefix = f"ступень {step}, " if step is not None else ""
        lead_count = int(state.get("lead_repeat") or 0)
        turn_mode: TurnMode = "normal"
        if turn_kind == "pull":
            turn_mode = "pull"
        elif settings.lead_repeat_threshold > 0 and lead_count >= settings.lead_repeat_threshold:
            turn_mode = "repeat"
        if turn_mode == "pull":
            lead_hint = ", режим pull, вытаскивание"
        elif settings.lead_repeat_threshold > 0:
            lead_hint = f", режим {turn_mode}, повтор шага {lead_count}"
        else:
            lead_hint = ""
        stage(
            "prompt",
            f"{step_prefix}сборка {kind}{lead_hint}, {reason}, "
            f"системное сообщение {system_len} символов",
            "done",
            chars=system_len,
            prompt=kind,
            reason=reason,
            step=step,
        )
        if messages:
            log.debug("[prompt|done] %s", messages[0].content)

        streamed: list[str] = []

        def _on_delta(delta: str) -> None:
            streamed.append(delta)
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
            reason_fail = str(exc) or "неизвестная причина"
            log.warning("Подстановка фолбэка: %s", reason_fail)
            last_error = str(exc)
            if not streamed:
                say(script.params.fallback)
                spoken.append(script.params.fallback)
                streamed.append(script.params.fallback)
            stage("respond", "модель не ответила, отдана аварийная реплика", "done")
            last_result = TurnResult(reply="".join(streamed))
            return "".join(streamed)

        last_result = _safe_result(raw)
        streamed_text = "".join(streamed)
        integrity = format_reply_integrity(streamed=streamed_text, final=last_result.reply)
        if integrity:
            stage("respond", integrity, "done")
            log.warning("%s", integrity)
        return streamed_text or (last_result.reply or "")

    if prompt_kind != "ladder":
        await _generate(prompt_kind, prompt_reason)
        out: dict[str, Any] = {
            "turn_result": {} if last_error else last_result.model_dump(),
            "spoken": list(state.get("spoken") or []) + spoken,
            "conversation_context": last_ctx.model_dump(),
            "expect_continuation": expect_continuation,
        }
        if last_error:
            out["last_error"] = last_error
        return out

    # Лестница: filler / waiting / full, не больше двух заглушек за ход.
    # Одна и та же сборка дважды подряд не выполняется: статус без смены → full.
    deadline = time.monotonic() + float(settings.ladder_deadline_seconds)
    stubs_spoken = 0
    step_index = 0
    prev_kind: str | None = None
    while True:
        last_ctx = await _load_context(state)
        if step_index > 0 and digest and last_ctx.dynamic_reply_hash != digest:
            stage("respond", "хеш реплики сменился, лестница прервана", "done")
            break

        force_full = time.monotonic() >= deadline
        status = last_ctx.dynamic_status or DYN_NONE
        same_reply = (not digest) or last_ctx.dynamic_reply_hash == digest
        if force_full:
            kind = "full"
            reason = "дедлайн лестницы"
        else:
            kind = _ladder_prompt_kind(
                status=status,
                same_reply=same_reply,
                stubs_spoken=stubs_spoken,
                force_full=False,
            )
            if prev_kind is not None and kind == prev_kind and kind != "full":
                kind = "full"
                reason = "сборка не сменилась"
            elif stubs_spoken >= 2:
                reason = "лимит заглушек"
            elif kind == "filler":
                reason = "статус в работе"
            elif kind == "waiting":
                reason = "статус поиска"
            else:
                reason = f"статус {status}"

        # Между ступенями — разделитель, чтобы commit не склеил фразы.
        if step_index > 0 and spoken:
            spoken.append(" ")
            say(" ")

        text = await _generate(kind, reason, step=step_index + 1)
        if text:
            local_history.append(AIMessage(content=text))
        step_index += 1
        if kind == "full":
            break
        prev_kind = kind
        stubs_spoken += 1

    return {
        "turn_result": last_result.model_dump(),
        "spoken": list(state.get("spoken") or []) + spoken,
        "conversation_context": last_ctx.model_dump(),
        "expect_continuation": False,
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
    progress = await _load_progress(state)
    turn_kind = str(state.get("turn_kind") or "client")
    no_client_reply = _no_client_reply(turn_kind)

    profile = dict(state.get("profile") or {})
    asides_done = list(state.get("asides_done") or [])
    patch: dict[str, Any] = {}

    spoken_text = "".join(list(state.get("spoken") or [])).strip()

    messages = list(state.get("messages") or [])
    if spoken_text:
        messages = messages + [AIMessage(content=spoken_text)]

    # Слепок прогресса в тред; в конце звонка — на постоянку.
    all_closed = all(
        progress.status.get(step_id) == "closed" or step_id not in progress.status
        for step_id in script.step_order
    )
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

    # Разговор закончен — по флагу из кеша контекста (фоновый агент прощания).
    # На продолжении и молчании флаг не трогаем: реплики человека не было.
    conversation_ended = bool(state.get("conversation_ended"))
    conversation_context = state.get("conversation_context") or {}
    if not no_client_reply or spoken_text:
        ctx = await _load_context(state)
        if not no_client_reply:
            conversation_ended = bool(ctx.conversation_ended)
            patch["conversation_ended"] = conversation_ended
        if spoken_text:
            ctx = ctx.model_copy(update={"last_agent_reply": spoken_text})
            ctx_patch = await _save_context(ctx, fields=CONTEXT_FIELDS_TURN)
            conversation_context = ctx_patch["conversation_context"]

    # Пустой эфир — pending не ставим. Без реплики клиента шаги не трогаем.
    pending_step = step.id if step is not None and spoken_text and not no_client_reply else None
    # Флаг считает respond_node; сюда доезжает через состояние — не затираем.
    expect_continuation = bool(state.get("expect_continuation"))
    patch.update(
        {
            "profile": profile,
            "asides_done": asides_done,
            "resume_step": None,
            "outcome": profile.get("outcome") or state.get("outcome"),
            "messages": messages,
            "pending_step": pending_step,
            "pending_len": len(spoken_text)
            if not no_client_reply
            else int(state.get("pending_len") or 0),
            "pending_ai_count": count_agent_messages(state.get("messages") or []),
            "city_slug": state.get("city_slug"),
            "city_name": state.get("city_name"),
            "branch_slug": state.get("branch_slug"),
            "conversation_context": conversation_context,
            "expect_continuation": expect_continuation,
        }
    )
    stage(
        "commit",
        (
            f"произнесено {len(spoken_text)} симв.: «{format_spoken_preview(spoken_text)}», "
            f"ожидание продолжения={expect_continuation}, "
            f"разговор закончен={conversation_ended}"
            if spoken_text
            else (
                f"шаг {step.id if step else '—'}, в эфир ничего не ушло, "
                f"ожидание продолжения={expect_continuation}, "
                f"разговор закончен={conversation_ended}"
            )
        ),
        "done",
        profile_keys=sorted(profile),
    )
    if spoken_text:
        log.info("полный текст реплики: %s", spoken_text)
    return patch
