"""Узлы графа: один ход разговора.

Ход устроен так:

    ingest → plan ─┬─► verbatim ──────────────► commit
                   ├─► lookup ──► respond ──┬─► verbatim ──► commit
                   └─► respond ─────────────┴─► commit

* **ingest** — принимает историю от бота, поднимает скрипт нужной версии,
  сверяет, дослушали ли прошлую реплику.
* **plan** — код решает, каким шагом занимаемся. Модель в выборе шага не
  участвует.
* **lookup** — приносит из справочника то, что требует шаг. Всегда до модели.
* **respond** — единственный вызов модели за ход: разбор ответа клиента и
  текст реплики приезжают вместе, текст уходит в эфир дельтами.
* **verbatim** — выталкивает дословный блок мимо модели. Инвариант «рантайм
  произносит дословно» становится свойством схемы, а не пожеланием к промпту.
* **commit** — раскладывает разобранное по состоянию и запоминает, что было
  намечено.

Граф отрабатывает один ход и умирает: паузу «клиент думает» держит телефонная
линия, очередью хода управляет бот. `interrupt()` здесь не применяется.
"""

from __future__ import annotations

import logging
import random
from typing import Any

from langchain_core.messages import AIMessage
from langgraph.runtime import Runtime

from core.config import settings
from graph.facts import collect_facts, confirm_branch, confirm_city, needs_of
from graph.history import (
    is_acknowledgement,
    last_agent_text,
    last_user_text,
    strip_system,
)
from graph.progress import say, stage
from graph.prompts import build_turn_messages, fill_facts
from graph.reconcile import count_agent_messages, reopen_if_interrupted
from graph.schemas import TURN_SCHEMA_NAME, TurnResult
from graph.state import CallContext, CallState, new_state_defaults
from kb.client import vector_kb
from script.build import CompiledScript
from script.models import Step
from script.planner import exhausted, pick_step, render_step_text, steps_to_skip
from script.source import registry
from utils.llm_gen import LLMTurnFailed, astream_structured, get_llm, response_format_from

log = logging.getLogger(__name__)

#: Маршруты хода.
ROUTE_VERBATIM = "verbatim"
ROUTE_LOOKUP = "lookup"
ROUTE_RESPOND = "respond"


def _script_of(state: CallState) -> CompiledScript:
    """Достаёт скомпилированный скрипт звонка из реестра.

    Args:
        state: состояние звонка.

    Returns:
        Скомпилированный скрипт той версии, на которой идёт звонок.
    """
    return registry.get(
        state.get("script_id") or settings.script_id,
        state.get("script_version") or settings.script_version,
    )


def _current_step(state: CallState) -> Step | None:
    """Возвращает шаг, выбранный планировщиком на этом ходу.

    Args:
        state: состояние звонка.

    Returns:
        Описание шага или None.
    """
    step_id = state.get("current_step")
    if not step_id:
        return None
    script = _script_of(state)
    return script.steps.get(step_id)


async def ingest_node(state: CallState, runtime: Runtime[CallContext]) -> dict[str, Any]:
    """Принимает ход: чистит историю, поднимает скрипт, сверяет произнесённое.

    Системные сообщения от бота отбрасываются: свой промпт граф собирает сам,
    а два промпта в одном запросе дерутся между собой.

    Args:
        state: состояние звонка.
        runtime: контекст запуска.

    Returns:
        Правки состояния.
    """
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
    patch["turn_result"] = {}
    patch["last_error"] = None

    # Город, известный заранее, — необязательный вход на будущее. Сейчас
    # телефон у сети один федеральный, и рабочий путь — вопрос клиенту.
    if not state.get("city_slug") and ctx.get("city_slug"):
        patch["city_slug"] = ctx["city_slug"]
        profile = dict(state.get("profile") or {})
        profile.setdefault("city", ctx["city_slug"])
        patch["profile"] = profile

    patch.update(
        reopen_if_interrupted(
            state=state,
            messages=messages,
            last_spoken=last_agent_text(messages),
        )
    )
    stage("ingest", f"ход {patch['turn']}, скрипт {script.id} v{script.version}", "start")
    return patch


async def plan_node(state: CallState, runtime: Runtime[CallContext]) -> dict[str, Any]:
    """Выбирает шаг хода и маршрут. Модель здесь не участвует.

    Args:
        state: состояние звонка.
        runtime: контекст запуска.

    Returns:
        Правки состояния: выбранный шаг, статусы пропущенных шагов, маршрут.
    """
    script = _script_of(state)
    status = dict(state.get("step_status") or {})
    profile = dict(state.get("profile") or {})
    attempts = dict(state.get("step_attempts") or {})

    for step_id in steps_to_skip(script, status=status, profile=profile):
        status[step_id] = "skipped"

    step = pick_step(script, status=status, profile=profile, resume=state.get("resume_step"))
    while step is not None and exhausted(step, attempts):
        status[step.id] = "refused"
        step = pick_step(script, status=status, profile=profile, resume=None)

    user_text = last_user_text(state.get("messages") or [])
    # Клиент только поддакнул на дословном шаге: разбирать нечего, модель не
    # нужна. За данными при этом сходить может быть всё равно надо — маршрут и
    # признак «модель не нужна» поэтому разные вещи.
    skip_model = bool(step and step.verbatim and is_acknowledgement(user_text))

    if step is None:
        route = ROUTE_RESPOND
    elif step.needs or (step.fills and "city" in step.fills and not state.get("city_slug")):
        route = ROUTE_LOOKUP
    elif skip_model:
        route = ROUTE_VERBATIM
    else:
        route = ROUTE_RESPOND

    stage(
        "plan",
        f"шаг {step.id if step else '—'}, маршрут {route}",
        "done",
        step=step.id if step else None,
        route=route,
    )
    return {
        "step_status": status,
        "current_step": step.id if step is not None else None,
        "route": route,
        "skip_model": skip_model,
    }


async def lookup_node(state: CallState, runtime: Runtime[CallContext]) -> dict[str, Any]:
    """Приносит из справочника то, что требует шаг, — до вызова модели.

    Args:
        state: состояние звонка.
        runtime: контекст запуска.

    Returns:
        Правки состояния: факты хода и журнал походов.
    """
    script = _script_of(state)
    step = _current_step(state)
    needs = needs_of(step)
    want_cities = bool(step and "city" in step.fills and not state.get("city_slug"))

    if settings.filler_threshold_ms > 0 and script.params.fillers:
        # Механизм живёт в данных скрипта; порог включается настройкой по
        # замерам на пилоте, узел при этом не меняется.
        say(random.choice(script.params.fillers) + " ")

    facts, journal = await collect_facts(
        vector_kb,
        script=script,
        needs=needs,
        city_slug=state.get("city_slug"),
        branch_slug=state.get("branch_slug"),
        want_city_choices=want_cities,
    )
    stage("lookup", f"справочник: {len(journal)} обращений", "done", calls=journal)
    return {
        "facts": facts,
        "tool_log": list(state.get("tool_log") or []) + journal,
    }


async def respond_node(state: CallState, runtime: Runtime[CallContext]) -> dict[str, Any]:
    """Единственный вызов модели за ход.

    Разбор ответа клиента и текст реплики приезжают одним вызовом, текст
    уходит в эфир дельтами по мере генерации. Модель не уложилась в бюджет —
    в трубку идёт аварийная реплика из данных скрипта, а не тишина.

    Args:
        state: состояние звонка.
        runtime: контекст запуска.

    Returns:
        Правки состояния: разбор модели и произнесённые куски.
    """
    script = _script_of(state)
    step = _current_step(state)
    facts = dict(state.get("facts") or {})
    spoken: list[str] = []

    messages = build_turn_messages(
        script=script,
        step=step,
        profile=dict(state.get("profile") or {}),
        facts=facts,
        history=state.get("messages") or [],
        asides_done=list(state.get("asides_done") or []),
    )
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
        }

    result = _safe_result(raw)
    stage(
        "respond",
        f"разбор: статус {result.step_status}, вопрос {result.aside_id or '—'}",
        "done",
        step_status=result.step_status,
        aside_id=result.aside_id,
    )
    return {
        "turn_result": result.model_dump(),
        "spoken": list(state.get("spoken") or []) + spoken,
    }


def _safe_result(raw: dict[str, Any]) -> TurnResult:
    """Валидирует ответ модели, не роняя ход на кривом JSON.

    Args:
        raw: последний собранный объект ответа.

    Returns:
        Разбор хода; при неудачной валидации — пустой с текстом из `reply`.
    """
    try:
        return TurnResult.model_validate(raw)
    except Exception as exc:  # noqa: BLE001
        log.warning("Ответ модели не прошёл валидацию: %s", exc)
        reply = raw.get("reply") if isinstance(raw, dict) else ""
        return TurnResult(reply=reply if isinstance(reply, str) else "")


async def verbatim_node(state: CallState, runtime: Runtime[CallContext]) -> dict[str, Any]:
    """Выталкивает дословный блок скрипта мимо модели.

    Текст берётся из данных и отдаётся писателем как есть: модель его не
    видит, поэтому переформулировать или сократить не может.

    Args:
        state: состояние звонка.
        runtime: контекст запуска.

    Returns:
        Правки состояния: произнесённые куски.
    """
    step = _current_step(state)
    if step is None:
        return {}

    facts = dict(state.get("facts") or {})
    profile = dict(state.get("profile") or {})
    text = render_step_text(step, profile)
    if not text:
        return {}

    filled = fill_facts(text, facts)
    spoken = list(state.get("spoken") or [])
    prefix = " " if spoken else ""
    say(prefix + filled)
    chunks = [prefix + filled]

    if step.kind == "inform_check" and step.check_question:
        say(" " + step.check_question)
        chunks.append(" " + step.check_question)

    stage("verbatim", f"дословный блок шага {step.id}", "done", step=step.id)
    return {"spoken": spoken + chunks}


async def commit_node(state: CallState, runtime: Runtime[CallContext]) -> dict[str, Any]:
    """Раскладывает разобранное по состоянию и запоминает намеченное.

    Здесь же появляется единственная запись в `messages` от графа: полная
    намеченная реплика. Бот на следующем ходу подменит историю своей, где
    лежит фактически произнесённое, — и мы сверим одно с другим.

    Args:
        state: состояние звонка.
        runtime: контекст запуска.

    Returns:
        Правки состояния.
    """
    script = _script_of(state)
    step = _current_step(state)
    result = dict(state.get("turn_result") or {})

    profile = dict(state.get("profile") or {})
    status = dict(state.get("step_status") or {})
    attempts = dict(state.get("step_attempts") or {})
    asides_done = list(state.get("asides_done") or [])
    patch: dict[str, Any] = {}

    for item in result.get("understood") or []:
        key = str(item.get("key", "")).strip()
        value = str(item.get("value", "")).strip()
        if key in script.profile_fields and value:
            profile[key] = value

    city_slug = state.get("city_slug")
    if not city_slug and profile.get("city"):
        city_slug = await confirm_city(vector_kb, profile["city"])
        if city_slug:
            profile["city"] = city_slug
            patch["city_slug"] = city_slug
        else:
            # Города нет в сети: значение не подтверждено, шаг остаётся открытым.
            profile.pop("city", None)

    if city_slug and profile.get("branch") and not state.get("branch_slug"):
        branch_slug = await confirm_branch(vector_kb, city_slug, profile["branch"])
        if branch_slug:
            profile["branch"] = branch_slug
            patch["branch_slug"] = branch_slug
        else:
            profile.pop("branch", None)

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
    if step is not None:
        status[step.id], attempts[step.id], resume = _apply_step_status(
            step=step,
            result=result,
            spoke_something=bool(spoken_text),
            profile=profile,
            attempts=attempts,
            aside_id=str(aside_id) if aside_id else None,
        )

    messages = list(state.get("messages") or [])
    if spoken_text:
        messages = messages + [AIMessage(content=spoken_text)]

    patch.update(
        {
            "profile": profile,
            "step_status": status,
            "step_attempts": attempts,
            "asides_done": asides_done,
            "resume_step": resume,
            "outcome": profile.get("outcome") or state.get("outcome"),
            "messages": messages,
            "pending_step": step.id if step is not None and status.get(step.id) == "done" else None,
            "pending_len": len(spoken_text),
            "pending_ai_count": count_agent_messages(state.get("messages") or []),
        }
    )
    stage(
        "commit",
        f"шаг {step.id if step else '—'} → {status.get(step.id) if step else '—'}",
        "done",
        profile_keys=sorted(profile),
    )
    return patch


def _apply_step_status(
    *,
    step: Step,
    result: dict[str, Any],
    spoke_something: bool,
    profile: dict[str, str],
    attempts: dict[str, int],
    aside_id: str | None,
) -> tuple[str, int, str | None]:
    """Считает новый статус шага, счётчик попыток и точку возврата.

    Шаги закрываются по-разному: вопрос — ответом клиента, информирование —
    фактом произнесения, информирование с проверкой — произнесением вместе с
    проверочным вопросом, целевое действие — результатом.

    Args:
        step: описание шага.
        result: разбор модели.
        spoke_something: прозвучало ли на этом ходу хоть что-то.
        profile: собранный профиль после разбора.
        attempts: счётчики возвратов к шагам.
        aside_id: посторонний вопрос этого хода, если был.

    Returns:
        Тройка «статус, число попыток, шаг для возврата».
    """
    count = int(attempts.get(step.id, 0))
    resume: str | None = None

    if step.kind in ("inform", "inform_check"):
        # Шаг информирования закрывается фактом произнесения, поэтому смотрим
        # на то, что реально ушло в эфир, а не на маршрут и не на ответ модели.
        # Если клиента перебили, шаг вернётся в работу на следующем ходу —
        # сверка произнесённого разберётся.
        return ("done" if spoke_something else "open", count, None)

    if step.kind in ("question", "action"):
        filled = all(profile.get(key) for key in step.fills) if step.fills else False
        status = str(result.get("step_status") or "unclear")
        if filled or status == "done":
            return ("done", count, None)
        if status == "refused":
            return ("refused", count, None)
        count += 1
        if aside_id and result.get("resume_step", True):
            resume = step.id
        return ("open", count, resume)

    return ("open", count, None)
