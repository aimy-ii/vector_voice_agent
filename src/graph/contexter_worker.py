"""Фоновый воркер контекстера: исполняет разбор реплики до конца.

Служебный проход живёт со стратегией interrupt: новая реплика человека
отменяет незавершённый проход. Поэтому сам проход контекстер не гоняет:
он ставит задачу сюда, в очередь enqueue. Воркер берёт задачи по порядку,
исполняет существующий ``run_contexter`` (агент, инструменты, запись
динамики — без изменений) и сливает результат в кеш звонка, откуда его
читают ход и следующий проход. Статус воркер доводит до итога сам:
зависшего «в поиске» после завершения задачи не остаётся.
Подбор ближайших филиалов тоже живёт здесь: в кеш звонка пишет один
процесс, и результат подбора не может быть затёрт устаревшей копией
служебного прохода.
Анкета разбирается здесь же, после контекстера и до подбора ближайших:
перебивание отменяет только служебный проход, а разбор реплики, анкета
и подбор доводятся до конца в очереди.
"""

from __future__ import annotations

import logging
import time
from typing import Any, TypedDict

from langgraph.graph import StateGraph

from core.config import settings
from graph.context import DYN_MISSING, DYN_READY, DYN_SEARCHING, ConversationContext
from graph.context_store import (
    CONTEXT_FIELDS_DYNAMIC,
    CONTEXT_FIELDS_STATIC,
    context_store,
    merge_context_fields,
)
from graph.contexter import reply_hash, run_contexter
from graph.nearby import apply_result, format_searching, lookup_nearby, should_refresh
from graph.profile_agent import guess_profile, profile_fields_of
from graph.profile_form import rewritable_keys
from graph.progress import stage
from graph.tools_registry import build_context_tools
from graph.transcript import to_messages
from kb.client import vector_kb
from script.source import registry
from script.store import merge_progress_fields, script_store

log = logging.getLogger(__name__)


class ContexterTaskState(TypedDict, total=False):
    """Одна задача воркеру: разобрать реплику и добыть контекст.

    Attributes:
        call_id: идентификатор звонка; ключ контекста в кеше.
        reply: реплика клиента.
        needs: потребности справочника по шапке хода.
        step_needs: строки знаний ведущего шага для агента.
        profile: слитый профиль разговора.
        script_id: идентификатор скрипта.
        script_version: версия скрипта.
    """

    call_id: str
    reply: str
    needs: list[str]
    step_needs: list[str]
    profile: dict[str, str]
    script_id: str
    script_version: str


def _keep_concurrent_dynamic(base: ConversationContext, overlay: ConversationContext) -> None:
    """Не даёт локальной динамике затереть текст, появившийся в кеше параллельно.

    Args:
        base: свежий слепок кеша перед записью.
        overlay: локальный результат разбора; ``dynamic_text`` дополняется
            чужим текстом на месте.
    """
    concurrent = (base.dynamic_text or "").strip()
    local = (overlay.dynamic_text or "").strip()
    if concurrent and concurrent not in (overlay.dynamic_text or ""):
        overlay.dynamic_text = f"{concurrent}\n{local}".strip() if local else concurrent


async def contexter_task_node(state: ContexterTaskState) -> dict[str, Any]:
    """Исполняет один разбор реплики до конца и пишет результат в кеш.

    Свежесть проверяется на входе: реплика уже разобрана (совпал
    ``last_reply_hash``) или разговор завершён — выход без работы.
    Итоговый статус ставится всегда: «готово» при добытых данных,
    «не нашлось» при пустом походе; без похода статус не трогается.

    Args:
        state: задача с идентификатором звонка и материалом реплики.

    Returns:
        Пустой патч: результат живёт в кеше контекста, не в состоянии графа.
    """
    started = time.perf_counter()
    call_id = str(state.get("call_id") or "").strip()
    reply = str(state.get("reply") or "")
    if not call_id or not reply.strip():
        log.warning("Воркер контекстера: пустая задача, call_id=%r", call_id)
        return {}

    cached = await context_store.load(call_id)
    context = cached if cached is not None else ConversationContext()
    if context.conversation_ended:
        stage("contexter-worker", f"звонок {call_id}: разговор завершён, пропуск", "done")
        return {}
    if reply_hash(reply) == (context.last_reply_hash or ""):
        stage("contexter-worker", f"звонок {call_id}: реплика уже разобрана, пропуск", "done")
        return {}

    try:
        script = registry.get(
            str(state.get("script_id") or settings.script_id),
            str(state.get("script_version") or settings.script_version) or None,
        )
    except Exception as exc:  # noqa: BLE001
        log.warning("Воркер контекстера: скрипт не загрузился: %s", exc)
        return {}

    updated = await run_contexter(
        context,
        reply=reply,
        tools=build_context_tools(script),
        needs=list(state.get("needs") or []),
        step_needs=list(state.get("step_needs") or []),
        profile=dict(state.get("profile") or {}),
        objections=script.objections,
    )

    if updated.dynamic_status == DYN_SEARCHING:
        # Разбор завершён, а статус остался поисковым — доводим до итога,
        # чтобы генератор не ждал то, что уже разобрано.
        updated.dynamic_status = DYN_READY if updated.dynamic_text.strip() else DYN_MISSING
        updated.situation_slug = None

    # Анкета: разбор реплики тем же воркером, что и контекстер. Профиль
    # читается свежим из прогресса (снимок задачи мог устареть), пишется
    # точечно полем profile — статус шагов остаётся за служебным проходом.
    progress = await script_store.load(call_id)
    profile = dict(state.get("profile") or {})
    if progress is not None:
        merged_profile = dict(progress.profile or {})
        for key, value in profile.items():
            merged_profile.setdefault(key, value)
        profile = merged_profile
    rewritable = rewritable_keys()
    try:
        fields = profile_fields_of(script)
        # Транскрипт свежим из кеша: основной ход мог дописать свою реплику,
        # без неё короткий ответ клиента не к чему привязать.
        fresh_for_history = await context_store.load(call_id)
        history = to_messages((fresh_for_history or updated).transcript)
        guess = await guess_profile(
            reply,
            history=history,
            known=profile,
            fields=fields,
            rewritable=rewritable,
        )
    except Exception as exc:  # noqa: BLE001
        log.warning("Воркер: анкета не разобралась: %s", exc)
        guess = None
    if guess is not None:
        changed = False
        for item in guess.values:
            key = item.key
            value = item.value
            if not value:
                continue
            current = str(profile.get(key) or "").strip()
            if current and (key not in rewritable or current == value.strip()):
                continue
            profile[key] = value
            changed = True
        # Город в анкету берём из справочника, а не из речи. Человек
        # отвечает падежом разговора — «в Санкт-Петербурге», — и в поле
        # ложится форма, по которой звонки между собой не сравнить.
        # Справочник уже разобрал сказанное в город сети и знает его
        # название в именительном.
        resolved_city = (updated.city_name or "").strip()
        if resolved_city and profile.get("city") != resolved_city:
            profile["city"] = resolved_city
            changed = True
        if changed and progress is not None:
            progress.profile = dict(profile)
            cached = await script_store.load(call_id)
            base = cached if cached is not None else progress
            to_save = merge_progress_fields(base, progress, frozenset({"profile"}))
            await script_store.save(call_id, to_save)

    # Подбор ближайших филиалов: решение принимает код по изменению поля
    # формы, а не агент. Живёт в воркере рядом с контекстером: писатель в
    # кеш один, и поздний искатель видит результат раннего.
    nearby_place = str(profile.get("location_hint") or "").strip()
    nearby_city = (updated.city_slug or "").strip()
    nearby_key_new = should_refresh(
        city_slug=nearby_city or None,
        place=nearby_place,
        current_key=updated.nearby_key,
    )
    if nearby_key_new:
        previous_nearby_text = updated.nearby_text
        previous_nearby_found = updated.nearby_found
        updated.nearby_key = nearby_key_new
        updated.nearby_text = format_searching(nearby_place)
        # Промежуточная запись до похода: ход идёт параллельно и должен
        # увидеть, что подбор начат, а не пустоту.
        interim = await context_store.load(call_id)
        interim_merged = merge_context_fields(
            interim if interim is not None else updated,
            updated,
            CONTEXT_FIELDS_DYNAMIC,
        )
        await context_store.save(call_id, interim_merged)
        nearby_result = await lookup_nearby(
            vector_kb,
            city_slug=nearby_city,
            place=nearby_place,
            key=nearby_key_new,
        )
        updated.nearby_text, updated.nearby_found = apply_result(
            previous_text=previous_nearby_text,
            previous_found=previous_nearby_found,
            result=nearby_result,
        )
        if nearby_result.branch_slugs:
            updated.branch_candidates = nearby_result.branch_slugs
            updated.branch_cards = nearby_result.branch_cards

    base = await context_store.load(call_id)
    if base is not None:
        if base.conversation_ended:
            stage("contexter-worker", f"звонок {call_id}: завершён во время разбора", "done")
            return {}
        _keep_concurrent_dynamic(base, updated)
    merged = merge_context_fields(
        base if base is not None else updated,
        updated,
        CONTEXT_FIELDS_STATIC | CONTEXT_FIELDS_DYNAMIC,
    )
    await context_store.save(call_id, merged)

    elapsed_ms = int((time.perf_counter() - started) * 1000)
    stage(
        "contexter-worker",
        f"звонок {call_id}: статус {merged.dynamic_status}, {elapsed_ms} мс",
        "done",
    )
    return {}


def build_contexter_worker_graph() -> StateGraph:
    """Собирает граф фонового контекстера: один узел."""
    builder: StateGraph = StateGraph(ContexterTaskState)
    builder.add_node("contexter_task", contexter_task_node)
    builder.set_entry_point("contexter_task")
    builder.set_finish_point("contexter_task")
    return builder


graph = build_contexter_worker_graph().compile(name="vector_contexter")
