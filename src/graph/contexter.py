"""Контекстер: единственный добытчик данных и единственный писатель динамики.

Два входа: потребности шага шапки (код, без модели) и реплика клиента
(агент решает, нужен ли контекст сверх шага). Если предстоит поход —
сразу ``DYN_SEARCHING`` с хешем реплики в кеш, до первого обращения к
справочнику. По завершении — ``DYN_READY`` или ``DYN_MISSING``.
Ничего не предстоит — статус не трогает.
"""

from __future__ import annotations

import hashlib
import logging
import re
import time
from typing import Any, Mapping, Sequence

from core.config import settings
from graph.context import (
    DYN_MISSING,
    DYN_NONE,
    DYN_READY,
    DYN_SEARCHING,
    ConversationContext,
    missing_needs,
    record_empty_needs,
)
from graph.context_agent import ContextAgent, decide_context
from graph.context_store import (
    CONTEXT_FIELDS_DYNAMIC,
    context_store,
    merge_context_fields,
)
from graph.history import find_aside
from graph.log_fmt import format_contexter_done
from graph.progress import stage
from graph.tools_registry import FACT_NEEDS, ContextTool
from kb.client import vector_kb
from script.models import Objection

log = logging.getLogger(__name__)

#: Схлопывание повторных пробелов при нормализации реплики для хеша.
_WS = re.compile(r"\s+")


def reply_hash(reply: str) -> str:
    """Считает sha256 нормализованной реплики.

    Нормализация: обрезка пробелов, нижний регистр, схлопывание повторных
    пробелов. Нужна, чтобы повтор той же реплики не гонял агента и инструменты.

    Args:
        reply: реплика клиента.

    Returns:
        Шестнадцатеричный дайджест sha256.
    """
    normalized = _WS.sub(" ", (reply or "").strip().lower())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _call_id() -> str:
    """Идентификатор звонка для ключа контекста в кеше."""
    try:
        from langgraph.config import get_config

        configurable = dict((get_config() or {}).get("configurable") or {})
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


def _triggers_catalogue(
    items: Mapping[str, Objection],
) -> dict[str, Sequence[str]]:
    """Идентификатор → признаки срабатывания."""
    return {item_id: item.triggers for item_id, item in items.items()}


def _valid_branch_slugs(
    slugs: Sequence[str],
    branches: Sequence[Mapping[str, Any]],
) -> list[str]:
    """Оставляет слаги из перечня, не больше трёх; порядок сохраняет."""
    allowed = {str(b.get("slug")) for b in branches if b.get("slug")}
    out: list[str] = []
    for slug in slugs:
        text = str(slug).strip()
        if not text or text not in allowed or text in out:
            continue
        out.append(text)
        if len(out) >= 3:
            break
    return out


def _append_dynamic(context: ConversationContext, text: str) -> None:
    """Дописывает текст в динамику, без пустых дублей в хвосте."""
    chunk = text.strip()
    if not chunk:
        return
    if chunk in context.dynamic_text:
        return
    dynamic = (context.dynamic_text + "\n" + chunk).strip()
    context.dynamic_text = dynamic


def _clear_subject_unless_searching(context: ConversationContext) -> None:
    """Чистит предмет, если статус не «в поиске»."""
    if context.dynamic_status != DYN_SEARCHING:
        context.situation_slug = None


def _mark_ready(context: ConversationContext) -> None:
    """Статус «готово»: генератор может опираться на контекст."""
    context.dynamic_status = DYN_READY
    _clear_subject_unless_searching(context)


def _mark_missing(context: ConversationContext) -> None:
    """Статус «не нашлось»: инструмент сходил и ничего не нашёл."""
    context.dynamic_status = DYN_MISSING
    _clear_subject_unless_searching(context)
    context.filler_spoken = False


def _mark_searching(context: ConversationContext, *, digest: str, subject: str = "") -> None:
    """Статус «в поиске» с хешем текущей реплики."""
    context.dynamic_status = DYN_SEARCHING
    context.dynamic_reply_hash = digest
    context.situation_slug = (subject or "").strip() or None
    context.filler_spoken = False


def _mark_none(context: ConversationContext) -> None:
    """Статус «не требуется»: контекст не нужен / возражение."""
    context.dynamic_status = DYN_NONE
    _clear_subject_unless_searching(context)


def _tool_by_name(tools: Sequence[ContextTool], name: str | None) -> ContextTool | None:
    """Находит инструмент по имени в реестре."""
    if not name:
        return None
    for tool in tools:
        if tool.name == name:
            return tool
    return None


async def _persist_dynamic(context: ConversationContext) -> None:
    """Пишет поля динамики в кеш, чтобы ход увидел статус «в поиске»."""
    call_id = _call_id()
    cached = await context_store.load(call_id)
    base = cached if cached is not None else context
    to_save = merge_context_fields(base, context, CONTEXT_FIELDS_DYNAMIC)
    await context_store.save(call_id, to_save)


async def _load_branches(
    context: ConversationContext,
    branches: Sequence[Mapping[str, Any]],
) -> list[Mapping[str, Any]]:
    """Возвращает переданные филиалы или подгружает из справочника.

    Если филиал уже выбран — список агенту не нужен.
    """
    if branches:
        return list(branches)
    if (context.branch_slug or "").strip():
        return []
    slug = (context.city_slug or "").strip()
    if not slug:
        return []
    try:
        return list(await vector_kb.list_branches(slug))
    except Exception as exc:  # noqa: BLE001
        log.warning("Контекстер: филиалы города не загрузились: %s", exc)
        return []


def _finish(
    updated: ConversationContext,
    *,
    reply: str,
    tool: str | None,
    subject: str,
    started: float,
    needed: bool,
    branch_slugs: Sequence[str] = (),
) -> ConversationContext:
    """Выставляет ``dynamic_reply`` и хеш, пишет лог и возвращает контекст."""
    updated.dynamic_reply = reply
    updated.last_reply_hash = reply_hash(reply)
    elapsed_ms = int((time.perf_counter() - started) * 1000)
    stage(
        "contexter",
        format_contexter_done(
            tool=tool,
            subject=subject,
            status=updated.dynamic_status,
            elapsed_ms=elapsed_ms,
            needed=needed,
            branch_slugs_count=len(branch_slugs) if tool == "branches" else None,
        ),
        "done",
    )
    return updated


def _apply_status(
    updated: ConversationContext,
    *,
    got: bool,
    invoked: bool,
    marked_searching: bool,
    prior_status: str,
    prior_hash: str,
    prior_subject: str | None,
) -> None:
    """Ставит итог по факту вызовов; без вызовов статус не трогает."""
    if got:
        _mark_ready(updated)
        updated.filler_spoken = False
        return
    if invoked:
        _mark_missing(updated)
        return
    if marked_searching:
        updated.dynamic_status = prior_status
        updated.dynamic_reply_hash = prior_hash
        updated.situation_slug = prior_subject


async def _run_tool(
    tool: ContextTool,
    query: str,
    context: ConversationContext,
    *,
    slugs: Sequence[str] = (),
    reply: str = "",
) -> str:
    """Запускает инструмент; ошибка → пустая строка, контекстер не роняем.

    Реплику клиента прокидываем только в инструмент города: при пустом
    ``query`` от агента он подставит её сам. Остальные инструменты
    общего аргумента ``reply`` не принимают.
    """
    try:
        if tool.name == "city":
            return await tool.run(query, context, slugs=slugs, reply=reply)  # type: ignore[call-arg]
        return await tool.run(query, context, slugs=slugs)
    except Exception as exc:  # noqa: BLE001
        log.warning("Инструмент контекстера не ответил: %s", exc)
        return ""


async def _fulfill_needs(
    context: ConversationContext,
    *,
    reply: str,
    needs: Sequence[str],
    tools: Sequence[ContextTool],
) -> tuple[bool, bool]:
    """Исполняет потребности шага без агента.

    Args:
        context: контекст; инструменты могут обновить статику на месте.
        reply: реплика клиента (для резолва города).
        needs: потребности справочника из ``missing_needs``.
        tools: реестр инструментов.

    Returns:
        Пара ``(got, invoked)``: данные получены; был хотя бы один вызов.
    """
    if not needs:
        return False, False
    got = False
    invoked = False
    city_before = context.city_slug
    branch_before = context.branch_slug

    if not context.city_slug and (reply or "").strip() and "city_choices" in needs:
        city_tool = _tool_by_name(tools, "city")
        if city_tool is not None:
            invoked = True
            found = await _run_tool(city_tool, reply, context, reply=reply)
            got_city = bool((found or "").strip()) or (
                bool(context.city_slug) and context.city_slug != city_before
            )
            if (found or "").strip():
                _append_dynamic(context, found)
            if got_city:
                got = True
            record_empty_needs(context, ["city_choices"], found=got_city)

    fact_needs = [n for n in needs if n in FACT_NEEDS]
    if fact_needs:
        facts_tool = _tool_by_name(tools, "facts")
        if facts_tool is not None and hasattr(facts_tool, "needs"):
            facts_tool.needs = list(fact_needs)  # type: ignore[attr-defined]
            invoked = True
            found = await _run_tool(facts_tool, "", context)
            preview = (found or "").strip()
            log.info(
                "Контекстер facts: потребности=%s, ответ=%s",
                fact_needs,
                preview[:200] if preview else "«пусто»",
            )
            got_facts = bool(preview) or (
                bool(context.city_slug) and context.city_slug != city_before
            )
            if preview:
                _append_dynamic(context, found)
            if got_facts:
                got = True
            record_empty_needs(context, fact_needs, found=got_facts)

    if "branch_meta" in needs and context.branch_slug:
        details = _tool_by_name(tools, "branch_details")
        if details is not None:
            invoked = True
            found = await _run_tool(details, "", context)
            got_branch = bool((found or "").strip()) or (
                bool(context.branch_slug) and context.branch_slug != branch_before
            )
            if got_branch:
                got = True
            record_empty_needs(context, ["branch_meta"], found=got_branch)

    return got, invoked


async def run_contexter(
    context: ConversationContext,
    *,
    reply: str,
    tools: Sequence[ContextTool],
    needs: Sequence[str] = (),
    step_needs: Sequence[str] = (),
    profile: Mapping[str, str] | None = None,
    objections: Mapping[str, Objection] | None = None,
    agent: ContextAgent | None = None,
    branches: Sequence[Mapping[str, Any]] = (),
) -> ConversationContext:
    """Наполняет динамику/статику и ставит статус поиска.

    Порядок: повтор реплики → недостающие потребности («в поиске» до
    справочника) → возражение (агента не зовём) → агент → инструменты.

    Args:
        context: текущий контекст разговора.
        reply: реплика клиента на этот ход.
        tools: реестр инструментов.
        needs: потребности справочника по шапке хода (``needs_of``).
        step_needs: потребности ведущего шага — строки ``knowledge`` скрипта
            для агента; пустой список — раздела в промпте не будет.
        profile: форма разговора; нужна ``missing_needs`` для города/филиала.
        objections: возражения скрипта; при совпадении агента не зовём.
        agent: подмена агента для офлайн-тестов.
        branches: филиалы города для отбора слагов агентом.

    Returns:
        Контекст с обновлёнными данными и статусом.
    """
    started = time.perf_counter()
    updated = context.model_copy(deep=True)
    digest = reply_hash(reply)
    if digest and digest == (updated.last_reply_hash or ""):
        stage("contexter", "повтор реплики, пропуск", "done")
        return updated

    need_list = [str(n).strip() for n in needs if str(n).strip()]
    agent_step_needs = [str(n).strip() for n in step_needs if str(n).strip()]
    missing = missing_needs(updated, need_list, profile)
    marked_searching = False
    got = False
    invoked = False
    prior_status = updated.dynamic_status
    prior_hash = updated.dynamic_reply_hash
    prior_subject = updated.situation_slug

    if missing:
        # Потребности шага → поход точно будет: статус до любого справочника.
        _mark_searching(updated, digest=digest)
        await _persist_dynamic(updated)
        marked_searching = True
        got, invoked = await _fulfill_needs(updated, reply=reply, needs=missing, tools=tools)

    # Возражения — тактика разговора; к потребностям шага отношения не имеют.
    if objections and find_aside(reply, _triggers_catalogue(objections)):
        if got:
            _mark_ready(updated)
            updated.filler_spoken = False
        elif invoked:
            _mark_missing(updated)
        elif marked_searching:
            updated.dynamic_status = prior_status
            updated.dynamic_reply_hash = prior_hash
            updated.situation_slug = prior_subject
        else:
            _mark_none(updated)
        return _finish(
            updated,
            reply=reply,
            tool=("facts" if missing else None),
            subject="",
            started=started,
            needed=bool(missing),
        )

    # Список филиалов агенту — только явно переданный; иначе подгрузим,
    # если агент сам выберет branches.
    branches_for_agent: list[Mapping[str, Any]] = list(branches) if branches else []

    decision = await decide_context(
        reply,
        updated,
        tools,
        agent=agent,
        branches=branches_for_agent,
        step_needs=agent_step_needs,
    )

    if not missing and not decision.need:
        # Ничего не предстоит — статус не трогаем.
        return _finish(
            updated,
            reply=reply,
            tool=None,
            subject=decision.subject,
            started=started,
            needed=False,
            branch_slugs=decision.branch_slugs,
        )

    agent_tool_name: str | None = None
    branch_slugs: list[str] = list(decision.branch_slugs)
    if decision.need:
        if not marked_searching:
            _mark_searching(updated, digest=digest, subject=decision.subject)
            await _persist_dynamic(updated)
            marked_searching = True
        tool = _tool_by_name(tools, decision.tool)
        if tool is not None:
            agent_tool_name = decision.tool
            slugs: Sequence[str] = decision.branch_slugs
            if decision.tool == "branches":
                loaded = await _load_branches(updated, branches)
                if loaded:
                    branch_slugs = _valid_branch_slugs(decision.branch_slugs, loaded)
                else:
                    # Перечень не подгрузился — слаги агента идут в инструмент как есть.
                    branch_slugs = [
                        str(s).strip() for s in decision.branch_slugs if str(s).strip()
                    ][:3]
                if not branch_slugs:
                    # Слаги агента не совпали с перечнем города — инструмент
                    # отберёт филиалы сам по запросу. Отменять вызов нельзя:
                    # адреса есть в справочнике, промах модели их не отменяет.
                    log.info(
                        "Контекстер: слаги агента не из перечня города, отбор по запросу: %s",
                        list(decision.branch_slugs),
                    )
                    slugs = []
                else:
                    slugs = branch_slugs
            if tool is not None:
                invoked = True
                found = await _run_tool(
                    tool,
                    decision.query,
                    updated,
                    slugs=slugs,
                    reply=reply,
                )
                if (found or "").strip():
                    _append_dynamic(updated, found)
                    got = True

    _apply_status(
        updated,
        got=got,
        invoked=invoked,
        marked_searching=marked_searching,
        prior_status=prior_status,
        prior_hash=prior_hash,
        prior_subject=prior_subject,
    )

    return _finish(
        updated,
        reply=reply,
        tool=agent_tool_name or ("facts" if missing else None),
        subject=decision.subject,
        started=started,
        needed=bool(missing or decision.need),
        branch_slugs=branch_slugs,
    )
