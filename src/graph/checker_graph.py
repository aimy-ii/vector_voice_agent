r"""Служебный граф чекера в реальном времени.

Лайв-канал: закрывает шаги, ставит разбор реплики фоновому контекстеру
(при недоступной очереди разбирает синхронно), разбирает профиль,
решает конец разговора, греет контекст под предстоящий шаг. В ``messages``
не пишет, реплик в эфир не выдаёт. Ошибка только в лог — ход генератора
не роняет.

Проход запускает бот, пока говорит человек, поэтому реплики самого бота
с ходов без реплики человека судья не видел вовсе. Их разбирает тот же
проход следом за репликой человека — ``check_agent_replies``.

Политика запусков (на стороне клиента SDK, см. настройки)::

    vector_checker  → multitask_strategy="interrupt"
        новый служебный проход отменяет предыдущий незавершённый;
        разбор реплики контекстером при этом не теряется — он живёт
        в очереди vector_contexter;

    vector_contexter → multitask_strategy="enqueue"
        фоновый контекстер: задачи копятся и доводятся до конца,
        результат уходит в кеш контекста звонка;

    vector_agent    → multitask_strategy="enqueue"
        основной ход не ждёт служебный: перед стартом клиент отменяет
        идущий ``vector_checker`` (или стартует с interrupt), иначе при
        enqueue сервер поставит основной ход в очередь за служебным.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Sequence

from langgraph.graph import StateGraph
from langgraph.runtime import Runtime

from core.config import settings
from graph.checker import CheckerClient, check_pass
from graph.context import (
    DYN_NONE,
    DYN_SEARCHING,
    DYN_WORKING,
    merge_static,
    raise_conversation_ended,
)
from graph.context_store import CONTEXT_FIELDS_DYNAMIC, CONTEXT_FIELDS_STATIC
from graph.contexter import reply_hash, run_contexter
from graph.facts import knowledge_of, needs_of
from graph.farewell_agent import decide_farewell
from graph.log_fmt import format_check_done, format_live_check_state
from graph.nearby import is_searching
from graph.nodes import (
    _call_id,
    _checker_client,
    _lead_from_progress,
    _load_context,
    _load_progress,
    _merge_profile,
    _no_client_reply,
    _save_context,
    _save_progress,
)
from graph.progress import stage
from graph.state import CallContext, CallState
from graph.tools_registry import build_context_tools
from graph.transcript import ROLE_AGENT, ROLE_CLIENT, TranscriptEntry, to_messages
from kb.client import vector_kb
from script.build import AnyStep
from script.models import SalesStep, Step
from script.planner import peek_next_step, pick_step
from script.price import price_line, price_line_from_kb
from script.source import registry
from script.store import ScriptProgress

log = logging.getLogger(__name__)

#: Треды очереди контекстера по идентификатору звонка. Кеш процесса:
#: тред создаётся платформой с её UUID, а привязка к звонку живёт в
#: метаданных; повторный поиск на каждой реплике не нужен.
_CONTEXTER_THREADS: dict[str, str] = {}


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


async def _enqueue_contexter(
    call_id: str,
    *,
    reply: str,
    needs: list[str],
    step_needs: list[str],
    profile: dict[str, str],
    state: CallState,
) -> bool:
    """Ставит разбор реплики фоновому контекстеру через очередь платформы.

    Сервер принимает только UUID тредов, поэтому тред создаётся без
    своего идентификатора: платформа выдаёт UUID, привязка к звонку
    лежит в метаданных треда (``vector_call_id``), найденный тред
    кешируется в памяти процесса. Стратегия enqueue: задачи копятся
    в треде и доводятся до конца, отмена служебного прохода их не трогает.

    Args:
        call_id: идентификатор звонка.
        reply: реплика клиента.
        needs: потребности справочника по шапке хода.
        step_needs: знания ведущего шага для агента.
        profile: слитый профиль.
        state: состояние прохода (идентификатор и версия скрипта).

    Returns:
        True — задача поставлена; False — не удалось, разбираем синхронно.
    """
    try:
        from langgraph_sdk import get_client

        client = get_client()
        thread_id = _CONTEXTER_THREADS.get(call_id)
        if not thread_id:
            found = await client.threads.search(metadata={"vector_call_id": call_id}, limit=1)
            if found:
                thread_id = str(found[0]["thread_id"])
            else:
                created = await client.threads.create(metadata={"vector_call_id": call_id})
                thread_id = str(created["thread_id"])
            _CONTEXTER_THREADS[call_id] = thread_id
        await client.runs.create(
            thread_id,
            "vector_contexter",
            input={
                "call_id": call_id,
                "reply": reply,
                "needs": needs,
                "step_needs": step_needs,
                "profile": profile,
                "script_id": str(state.get("script_id") or ""),
                "script_version": str(state.get("script_version") or ""),
            },
            multitask_strategy="enqueue",
        )
        return True
    except Exception as exc:  # noqa: BLE001
        _CONTEXTER_THREADS.pop(call_id, None)
        log.warning("Очередь контекстера недоступна, разбираю синхронно: %s", exc)
        return False


def agent_replies_to_check(
    entries: Sequence[TranscriptEntry],
    *,
    checked_entry_id: str,
) -> list[TranscriptEntry]:
    """Реплики бота после последней фразы человека, ещё не разобранные судьёй.

    Служебный проход запускается, пока говорит человек, поэтому судья
    видит только его реплику. Ответ бота на этот ход попадает в историю
    позже — на следующем служебном проходе. Шаги речи («сказать»,
    «предложить»), которые бот выполняет в ответ человеку, закрываются
    именно его репликой, а не словами клиента — их тоже берём на разбор.

    На ходах ``continuation`` / ``silence`` / ``pull`` после последней
    фразы человека идут только реплики бота — их тоже включаем. Подряд
    идущие реплики бота судья получает одним блоком в ``check_agent_replies``.

    Args:
        entries: полная история звонка из кеша контекста.
        checked_entry_id: ``entry_id`` последней разобранной реплики бота;
            пусто — разбора ещё не было.

    Returns:
        Реплики бота на разбор, в порядке разговора; пусто — разбирать нечего.
    """
    last_client = -1
    for index, entry in enumerate(entries):
        if entry.role == ROLE_CLIENT:
            last_client = index
    tail = list(entries[last_client + 1 :])
    pending = [entry for entry in tail if entry.role == ROLE_AGENT and entry.text.strip()]
    if not checked_entry_id:
        return pending
    for index, entry in enumerate(pending):
        if entry.entry_id == checked_entry_id:
            return pending[index + 1 :]
    return pending


#: Слова, которыми в требованиях шага записана добыча ответа у человека.
#:
#: Ищутся только в первой фразе требований и в условии закрытия — там, где
#: сказано, что шаг обязывает сделать. Дальше по тексту те же слова стоят в
#: оговорках про самого бота («уточнить срок при оформлении») и про клиента
#: («перечень — только если человек сам спросит»), и по ним шаг записывать в
#: требующие ответа нельзя.
_ASK_MARKERS: tuple[str, ...] = ("спрос", "спраш", "узнать", "уточнить", "выбрать", "выбор")

#: Кем в условии закрытия названа сторона, чей ответ шаг ждёт.
_HUMAN_WORDS: tuple[str, ...] = ("человек", "клиент", "собеседник")


def _requirement_sentences(requirements: str) -> list[str]:
    """Фразы требований шага без разделов «Зачем» и «Нельзя».

    Требования шага продаж написаны по одному образцу: сначала что сделать,
    следом необязательное условие закрытия, а «Зачем» и «Нельзя» — пояснение
    и запреты. Для решения о закрытии годятся только первые две части:
    в запретах перечислено то, чего на шаге не делают вовсе.

    Args:
        requirements: текст требований шага продаж.

    Returns:
        Фразы в нижнем регистре, в порядке текста.
    """
    sentences: list[str] = []
    for line in requirements.splitlines():
        text = line.strip()
        if not text or text.startswith(("Зачем:", "Нельзя:")):
            continue
        sentences.extend(part.strip().lower() for part in text.split(". ") if part.strip())
    return sentences


def _needs_client_answer(step: SalesStep) -> bool:
    """Нужен ли шагу продаж ответ человека, или бот закрывает его сам.

    Вида у шага продаж нет, но требования написаны единообразно, и по ним
    два рода шагов различаются. Шаг добычи открывается глаголом получения:
    спросить, узнать, уточнить, дать выбрать — «Выявление города» закрывает
    названный город, а не заданный вопрос. Шаг речи открывается глаголом
    высказывания: сказать, рассказать, назвать, предложить, попросить —
    «Допродажа второй категории» и «Приглашение знакомых» требуют от бота
    один раз произнести своё, и вопросительная концовка реплики этого не
    отменяет. Смотрим первую фразу: дальше в тексте те же глаголы стоят в
    оговорках, к закрытию отношения не имеющих.

    Отдельно читается условие закрытия «Шаг закрыт, когда...»: где оно
    названо через человека, шаг ждёт именно человека, даже если требование
    начиналось с высказывания («Предложить закрепить условия» закрывает
    согласие).

    Args:
        step: шаг скрипта продаж.

    Returns:
        True — закрыть шаг может только ответ человека.
    """
    sentences = _requirement_sentences(step.requirements)
    if not sentences:
        return True
    if any(marker in sentences[0] for marker in _ASK_MARKERS):
        return True
    return any(
        sentence.startswith("шаг закрыт") and any(word in sentence for word in _HUMAN_WORDS)
        for sentence in sentences
    )


def closable_by_agent_reply(step: AnyStep | None) -> bool:
    """Может ли реплика самого бота закрыть шаг.

    У старого формата вид шага и есть критерий закрытия: у ``question`` и
    ``inform_check`` это дословно «клиент ответил», и реплика бота такой шаг
    не закрывает — бот спросил про город, а закрывает шаг названный человеком
    город. ``inform`` закрывает доставка, до судьи он не доходит. Остаётся
    ``action``: его критерий — результат в диалоге, и договорённость,
    проговорённая ботом, этот результат даёт.

    У шага продаж вида нет, и раньше сюда проходили все такие шаги: решал
    один судья, а он на реплике бота держит запрет закрывать шаг, которому
    нужен ответ человека, и под запрет попадали шаги речи — бот своё сказал,
    но кончил вопросом. Теперь род шага код читает из требований
    (``_needs_client_answer``), а судья остаётся второй проверкой: выполнена
    ли требуемая речь этой самой репликой.

    Args:
        step: шаг скрипта; ``None`` — шага нет в скрипте.

    Returns:
        True — шаг можно отдать судье с репликой бота.
    """
    if step is None:
        return False
    if isinstance(step, Step):
        return step.kind == "action"
    return not _needs_client_answer(step)


async def check_agent_replies(
    state: CallState,
    *,
    progress: ScriptProgress,
    profile: dict[str, str],
    entries: Sequence[TranscriptEntry],
    judge: CheckerClient | None = None,
) -> tuple[ScriptProgress, list[tuple[str, str]], str]:
    """Прогоняет судью по репликам бота с ходов без реплики человека.

    Отдельного прохода под это нет: разбор идёт тем же служебным проходом,
    что и реплика человека, и после неё — реплика человека не должна ждать
    чужую работу. Новых запусков служебного канала не появляется, поэтому
    накладываться друг на друга проходам не на чем.

    Шаги, которым нужен ответ человека, до судьи не доходят вовсе: список
    ``in_work`` для этого разбора урезан ``closable_by_agent_reply``. Из
    вердиктов принимается только закрытие по диалогу — «спрашивать
    бессмысленно» по собственной реплике бота не решают. Признак
    «клиент просит рассказать» из этого разбора тоже не берётся: клиент
    в нём не говорил.

    Счётчики попыток и пометки взятия в работу не трогаются: закрытие
    пишется в ``status``, а ``attempts`` и ``in_work`` — поля канала
    генератора.

    Args:
        state: состояние звонка.
        progress: прогресс из кеша; закрытия проставляются в него.
        profile: слитый профиль.
        entries: полная история звонка из кеша контекста.
        judge: клиент модели; пусто — боевой.

    Returns:
        Прогресс, список закрытий ``(step_id, основание)`` и ``entry_id``
        последней разобранной реплики бота (пусто — разбирать было нечего).
    """
    pending_entries = agent_replies_to_check(
        entries,
        checked_entry_id=str(state.get("last_checked_agent_entry") or ""),
    )
    if not pending_entries:
        return progress, [], ""

    marker = pending_entries[-1].entry_id
    script = _script_of_state(state)
    allowed = [
        step_id
        for step_id in progress.in_work
        if closable_by_agent_reply(script.steps.get(step_id))
    ]
    if not allowed:
        stage(
            "live-check",
            f"реплик бота на разбор {len(pending_entries)}, закрывать нечего",
            "state",
        )
        return progress, [], ""

    # Подряд идущие реплики бота — одна его речь: судье уходят одним блоком,
    # чтобы разбор стоил один круг вызовов, а не круг на каждую реплику.
    reply = "\n".join(entry.text for entry in pending_entries)
    limited = ScriptProgress.from_mapping(progress.to_dict())
    limited.in_work = allowed
    history_entries = list(entries)[: len(entries) - len(pending_entries)]
    state_for_check: dict[str, Any] = {
        **state,
        "profile": profile,
        "messages": to_messages(history_entries),
    }
    updated, closures, _asks = await check_pass(
        state_for_check,
        reply=reply,
        judge=judge,
        progress=limited,
        speaker="agent",
    )
    accepted = [(step_id, reason) for step_id, reason in closures if reason == "диалог"]
    for step_id, _reason in accepted:
        progress.status[step_id] = updated.status.get(step_id, "closed")
    stage(
        "live-check",
        f"реплик бота на разбор {len(pending_entries)}, шагов {len(allowed)}: "
        f"{format_check_done(accepted) if accepted else 'ничего'}",
        "state",
    )
    return progress, accepted, marker


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

    Чекер работает дважды: сначала по реплике человека — слово в слово как
    раньше, — потом по репликам бота, произнесённым на ходах без реплики
    человека (``check_agent_replies``). Второй разбор бывает редко, новых
    запусков служебного канала не создаёт и в цикл хода не лезет.

    Первым делом ставит статус «в работе» с хешем реплики — ход видит,
    что фон уже взял реплику, ещё до профиля и контекстера. Контекстер
    дальше сменит статус на «в поиске» / итог. Любой выход, в том числе
    по исключению, не оставляет «в работе»: иначе ход тянет заглушки
    до конца звонка.

    Точка отсчёта сбрасывается при смене ``partial_utterance_id``;
    порог прироста — только внутри одной реплики.

    Агент прощания смотрит на реплику человека, поэтому на ходах без неё
    (``continuation``, ``silence``, ``pull``) не вызывается: прощание в
    репликах самого бота ловит ``is_farewell_reply`` в ``commit_node``.

    Args:
        state: состояние звонка с накопленной репликой человека.
        runtime: рантайм LangGraph; здесь не используется.

    Returns:
        Правки ``CallState``: прогресс, профиль, слаги справочника,
        зеркало контекста и точка отсчёта прироста.
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

        # Диспетчер: разбор реплики уезжает фоновому контекстеру в очередь.
        # Не встала задача — разбираем синхронно, как раньше (запасной путь).
        queued = await _enqueue_contexter(
            _call_id(),
            reply=reply,
            needs=list(needs),
            step_needs=list(step_needs),
            profile=dict(profile),
            state=state,
        )
        if queued:
            ctx = ctx.model_copy(update={"dynamic_status": DYN_SEARCHING, "filler_spoken": False})
            await _save_context(ctx, fields=CONTEXT_FIELDS_DYNAMIC)
        else:
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

        # Реплика человека разобрана — теперь то, что бот наговорил на ходах
        # без неё. Порядок именно такой: разбор человека не ждёт чужую работу
        # и не теряется, если проход отменят новой репликой. Сбой добавки
        # уходит в лог: разбор человека к этому моменту уже сделан.
        try:
            progress, agent_closures, agent_marker = await check_agent_replies(
                state,
                progress=progress,
                profile=profile,
                entries=ctx.transcript,
                judge=_checker_client,
            )
        except Exception as exc:  # noqa: BLE001
            log.warning("Разбор реплик бота не удался: %s", exc)
            agent_closures, agent_marker = [], ""
        if agent_marker:
            patch["last_checked_agent_entry"] = agent_marker
        closures = closures + agent_closures

        # Слаги города/филиала — в патч состояния (не в форму профиля).
        # При очереди они приезжают в кеш фоновым контекстером и попадают
        # сюда со следующего прохода; при синхронном запасном пути — сразу.
        if ctx.city_slug and not state.get("city_slug"):
            patch["city_slug"] = ctx.city_slug
            if ctx.city_name:
                patch["city_name"] = ctx.city_name
        if ctx.branch_slug and not state.get("branch_slug"):
            patch["branch_slug"] = ctx.branch_slug

        # Анкету разбирает фоновый воркер и пишет поле profile сам.
        # Проход пишет только статус шагов.
        progress_patch = await _save_progress(progress, fields=frozenset({"status"}))
        patch.update(progress_patch)
        patch["profile"] = profile

        turn_kind = str(state.get("turn_kind") or "client").strip().lower()
        # Признак один на весь граф: своя копия перечня видов хода забыла
        # про «pull», и на договаривании агент мог решать по чужой реплике.
        if _no_client_reply(turn_kind):
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
                ended = raise_conversation_ended(
                    ctx.conversation_ended,
                    bool(decision.conversation_ended),
                )
                ctx = ctx.model_copy(update={"conversation_ended": ended})
                farewell_note = f"закончен={ended}"

        ctx = await _warmup_next_step(
            state,
            progress=progress,
            profile=profile,
            ctx=ctx,
            asks_inform=asks_inform,
        )

        # Финальная запись — слияние со свежим кешем: пока проход работал,
        # воркер мог доложить данные, и копия прохода не должна их накрыть.
        fresh = await _load_context(state)
        fresh_dynamic = (fresh.dynamic_text or "").strip()
        if fresh_dynamic and fresh_dynamic not in (ctx.dynamic_text or ""):
            local_dynamic = (ctx.dynamic_text or "").strip()
            ctx.dynamic_text = (
                f"{fresh_dynamic}\n{local_dynamic}".strip() if local_dynamic else fresh_dynamic
            )
        for field in (
            "nearby_text",
            "nearby_key",
            "nearby_found",
            "branch_candidates",
            "branch_cards",
        ):
            setattr(ctx, field, getattr(fresh, field))
        if (fresh.city_slug or "").strip() and not (ctx.city_slug or "").strip():
            ctx.city_slug = fresh.city_slug
            ctx.city_name = fresh.city_name or ctx.city_name
        ctx_patch = await _save_context(ctx, fields=CONTEXT_FIELDS_STATIC | CONTEXT_FIELDS_DYNAMIC)
        patch.update(ctx_patch)

        if closures:
            checker_text = format_check_done(closures)
        else:
            checker_text = "ничего"
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        subject = (ctx.situation_slug or "").strip()
        subject_part = f", предмет «{subject}»" if subject else ""
        contexter_part = (
            f"контекстер: в очереди, статус {ctx.dynamic_status}"
            if queued
            else f"контекстер: статус {ctx.dynamic_status}"
        )
        stage(
            "live-check",
            f"чекер: {checker_text}; {contexter_part}"
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
