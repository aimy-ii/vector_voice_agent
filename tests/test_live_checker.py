"""Тесты служебного чекера и общего ядра ``check_pass``."""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from graph.checker import CheckerVerdict, check_pass, history_slice_for, run_checker
from graph.checker_graph import (
    _warmup_next_step,
    growth_below_threshold,
    is_new_utterance,
    live_check_node,
)
from graph.context import ConversationContext
from graph.context_agent import ContextDecision
from graph.context_store import MemoryContextStore
from graph.facts import needs_of
from script.planner import peek_next_step, pick_step, script_head
from script.store import ScriptProgress, progress_to_state


@pytest.fixture(autouse=True)
def _offline_context(monkeypatch):
    """Офлайн: кеш контекста в памяти, агенты не зовут модель."""
    from graph import contexter as contexter_module
    from graph import nodes as nodes_module
    from graph.profile_agent import ProfileGuess

    mem = MemoryContextStore()
    monkeypatch.setattr(nodes_module, "context_store", mem)
    monkeypatch.setattr(contexter_module, "context_store", mem)

    async def _no_need(*_a, **_k):
        return ContextDecision(need=False)

    async def _no_profile(*_a, **_k):
        return ProfileGuess()

    monkeypatch.setattr("graph.contexter.decide_context", _no_need)
    monkeypatch.setattr("graph.checker_graph.guess_profile", _no_profile)
    return mem


class FakeChecker:
    """Заглушка модели чекера."""

    def __init__(self, verdicts: list[CheckerVerdict | None]) -> None:
        self.verdicts = list(verdicts)
        self.calls: list[dict] = []

    async def judge(
        self,
        *,
        history_slice,
        client_reply,
        step,
        step_text,
        attempts: int = 0,
        age: int = 0,
        in_work: bool = False,
    ):
        self.calls.append(
            {
                "history_slice": history_slice,
                "client_reply": client_reply,
                "step_id": step.id,
                "step_text": step_text,
                "attempts": attempts,
                "age": age,
                "in_work": in_work,
            }
        )
        if not self.verdicts:
            return None
        return self.verdicts.pop(0)


def _name_progress() -> ScriptProgress:
    """Прогресс: шаг name задан, ждёт ответа."""
    return ScriptProgress(
        status={"name": "pending"},
        attempts={"name": 1},
        taken_turn={"name": 1},
    )


def _state(
    script,
    *,
    reply_in_messages: str | None = None,
    partial: str = "",
    messages: list | None = None,
    progress: ScriptProgress | None = None,
    profile: dict[str, str] | None = None,
    turn: int = 2,
    last_checked: str = "",
    utterance_id: str = "",
    last_utterance_id: str = "",
    is_final: bool = False,
) -> dict[str, Any]:
    """Собирает минимальное состояние для ``check_pass`` / live_check."""
    prog = progress or _name_progress()
    if messages is None:
        messages = [AIMessage(content="Как к вам обращаться?")]
        if reply_in_messages is not None:
            messages.append(HumanMessage(content=reply_in_messages))
    state: dict[str, Any] = {
        "script_id": script.id,
        "script_version": script.version,
        "messages": messages,
        "profile": profile or {},
        "turn": turn,
        "partial_reply": partial,
        "partial_utterance_id": utterance_id,
        "partial_is_final": is_final,
        "last_checked_partial": last_checked,
        "last_checked_utterance_id": last_utterance_id,
    }
    state.update(progress_to_state(prog))
    return state


@pytest.mark.parametrize("source", ["full", "partial"])
async def test_check_pass_одинаков_для_полной_и_partial(script, source):
    """Один текст — один результат, источник роли не играет."""
    text = "Меня зовут Андрей"
    client = FakeChecker([CheckerVerdict(reply_usable=True, step_closed=False)])
    progress = _name_progress()
    if source == "full":
        state = _state(script, reply_in_messages=text, progress=progress)
    else:
        state = _state(script, partial=text, progress=progress)

    updated, closures, _asks = await check_pass(
        state,
        reply=text,
        judge=client,
        progress=progress,
    )
    assert updated.status.get("name") == "pending"
    assert closures == []
    assert client.calls[0]["client_reply"] == text
    assert text not in client.calls[0]["history_slice"]
    assert "Как к вам обращаться" in client.calls[0]["history_slice"]


async def test_partial_не_попадает_в_срез_истории(script):
    """Накопленная реплика — отдельное поле, не хвост истории."""
    client = FakeChecker([CheckerVerdict(reply_usable=True, step_closed=False)])
    partial = "я из Пер"
    messages = [
        AIMessage(content="Как к вам обращаться?"),
        HumanMessage(content="Андрей"),
        AIMessage(content="Из какого города?"),
    ]
    progress = ScriptProgress(
        status={"name": "closed", "city": "pending"},
        attempts={"name": 1, "city": 1},
        taken_turn={"name": 1, "city": 2},
    )
    state = _state(
        script,
        messages=messages,
        partial=partial,
        progress=progress,
        profile={"caller_name": "Андрей"},
        turn=3,
    )
    await check_pass(state, reply=partial, judge=client, progress=progress)
    assert client.calls
    call = client.calls[0]
    assert call["client_reply"] == partial
    assert call["step_id"] == "city"
    assert partial not in call["history_slice"]
    assert "Из какого города" in call["history_slice"]
    # Поля раздельные: срез и реплика — разное содержимое.
    assert call["history_slice"] != call["client_reply"]


async def test_срез_не_отрезает_чужой_human_при_partial(script):
    """Partial ещё не в messages — прошлый ответ клиента остаётся в срезе."""
    progress = ScriptProgress(
        status={"city": "pending"},
        attempts={"city": 1},
        taken_turn={"city": 1},
    )
    messages = [
        AIMessage(content="имя?"),
        HumanMessage(content="Андрей"),
        AIMessage(content="город?"),
    ]
    sliced = history_slice_for(
        messages,
        steps=[script.step("city")],
        progress=progress,
        turn=3,
        reply="я из Пер",
    )
    assert any(m.content == "Андрей" for m in sliced)
    # Если бы reply совпал с хвостом — отрезали бы; чужой human не трогаем.
    assert sliced[-1].type == "ai"


async def test_служебный_ниже_порога_модель_не_зовётся(script):
    """Прирост меньше порога и не первый проход — тихий выход."""
    client = FakeChecker([CheckerVerdict(reply_usable=True, step_closed=True)])
    previous = "Меня зовут"
    reply = previous + " А"  # прирост 2 < 10
    assert growth_below_threshold(reply, previous, min_growth=10)

    progress = _name_progress()
    state = _state(script, partial=reply, progress=progress, last_checked=previous)

    class _MemStore:
        def __init__(self) -> None:
            self.saved = False

        async def load(self, call_id: str):
            return progress

        async def save(self, call_id: str, prog):
            self.saved = True
            return True

    store = _MemStore()
    with (
        patch("graph.checker_graph._checker_client", client),
        patch("graph.checker_graph._load_progress", side_effect=lambda s: store.load("t")),
        patch(
            "graph.checker_graph._save_progress",
            side_effect=lambda p, **kw: store.save("t", p) or {},
        ),
        patch("graph.checker_graph.settings") as mock_settings,
    ):
        mock_settings.checker_min_growth_chars = 10
        patch_out = await live_check_node(state, runtime=None)  # type: ignore[arg-type]

    assert patch_out == {}
    assert client.calls == []
    assert not store.saved


async def test_финальная_реплика_при_нулевом_приросте_разбирается(script):
    """partial_is_final=True — порог игнорируется, даже прирост 0."""
    # Текст без fills: иначе имя закрывается кодом до модели.
    text = "пока думаю как ответить на вопрос"
    client = FakeChecker([CheckerVerdict(reply_usable=True, step_closed=False)])
    progress = _name_progress()
    state = _state(
        script,
        partial=text,
        progress=progress,
        last_checked=text,
        utterance_id="u1",
        last_utterance_id="u1",
        is_final=True,
    )
    assert growth_below_threshold(text, text, min_growth=10)

    async def fake_load(_state):
        return progress

    async def fake_save(prog, *, persist_state=True, fields=None):
        return progress_to_state(prog)

    async def fake_warmup(*args, **kwargs):
        return kwargs["ctx"]

    with (
        patch("graph.checker_graph._checker_client", client),
        patch("graph.checker_graph._load_progress", side_effect=fake_load),
        patch("graph.checker_graph._save_progress", side_effect=fake_save),
        patch("graph.checker_graph._warmup_next_step", side_effect=fake_warmup),
        patch("graph.checker_graph.settings") as mock_settings,
    ):
        mock_settings.checker_min_growth_chars = 10
        mock_settings.script_id = script.id
        mock_settings.script_version = script.version
        mock_settings.pending_steps_soft_cap = 4
        patch_out = await live_check_node(state, runtime=None)  # type: ignore[arg-type]

    assert patch_out != {}
    assert client.calls
    assert client.calls[0]["client_reply"] == text
    assert patch_out.get("last_checked_partial") == text


async def test_служебный_с_приростом_как_синхронный(script):
    """Достаточный прирост → те же закрытия, что дал бы sync на том же тексте."""
    text = "Меня зовут Андрей Петров"
    # Закрытие по fills кодом — одинаково для обоих источников текста.
    profile = {"caller_name": "Андрей"}
    progress = _name_progress()

    live_updated, live_closures, _asks = await check_pass(
        _state(script, partial=text, progress=progress, profile=profile),
        reply=text,
        judge=FakeChecker([CheckerVerdict(reply_usable=True, step_closed=True)]),
        progress=progress,
    )
    sync_updated, sync_closures = await run_checker(
        script=script,
        progress=progress,
        messages=[
            AIMessage(content="Как к вам обращаться?"),
            HumanMessage(content=text),
        ],
        profile=profile,
        turn=2,
        client=FakeChecker([CheckerVerdict(reply_usable=True, step_closed=True)]),
    )
    assert live_updated.status == sync_updated.status
    assert live_closures == sync_closures
    assert live_updated.status["name"] == "closed"
    assert ("name", "диалог") in live_closures


async def test_закрытия_служебного_видны_в_шапке(script):
    """После служебного check_pass закрытый шаг не попадает в шапку основного."""
    progress = ScriptProgress(
        status={"name": "pending", "city": "pending"},
        attempts={"name": 1, "city": 0},
        taken_turn={"name": 1},
    )
    profile = {"caller_name": "Андрей"}
    updated, _, _asks = await check_pass(
        _state(script, partial="Меня зовут Андрей", progress=progress, profile=profile),
        reply="Меня зовут Андрей",
        judge=FakeChecker([]),
        progress=progress,
    )
    assert updated.status["name"] == "closed"
    head = script_head(
        script,
        status=updated.status,
        attempts=updated.attempts,
        profile=profile,
        pending_soft_cap=4,
    )
    assert all(step.id != "name" for step in head)


async def test_первый_проход_не_режется_порогом():
    """Пустой last_checked — всегда пропускаем к модели."""
    assert not growth_below_threshold("коротко", "", min_growth=10)
    assert growth_below_threshold("коротко+", "коротко", min_growth=10)


def test_новая_реплика_по_utterance_id():
    """Смена идентификатора от бота — новая реплика, независимо от длины."""
    assert is_new_utterance("u2", "u1")
    assert is_new_utterance("u1", "")
    assert not is_new_utterance("u1", "u1")
    assert not is_new_utterance("", "u1")


async def test_прирост_внутри_реплики_от_разобранного_ранее():
    """Внутри одной реплики порог считается от last_checked этой же реплики."""
    previous = "Меня зовут"
    # +2 символа — ниже порога.
    assert growth_below_threshold(previous + " А", previous, min_growth=10)
    # +12 — выше порога.
    assert not growth_below_threshold(previous + " Андрей Петр", previous, min_growth=10)


async def test_live_check_новая_реплика_сбрасывает_точку_независимо_от_знака(script):
    """Новый utterance_id сбрасывает точку отсчёта даже при положительном приросте."""
    # Новая реплика длиннее прошлого last_checked — без сброса прирост был бы +8 < 10.
    previous = "xxx"
    text = previous + "12345678"
    assert len(text) - len(previous) == 8
    assert growth_below_threshold(text, previous, min_growth=10)
    client = FakeChecker([CheckerVerdict(reply_usable=True, step_closed=False)])
    progress = _name_progress()
    state = _state(
        script,
        partial=text,
        progress=progress,
        last_checked=previous,
        utterance_id="u2",
        last_utterance_id="u1",
    )

    async def fake_load(_state):
        return progress

    async def fake_save(prog, *, persist_state=True, fields=None):
        return progress_to_state(prog)

    async def fake_warmup(*args, **kwargs):
        return kwargs["ctx"]

    with (
        patch("graph.checker_graph._checker_client", client),
        patch("graph.checker_graph._load_progress", side_effect=fake_load),
        patch("graph.checker_graph._save_progress", side_effect=fake_save),
        patch("graph.checker_graph._warmup_next_step", side_effect=fake_warmup),
        patch("graph.checker_graph.settings") as mock_settings,
    ):
        mock_settings.checker_min_growth_chars = 10
        mock_settings.script_id = script.id
        mock_settings.script_version = script.version
        mock_settings.pending_steps_soft_cap = 4
        patch_out = await live_check_node(state, runtime=None)  # type: ignore[arg-type]

    assert client.calls
    assert client.calls[0]["client_reply"] == text
    assert patch_out.get("last_checked_partial") == text
    assert patch_out.get("last_checked_utterance_id") == "u2"


async def test_live_check_прирост_внутри_реплики_сравнивается_с_порогом(script):
    """Тот же utterance_id: прирост ниже порога — тихий пропуск."""
    previous = "Меня зовут"
    reply = previous + " А"
    assert growth_below_threshold(reply, previous, min_growth=10)
    client = FakeChecker([CheckerVerdict(reply_usable=True, step_closed=True)])
    progress = _name_progress()
    state = _state(
        script,
        partial=reply,
        progress=progress,
        last_checked=previous,
        utterance_id="u1",
        last_utterance_id="u1",
    )

    class _MemStore:
        def __init__(self) -> None:
            self.saved = False

        async def load(self, call_id: str):
            return progress

        async def save(self, call_id: str, prog):
            self.saved = True
            return True

    store = _MemStore()
    with (
        patch("graph.checker_graph._checker_client", client),
        patch("graph.checker_graph._load_progress", side_effect=lambda s: store.load("t")),
        patch(
            "graph.checker_graph._save_progress",
            side_effect=lambda p, **kw: store.save("t", p) or {},
        ),
        patch("graph.checker_graph.settings") as mock_settings,
    ):
        mock_settings.checker_min_growth_chars = 10
        patch_out = await live_check_node(state, runtime=None)  # type: ignore[arg-type]

    assert patch_out == {}
    assert client.calls == []
    assert not store.saved


async def test_live_check_done_содержит_длительность_мс(script):
    """В [live-check|done] есть длительность прохода в миллисекундах."""
    text = "пока ещё думаю над ответом длинный"
    client = FakeChecker([CheckerVerdict(reply_usable=True, step_closed=False)])
    progress = _name_progress()
    state = _state(script, partial=text, progress=progress, last_checked="")
    stages: list[tuple] = []

    async def fake_load(_state):
        return progress

    async def fake_save(prog, *, persist_state=True, fields=None):
        return progress_to_state(prog)

    async def fake_warmup(*args, **kwargs):
        return kwargs["ctx"]

    def _stage(name, text, kind="done", **kwargs):
        stages.append((name, text, kind))

    with (
        patch("graph.checker_graph._checker_client", client),
        patch("graph.checker_graph._load_progress", side_effect=fake_load),
        patch("graph.checker_graph._save_progress", side_effect=fake_save),
        patch("graph.checker_graph._warmup_next_step", side_effect=fake_warmup),
        patch("graph.checker_graph.stage", side_effect=_stage),
        patch("graph.checker_graph.settings") as mock_settings,
    ):
        mock_settings.checker_min_growth_chars = 10
        mock_settings.script_id = script.id
        mock_settings.script_version = script.version
        mock_settings.pending_steps_soft_cap = 4
        await live_check_node(state, runtime=None)  # type: ignore[arg-type]

    done = [t for n, t, k in stages if n == "live-check" and k == "done"]
    assert done
    assert " мс" in done[0]


async def test_live_check_с_приростом_зовёт_чекер_и_пишет_last_checked(script):
    """Достаточный прирост: чекер отрабатывает, last_checked_partial обновляется."""
    # Текст без имени — иначе фон закроет name по fills без модели.
    text = "пока ещё думаю над ответом длинный"
    client = FakeChecker([CheckerVerdict(reply_usable=True, step_closed=False)])
    progress = _name_progress()
    state = _state(
        script,
        partial=text,
        progress=progress,
        last_checked="пока",  # прирост >> 10
    )

    async def fake_load(_state):
        return progress

    async def fake_save(prog, *, persist_state=True, fields=None):
        return progress_to_state(prog)

    async def fake_warmup(*args, **kwargs):
        return kwargs["ctx"]

    with (
        patch("graph.checker_graph._checker_client", client),
        patch("graph.checker_graph._load_progress", side_effect=fake_load),
        patch("graph.checker_graph._save_progress", side_effect=fake_save),
        patch("graph.checker_graph._warmup_next_step", side_effect=fake_warmup),
        patch("graph.checker_graph.settings") as mock_settings,
    ):
        mock_settings.checker_min_growth_chars = 10
        mock_settings.script_id = script.id
        mock_settings.script_version = script.version
        mock_settings.pending_steps_soft_cap = 4
        patch_out = await live_check_node(state, runtime=None)  # type: ignore[arg-type]

    assert client.calls
    assert client.calls[0]["client_reply"] == text
    assert patch_out.get("last_checked_partial") == text


async def test_логи_не_роняют_ход_при_пустом_прогрессе(script, caplog):
    """Пустой прогресс и профиль: [live-check|state] и [check|pending] без падения."""
    import logging

    text = "пока ещё думаю что ответить на вопрос"
    client = FakeChecker([])
    progress = ScriptProgress()
    state = _state(script, partial=text, progress=progress, profile={}, last_checked="")

    async def fake_load(_state):
        return progress

    async def fake_save(prog, *, persist_state=True, fields=None):
        return progress_to_state(prog)

    async def fake_warmup(*args, **kwargs):
        return kwargs["ctx"]

    with (
        caplog.at_level(logging.INFO),
        patch("graph.checker_graph._checker_client", client),
        patch("graph.checker_graph._load_progress", side_effect=fake_load),
        patch("graph.checker_graph._save_progress", side_effect=fake_save),
        patch("graph.checker_graph._warmup_next_step", side_effect=fake_warmup),
        patch("graph.checker_graph._call_id", return_value="diag-call"),
        patch("graph.checker_graph.settings") as mock_settings,
    ):
        mock_settings.checker_min_growth_chars = 10
        mock_settings.script_id = script.id
        mock_settings.script_version = script.version
        mock_settings.pending_steps_soft_cap = 4
        patch_out = await live_check_node(state, runtime=None)  # type: ignore[arg-type]

    assert patch_out != {}
    assert client.calls == []
    messages = [rec.message for rec in caplog.records]
    assert any("[live-check|state]" in m and "счётчики {}" in m for m in messages)
    assert any("[live-check|start]" in m and "звонок diag-call" in m for m in messages)
    assert any("[check|pending]" in m and "на проверку: пусто" in m for m in messages)


async def test_check_pass_логирует_висящие_перед_моделью(script, caplog):
    """[check|pending] показывает шаг со счётчиком до вызова модели."""
    import logging

    text = "В городе Санкт-Петербург"
    client = FakeChecker([CheckerVerdict(reply_usable=True, step_closed=False)])
    progress = ScriptProgress(
        status={"name": "closed", "city": "pending"},
        attempts={"name": 1, "city": 1},
        taken_turn={"name": 1, "city": 2},
    )
    state = _state(
        script,
        partial=text,
        progress=progress,
        profile={"caller_name": "Андрей"},
        turn=3,
    )
    with caplog.at_level(logging.INFO, logger="graph.checker"):
        await check_pass(state, reply=text, judge=client, progress=progress)

    assert client.calls
    pending_logs = [r.message for r in caplog.records if "[check|pending]" in r.message]
    assert pending_logs
    assert "на проверку: [city(1)]" in pending_logs[0]
    assert "name — исчерпан" in pending_logs[0]


async def test_live_check_контекстер_пишет_динамику_в_кеш(script, monkeypatch, _offline_context):
    """Лайв-канал читает контекст из кеша с городом и пишет только динамику."""
    from graph.context import DYN_READY, ConversationContext
    from tests.conftest import FakeKB

    text = "какие филиалы у Ленина?"
    fact = "Филиалы под запрос: ул. Ленина, 1."
    client = FakeChecker([CheckerVerdict(reply_usable=True, step_closed=False)])
    progress = _name_progress()
    state = _state(
        script,
        partial=text,
        progress=progress,
        last_checked="",
    )

    ctx_store = _offline_context
    seeded = ConversationContext(
        static_text="Город: Пермь",
        city_slug="perm",
        city_name="Пермь",
        frozen=False,
    )
    await ctx_store.save("local", seeded)

    class _Tool:
        name = "branches"
        description = "филиалы"

        async def run(self, query: str, context: ConversationContext, *, slugs=()) -> str:
            assert context.city_slug == "perm"
            return fact

    async def _decide(*a, **k):
        return ContextDecision(
            need=True,
            tool="branches",
            query="Ленина",
            subject="филиалы",
            branch_slugs=["perm_lenina"],
        )

    async def fake_load(_state):
        return progress

    async def fake_save(prog, *, persist_state=True, fields=None):
        return progress_to_state(prog)

    fake_kb = FakeKB(
        cities=[],
        city=None,
        branches=[{"slug": "perm_lenina", "address": "ул. Ленина, 1"}],
        branch=None,
    )
    monkeypatch.setattr("graph.contexter.decide_context", _decide)
    monkeypatch.setattr("graph.contexter.vector_kb", fake_kb)
    monkeypatch.setattr("graph.checker_graph.vector_kb", fake_kb)

    with (
        patch("graph.checker_graph._checker_client", client),
        patch("graph.checker_graph._load_progress", side_effect=fake_load),
        patch("graph.checker_graph._save_progress", side_effect=fake_save),
        patch("graph.checker_graph.build_context_tools", lambda script: [_Tool()]),
        patch("graph.checker_graph.settings") as mock_settings,
    ):
        mock_settings.checker_min_growth_chars = 10
        mock_settings.script_id = script.id
        mock_settings.script_version = script.version
        patch_out = await live_check_node(state, runtime=None)  # type: ignore[arg-type]

    ctx = patch_out.get("conversation_context") or {}
    assert ctx.get("dynamic_status") == DYN_READY
    assert fact in ctx.get("dynamic_text", "")
    assert ctx.get("city_slug") == "perm"
    assert ctx.get("static_text") == "Город: Пермь"
    assert patch_out.get("last_checked_partial") == text

    loaded = await ctx_store.load("local")
    assert loaded is not None
    assert loaded.dynamic_status == DYN_READY
    assert fact in loaded.dynamic_text
    assert loaded.city_slug == "perm"
    assert loaded.static_text == "Город: Пермь"


async def test_live_check_не_тянет_филиалы_пока_агент_не_выбрал(
    script, monkeypatch, _offline_context
):
    """Пока агент не выбрал branches, list_branches не зовём и список агенту пуст."""
    from graph.context import ConversationContext
    from tests.conftest import FakeKB

    text = "какие филиалы у Ленина?"
    client = FakeChecker([CheckerVerdict(reply_usable=True, step_closed=False)])
    progress = _name_progress()
    state = _state(script, partial=text, progress=progress, last_checked="")

    await _offline_context.save(
        "local",
        ConversationContext(
            static_text="Город: Пермь",
            city_slug="perm",
            city_name="Пермь",
        ),
    )

    branches = [
        {"slug": "perm_lenina", "address": "ул. Ленина, 1", "landmark": "центр"},
        {"slug": "perm_mira", "address": "ул. Мира, 2"},
    ]
    fake_kb = FakeKB(cities=[], city=None, branches=branches, branch=None)
    seen: dict[str, Any] = {}

    async def _decide(reply, context, tools, *, branches=(), agent=None):
        seen["branches"] = list(branches)
        return ContextDecision(need=False)

    async def fake_load(_state):
        return progress

    async def fake_save(prog, *, persist_state=True, fields=None):
        return progress_to_state(prog)

    monkeypatch.setattr("graph.contexter.decide_context", _decide)
    monkeypatch.setattr("graph.contexter.vector_kb", fake_kb)
    monkeypatch.setattr("graph.checker_graph.vector_kb", fake_kb)

    with (
        patch("graph.checker_graph._checker_client", client),
        patch("graph.checker_graph._load_progress", side_effect=fake_load),
        patch("graph.checker_graph._save_progress", side_effect=fake_save),
        patch("graph.checker_graph.settings") as mock_settings,
    ):
        mock_settings.checker_min_growth_chars = 10
        mock_settings.script_id = script.id
        mock_settings.script_version = script.version
        await live_check_node(state, runtime=None)  # type: ignore[arg-type]

    assert "list_branches" not in fake_kb.calls
    assert seen["branches"] == []


async def test_live_check_при_выбранном_филиале_список_не_тянется(
    script, monkeypatch, _offline_context
):
    """Если филиал уже выбран — list_branches для агента не зовём."""
    from graph.context import ConversationContext
    from tests.conftest import FakeKB

    text = "а адрес какой?"
    client = FakeChecker([CheckerVerdict(reply_usable=True, step_closed=False)])
    progress = _name_progress()
    state = _state(script, partial=text, progress=progress, last_checked="")

    await _offline_context.save(
        "local",
        ConversationContext(
            static_text="Город: Пермь",
            city_slug="perm",
            city_name="Пермь",
            branch_slug="perm_lenina",
        ),
    )

    fake_kb = FakeKB(
        cities=[],
        city=None,
        branches=[{"slug": "perm_lenina", "address": "ул. Ленина, 1"}],
        branch={"slug": "perm_lenina", "address": "ул. Ленина, 1"},
    )
    seen: dict[str, Any] = {}

    async def _decide(reply, context, tools, *, branches=(), agent=None):
        seen["branches"] = list(branches)
        return ContextDecision(need=False)

    async def fake_load(_state):
        return progress

    async def fake_save(prog, *, persist_state=True, fields=None):
        return progress_to_state(prog)

    monkeypatch.setattr("graph.contexter.decide_context", _decide)
    monkeypatch.setattr("graph.contexter.vector_kb", fake_kb)
    monkeypatch.setattr("graph.checker_graph.vector_kb", fake_kb)

    with (
        patch("graph.checker_graph._checker_client", client),
        patch("graph.checker_graph._load_progress", side_effect=fake_load),
        patch("graph.checker_graph._save_progress", side_effect=fake_save),
        patch("graph.checker_graph.settings") as mock_settings,
    ):
        mock_settings.checker_min_growth_chars = 10
        mock_settings.script_id = script.id
        mock_settings.script_version = script.version
        await live_check_node(state, runtime=None)  # type: ignore[arg-type]

    assert "list_branches" not in fake_kb.calls
    assert seen["branches"] == []


async def test_live_contexter_до_check_pass(script, monkeypatch, _offline_context):
    """Контекстер вызывается до check_pass; _lookup_in_live больше нет."""
    from graph.context import ConversationContext

    text = "пока ещё думаю над ответом длинный"
    progress = _name_progress()
    state = _state(script, partial=text, progress=progress, last_checked="")
    await _offline_context.save("local", ConversationContext())

    order: list[str] = []

    async def fake_load(_state):
        return progress

    async def fake_save(prog, *, persist_state=True, fields=None):
        return progress_to_state(prog)

    async def fake_contexter(ctx, **kwargs):
        order.append("contexter")
        assert "check" not in order
        return ctx

    class _OrderChecker:
        async def judge(self, **kwargs):
            order.append("check")
            return CheckerVerdict(reply_usable=True, step_closed=False)

    async def fake_warmup(*args, **kwargs):
        order.append("warmup")
        return kwargs["ctx"]

    assert not hasattr(__import__("graph.checker_graph", fromlist=["x"]), "_lookup_in_live")

    monkeypatch.setattr("graph.checker_graph.run_contexter", fake_contexter)

    with (
        patch("graph.checker_graph._checker_client", _OrderChecker()),
        patch("graph.checker_graph._load_progress", side_effect=fake_load),
        patch("graph.checker_graph._save_progress", side_effect=fake_save),
        patch("graph.checker_graph._warmup_next_step", side_effect=fake_warmup),
        patch("graph.checker_graph.settings") as mock_settings,
    ):
        mock_settings.checker_min_growth_chars = 10
        mock_settings.script_id = script.id
        mock_settings.script_version = script.version
        mock_settings.pending_steps_soft_cap = 4
        await live_check_node(state, runtime=None)  # type: ignore[arg-type]

    assert order == ["contexter", "check", "warmup"]


async def test_live_check_контекстер_не_пишет_город_и_филиал_в_профиль(
    script, monkeypatch, _offline_context
):
    """Слаги города и филиала уходят в контекст/состояние, не в форму профиля."""
    from graph.context import ConversationContext
    from graph.profile_agent import ProfileGuess

    text = "я из Перми, филиал на Ленина и ещё символов"
    progress = _name_progress()
    state = _state(script, partial=text, progress=progress, last_checked="", profile={})
    await _offline_context.save("local", ConversationContext())

    async def fake_load(_state):
        return progress

    async def fake_save(prog, *, persist_state=True, fields=None):
        return progress_to_state(prog)

    async def fake_contexter(ctx, **kwargs):
        # Форму контекстер видит аргументом — но сам её не заполняет.
        assert "profile" in kwargs
        return ctx.model_copy(
            update={
                "city_slug": "perm",
                "city_name": "Пермь",
                "branch_slug": "perm_lenina",
            }
        )

    async def fake_warmup(*args, **kwargs):
        return kwargs["ctx"]

    async def fake_guess(*_a, **_k):
        return ProfileGuess()

    monkeypatch.setattr("graph.checker_graph.run_contexter", fake_contexter)
    monkeypatch.setattr("graph.checker_graph.guess_profile", fake_guess)

    with (
        patch("graph.checker_graph._checker_client", FakeChecker([None])),
        patch("graph.checker_graph._load_progress", side_effect=fake_load),
        patch("graph.checker_graph._save_progress", side_effect=fake_save),
        patch("graph.checker_graph._warmup_next_step", side_effect=fake_warmup),
        patch("graph.checker_graph.settings") as mock_settings,
    ):
        mock_settings.checker_min_growth_chars = 10
        mock_settings.script_id = script.id
        mock_settings.script_version = script.version
        mock_settings.pending_steps_soft_cap = 4
        out = await live_check_node(state, runtime=None)  # type: ignore[arg-type]

    assert out.get("city_slug") == "perm"
    assert out.get("city_name") == "Пермь"
    assert out.get("branch_slug") == "perm_lenina"
    profile = out.get("profile") or {}
    assert not str(profile.get("city") or "").strip()
    assert not str(profile.get("branch") or "").strip()
    ctx = out.get("conversation_context") or {}
    assert ctx.get("city_slug") == "perm"
    assert ctx.get("branch_slug") == "perm_lenina"


async def test_live_без_разбора_статус_не_трогает(script, monkeypatch, _offline_context):
    """Ход без потребностей и без похода агента — статус не меняем."""
    from graph.context import DYN_READY, ConversationContext

    text = "пока ещё думаю над ответом длинный"
    progress = _name_progress()
    state = _state(script, partial=text, progress=progress, last_checked="")

    seeded = ConversationContext(
        static_text="статика",
        dynamic_status=DYN_READY,
        dynamic_text="уже было",
        dynamic_turn=1,
        pending_fields=[],
    )
    await _offline_context.save("local", seeded)

    async def fake_load(_state):
        return progress

    async def fake_save(prog, *, persist_state=True, fields=None):
        return progress_to_state(prog)

    async def fake_warmup(*args, **kwargs):
        return kwargs["ctx"]

    with (
        patch("graph.checker_graph._checker_client", FakeChecker([None])),
        patch("graph.checker_graph._load_progress", side_effect=fake_load),
        patch("graph.checker_graph._save_progress", side_effect=fake_save),
        patch("graph.checker_graph._warmup_next_step", side_effect=fake_warmup),
        patch("graph.checker_graph.settings") as mock_settings,
    ):
        mock_settings.checker_min_growth_chars = 10
        mock_settings.script_id = script.id
        mock_settings.script_version = script.version
        mock_settings.pending_steps_soft_cap = 4
        out = await live_check_node(state, runtime=None)  # type: ignore[arg-type]

    loaded = await _offline_context.load("local")
    assert loaded is not None
    assert loaded.dynamic_status == DYN_READY
    assert loaded.dynamic_text == "уже было"
    final = out.get("conversation_context") or {}
    assert final.get("dynamic_status") == DYN_READY


async def test_live_lookup_ошибка_даёт_missing(script, monkeypatch, _offline_context):
    from graph.context import DYN_MISSING, ConversationContext
    from graph.contexter import reply_hash
    from tests.conftest import FakeKB

    text = "Пермь"
    client = FakeChecker([CheckerVerdict(reply_usable=True, step_closed=False)])
    progress = ScriptProgress(
        status={"name": "closed", "city": "pending"},
        attempts={"city": 1},
        taken_turn={"city": 1},
        profile={"caller_name": "Мария"},
    )
    state = _state(
        script,
        partial=text,
        progress=progress,
        profile={"caller_name": "Мария"},
        turn=2,
    )
    state["current_step"] = "city"
    state["head_steps"] = ["city"]
    await _offline_context.save("local", ConversationContext())

    class BoomKB(FakeKB):
        async def list_cities(self):
            raise RuntimeError("kb down")

    fake_kb = BoomKB(cities=[], city=None, branches=[], branch=None)

    async def fake_load(_state):
        return progress

    async def fake_save(prog, *, persist_state=True, fields=None):
        return progress_to_state(prog)

    async def _decide(*_a, **_k):
        return ContextDecision(need=False)

    monkeypatch.setattr("graph.contexter.decide_context", _decide)
    monkeypatch.setattr("graph.contexter.vector_kb", fake_kb)
    monkeypatch.setattr("graph.tools_registry.vector_kb", fake_kb)
    monkeypatch.setattr("graph.checker_graph.vector_kb", fake_kb)
    # v2-шаг city без needs — принудительно даём потребность, как у продаж.
    monkeypatch.setattr(
        "graph.checker_graph._head_needs",
        lambda *a, **k: ["city_choices"],
    )

    with (
        patch("graph.checker_graph._checker_client", client),
        patch("graph.checker_graph._load_progress", side_effect=fake_load),
        patch("graph.checker_graph._save_progress", side_effect=fake_save),
        patch("graph.checker_graph.settings") as mock_settings,
    ):
        mock_settings.checker_min_growth_chars = 10
        mock_settings.script_id = script.id
        mock_settings.script_version = script.version
        mock_settings.pending_steps_soft_cap = 4
        out = await live_check_node(state, runtime=None)  # type: ignore[arg-type]

    ctx = out.get("conversation_context") or {}
    assert ctx.get("dynamic_status") == DYN_MISSING
    assert ctx.get("dynamic_reply_hash") == reply_hash(text)
    loaded = await _offline_context.load("local")
    assert loaded is not None
    assert loaded.dynamic_reply_hash == reply_hash(text)


async def test_live_профиль_попадает_в_кеш_чекера(script, monkeypatch, _offline_context):
    """Разбор профиля после check_pass пишется набором PROGRESS_FIELDS_CHECKER."""
    from graph.profile_agent import ProfileGuess, ProfileValue
    from script.store import MemoryScriptStore

    text = "Меня зовут Андрей Андреевич"
    progress = _name_progress()
    state = _state(script, partial=text, progress=progress, last_checked="")

    mem = MemoryScriptStore()
    await mem.save("local", progress)

    async def fake_guess(reply, *, known, fields, history=(), agent=None):
        return ProfileGuess(values=[ProfileValue(key="caller_name", value="Андрей")])

    async def fake_warmup(*args, **kwargs):
        return kwargs["ctx"]

    monkeypatch.setattr("graph.checker_graph.guess_profile", fake_guess)
    monkeypatch.setattr("graph.nodes.script_store", mem)

    with (
        patch("graph.checker_graph._checker_client", FakeChecker([None])),
        patch("graph.checker_graph._warmup_next_step", side_effect=fake_warmup),
        patch("graph.checker_graph.settings") as mock_settings,
    ):
        mock_settings.checker_min_growth_chars = 10
        mock_settings.script_id = script.id
        mock_settings.script_version = script.version
        mock_settings.pending_steps_soft_cap = 4
        out = await live_check_node(state, runtime=None)  # type: ignore[arg-type]

    assert out.get("profile", {}).get("caller_name") == "Андрей"
    stored = await mem.load("local")
    assert stored is not None
    assert stored.profile.get("caller_name") == "Андрей"


def _branch_pending_progress() -> ScriptProgress:
    """Прогресс: ведущий шаг ``branch`` уже взят, следующий — ``price``."""
    closed = [
        "name",
        "city",
        "who_studies",
        "experience",
        "transmission",
        "terms",
        "theory_format",
        "included",
        "practice",
    ]
    return ScriptProgress(
        status={**{c: "closed" for c in closed}, "branch": "pending"},
        attempts={**{c: 1 for c in closed}, "branch": 1},
    )


def _branch_profile() -> dict[str, str]:
    """Профиль, при котором ``branch`` доступен."""
    return {
        "caller_name": "Андрей",
        "city": "Пермь",
        "student_is_caller": "да",
        "experience": "впервые",
        "transmission": "механика",
        "theory_format": "очно",
    }


async def test_warmup_без_current_step_греет_следующий(script, fake_kb, monkeypatch):
    """Без ``current_step`` прогрев целится в следующий шаг, не в ведущий."""
    from core.config import settings

    progress = _branch_pending_progress()
    profile = _branch_profile()
    soft_cap = settings.pending_steps_soft_cap

    lead = pick_step(
        script,
        status=progress.status,
        profile=profile,
        attempts=progress.attempts,
        inform_reason=False,
        pending_soft_cap=soft_cap,
    )
    assert lead is not None and lead.id == "branch"
    assert needs_of(lead) == ["branches"]
    nxt = peek_next_step(
        script,
        current=lead,
        status=progress.status,
        profile=profile,
        attempts=progress.attempts,
        inform_reason=False,
        pending_soft_cap=soft_cap,
    )
    assert nxt is not None and nxt.id == "price"
    assert needs_of(nxt) == ["price"]

    monkeypatch.setattr("graph.checker_graph.vector_kb", fake_kb)
    state: dict[str, Any] = {
        "script_id": script.id,
        "script_version": script.version,
        "city_slug": "perm",
        "city_name": "Пермь",
        "profile": profile,
    }
    ctx = ConversationContext()
    out = await _warmup_next_step(
        state,
        progress=progress,
        profile=profile,
        ctx=ctx,
        asks_inform=False,
    )

    # Ведущий branch → branches; следующий price → get_city. Без current_step
    # раньше грели бы branch (list_branches); теперь — price.
    assert "list_branches" not in fake_kb.calls
    assert "get_city" in fake_kb.calls
    assert out is not None


async def test_warmup_ошибка_справочника_не_роняет(script, monkeypatch):
    """Исключение справочника глотается: функция возвращает контекст."""

    class BoomKB:
        """Заглушка, которая всегда бросает."""

        async def list_cities(self):
            raise RuntimeError("kb down")

        async def get_city(self, city_slug: str):
            raise RuntimeError("kb down")

        async def list_branches(self, city_slug: str):
            raise RuntimeError("kb down")

        async def get_branch(self, branch_slug: str):
            raise RuntimeError("kb down")

    monkeypatch.setattr("graph.checker_graph.vector_kb", BoomKB())
    progress = _branch_pending_progress()
    profile = _branch_profile()
    state: dict[str, Any] = {
        "script_id": script.id,
        "script_version": script.version,
        "city_slug": "perm",
        "city_name": "Пермь",
        "profile": profile,
    }
    ctx = ConversationContext()
    out = await _warmup_next_step(
        state,
        progress=progress,
        profile=profile,
        ctx=ctx,
        asks_inform=False,
    )
    assert out is ctx


async def test_warmup_ошибка_lead_from_progress_не_роняет(script, monkeypatch):
    """Исключение ``_lead_from_progress`` глотается: возвращается переданный ctx."""

    def boom(_state, *, progress, profile):
        raise RuntimeError("lead broken")

    monkeypatch.setattr("graph.checker_graph._lead_from_progress", boom)
    progress = _branch_pending_progress()
    profile = _branch_profile()
    state: dict[str, Any] = {
        "script_id": script.id,
        "script_version": script.version,
        "city_slug": "perm",
        "city_name": "Пермь",
        "profile": profile,
    }
    ctx = ConversationContext()
    out = await _warmup_next_step(
        state,
        progress=progress,
        profile=profile,
        ctx=ctx,
        asks_inform=False,
    )
    assert out is ctx


async def test_live_ставит_в_работе_первым_действием(script, monkeypatch, _offline_context):
    """«в работе» пишется в кеш до контекстера; на выходе — конечный статус."""
    from graph.context import DYN_READY, DYN_WORKING, ConversationContext
    from graph.contexter import reply_hash

    text = "пока ещё думаю над ответом длинный"
    progress = _name_progress()
    state = _state(script, partial=text, progress=progress, last_checked="")
    await _offline_context.save("local", ConversationContext(static_text="статика"))

    seen_before_contexter: list[str] = []

    async def spy_contexter(ctx, **kwargs):
        loaded = await _offline_context.load("local")
        assert loaded is not None
        seen_before_contexter.append(loaded.dynamic_status)
        assert loaded.dynamic_reply_hash == reply_hash(text)
        return ctx.model_copy(update={"dynamic_status": DYN_READY})

    async def fake_load(_state):
        return progress

    async def fake_save(prog, *, persist_state=True, fields=None):
        return progress_to_state(prog)

    async def fake_warmup(*args, **kwargs):
        return kwargs["ctx"]

    monkeypatch.setattr("graph.checker_graph.run_contexter", spy_contexter)

    with (
        patch("graph.checker_graph._checker_client", FakeChecker([None])),
        patch("graph.checker_graph._load_progress", side_effect=fake_load),
        patch("graph.checker_graph._save_progress", side_effect=fake_save),
        patch("graph.checker_graph._warmup_next_step", side_effect=fake_warmup),
        patch("graph.checker_graph.settings") as mock_settings,
    ):
        mock_settings.checker_min_growth_chars = 10
        mock_settings.script_id = script.id
        mock_settings.script_version = script.version
        mock_settings.pending_steps_soft_cap = 4
        out = await live_check_node(state, runtime=None)  # type: ignore[arg-type]

    assert seen_before_contexter == [DYN_WORKING]
    loaded = await _offline_context.load("local")
    assert loaded is not None
    assert loaded.dynamic_status == DYN_READY
    assert loaded.dynamic_status != DYN_WORKING
    assert (out.get("conversation_context") or {}).get("dynamic_status") == DYN_READY


async def test_live_исключение_снимает_в_работе(script, monkeypatch, _offline_context):
    """При исключении после «в работе» статус сменяется на конечный."""
    from graph.context import DYN_READY, DYN_WORKING, ConversationContext
    from graph.contexter import reply_hash

    text = "пока ещё думаю над ответом длинный"
    progress = _name_progress()
    state = _state(script, partial=text, progress=progress, last_checked="")
    await _offline_context.save(
        "local",
        ConversationContext(static_text="статика", dynamic_status=DYN_READY),
    )

    async def boom_contexter(ctx, **kwargs):
        loaded = await _offline_context.load("local")
        assert loaded is not None
        assert loaded.dynamic_status == DYN_WORKING
        assert loaded.dynamic_reply_hash == reply_hash(text)
        raise RuntimeError("contexter failed")

    async def fake_load(_state):
        return progress

    monkeypatch.setattr("graph.checker_graph.run_contexter", boom_contexter)

    with (
        patch("graph.checker_graph._checker_client", FakeChecker([None])),
        patch("graph.checker_graph._load_progress", side_effect=fake_load),
        patch("graph.checker_graph.settings") as mock_settings,
    ):
        mock_settings.checker_min_growth_chars = 10
        mock_settings.script_id = script.id
        mock_settings.script_version = script.version
        mock_settings.pending_steps_soft_cap = 4
        with pytest.raises(RuntimeError, match="contexter failed"):
            await live_check_node(state, runtime=None)  # type: ignore[arg-type]

    loaded = await _offline_context.load("local")
    assert loaded is not None
    assert loaded.dynamic_status == DYN_READY
    assert loaded.dynamic_status != DYN_WORKING
