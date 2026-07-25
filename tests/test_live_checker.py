"""Тесты служебного чекера и общего ядра ``check_pass``."""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from graph.checker import CheckerVerdict, check_pass, history_slice_for, run_checker
from graph.checker_graph import growth_below_threshold, live_check_node
from script.planner import script_head
from script.store import ScriptProgress, progress_to_state


class FakeChecker:
    """Заглушка модели чекера."""

    def __init__(self, verdicts: list[CheckerVerdict | None]) -> None:
        self.verdicts = list(verdicts)
        self.calls: list[dict] = []

    async def judge(self, *, history_slice, client_reply, step, step_text):
        self.calls.append(
            {
                "history_slice": history_slice,
                "client_reply": client_reply,
                "step_id": step.id,
                "step_text": step_text,
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
        "last_checked_partial": last_checked,
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


async def test_отрицательный_прирост_не_пропускает_проход():
    """Новая реплика (прирост < 0) — не skip, а сброс точки отсчёта."""
    assert not growth_below_threshold("да", "Меня зовут Андрей Петров", min_growth=10)
    assert not growth_below_threshold("к", "длинный прошлый текст", min_growth=10)


async def test_прирост_внутри_реплики_от_разобранного_ранее():
    """Внутри одной реплики порог считается от last_checked этой же реплики."""
    previous = "Меня зовут"
    # +2 символа — ниже порога.
    assert growth_below_threshold(previous + " А", previous, min_growth=10)
    # +12 — выше порога.
    assert not growth_below_threshold(previous + " Андрей Петр", previous, min_growth=10)


async def test_live_check_отрицательный_прирост_сбрасывает_и_зовёт_чекер(script):
    """Буфер сброшен: прирост отрицательный → разбор как новой реплики."""
    text = "да"
    client = FakeChecker([CheckerVerdict(reply_usable=True, step_closed=False)])
    progress = _name_progress()
    state = _state(
        script,
        partial=text,
        progress=progress,
        last_checked="Меня зовут Андрей очень длинный прошлый",
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


async def test_live_check_контекстер_кладёт_справку_в_контекст(script):
    """После чекера контекстер печёт справку в conversation_context."""
    from graph.context import DYN_READY

    text = "а когда медкомиссию проходить?"
    client = FakeChecker([CheckerVerdict(reply_usable=True, step_closed=False)])
    progress = _name_progress()
    state = _state(
        script,
        partial=text,
        progress=progress,
        last_checked="",
    )
    state["conversation_context"] = {"static_text": "Город: Пермь"}

    async def fake_load(_state):
        return progress

    async def fake_save(prog, *, persist_state=True, fields=None):
        return progress_to_state(prog)

    with (
        patch("graph.checker_graph._checker_client", client),
        patch("graph.checker_graph._load_progress", side_effect=fake_load),
        patch("graph.checker_graph._save_progress", side_effect=fake_save),
        patch("graph.checker_graph.settings") as mock_settings,
    ):
        mock_settings.checker_min_growth_chars = 10
        mock_settings.script_id = script.id
        mock_settings.script_version = script.version
        patch_out = await live_check_node(state, runtime=None)  # type: ignore[arg-type]

    ctx = patch_out.get("conversation_context") or {}
    assert ctx.get("dynamic_status") == DYN_READY
    assert script.helps["medcheck"].text in ctx.get("dynamic_text", "")
    assert patch_out.get("last_checked_partial") == text
