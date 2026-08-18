"""Офлайн-тесты строки итога служебного прохода про постановку в очередь."""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

import pytest

from graph.checker_graph import live_check_node
from graph.context import ConversationContext
from graph.context_store import MemoryContextStore
from graph.farewell_agent import FarewellDecision
from graph.profile_agent import ProfileGuess
from script.store import progress_to_state
from tests.test_live_checker import FakeChecker, _name_progress, _state


async def _live_check_done_texts(
    script: Any,
    monkeypatch: pytest.MonkeyPatch,
    *,
    queued: bool,
) -> list[str]:
    """Гоняет live_check_node и возвращает тексты stage live-check/done."""
    mem = MemoryContextStore()
    monkeypatch.setattr("graph.nodes.context_store", mem)
    await mem.save("local", ConversationContext())

    progress = _name_progress()
    text = "пока ещё думаю над ответом длинный"
    state = _state(script, partial=text, progress=progress, last_checked="")
    stages: list[tuple] = []

    async def fake_load(_state: object) -> object:
        return progress

    async def fake_save(prog: object, *, persist_state: bool = True, fields: object = None) -> dict:
        _ = (persist_state, fields)
        return progress_to_state(prog)  # type: ignore[arg-type]

    async def fake_warmup(*_args: object, **kwargs: Any) -> Any:
        return kwargs["ctx"]

    async def _enqueue(*_args: object, **_kwargs: object) -> bool:
        return queued

    async def _counted(ctx: ConversationContext, **_kwargs: object) -> ConversationContext:
        return ctx

    async def _no_profile(*_args: object, **_kwargs: object) -> ProfileGuess:
        return ProfileGuess()

    async def _no_farewell(*_args: object, **_kwargs: object) -> FarewellDecision:
        return FarewellDecision(conversation_ended=False)

    def _stage(name: str, text: str, kind: str = "done", **_kwargs: object) -> None:
        stages.append((name, text, kind))

    monkeypatch.setattr("graph.checker_graph._enqueue_contexter", _enqueue)
    monkeypatch.setattr("graph.checker_graph.run_contexter", _counted)
    monkeypatch.setattr("graph.checker_graph.guess_profile", _no_profile)
    monkeypatch.setattr("graph.checker_graph.decide_farewell", _no_farewell)

    with (
        patch("graph.checker_graph._checker_client", FakeChecker([None])),
        patch("graph.checker_graph._load_progress", side_effect=fake_load),
        patch("graph.checker_graph._save_progress", side_effect=fake_save),
        patch("graph.checker_graph._warmup_next_step", side_effect=fake_warmup),
        patch("graph.checker_graph.stage", side_effect=_stage),
        patch("graph.checker_graph.settings") as mock_settings,
    ):
        mock_settings.checker_min_growth_chars = 10
        mock_settings.farewell_min_messages = 5
        mock_settings.script_id = script.id
        mock_settings.script_version = script.version
        mock_settings.pending_steps_soft_cap = 4
        mock_settings.live_thread_suffix = "-live"
        await live_check_node(state, runtime=None)  # type: ignore[arg-type]

    return [t for n, t, k in stages if n == "live-check" and k == "done"]


async def test_итог_прохода_очередь_пишет_в_очереди(
    script: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Постановка удалась: в строке live-check/done есть «в очереди»."""
    done = await _live_check_done_texts(script, monkeypatch, queued=True)

    assert done
    assert "в очереди" in done[0]


async def test_итог_прохода_синхронно_без_в_очереди(
    script: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Постановка не удалась: в строке live-check/done нет «в очереди»."""
    done = await _live_check_done_texts(script, monkeypatch, queued=False)

    assert done
    assert "в очереди" not in done[0]
