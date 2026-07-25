"""Тесты хранилища прогресса скрипта в Redis (заглушка)."""

from __future__ import annotations

import pytest

from script.store import (
    MemoryScriptStore,
    ScriptProgress,
    progress_from_state,
    progress_to_state,
)


@pytest.mark.asyncio
async def test_скрипт_пишется_и_читается_по_звонку(memory_store: MemoryScriptStore):
    progress = ScriptProgress(status={"name": "pending"}, attempts={"name": 1})
    assert await memory_store.save("t1", progress) is True
    loaded = await memory_store.load("t1")
    assert loaded is not None
    assert loaded.status["name"] == "pending"
    assert loaded.attempts["name"] == 1


@pytest.mark.asyncio
async def test_недоступный_редис_не_роняет_ход(memory_store: MemoryScriptStore):
    memory_store.fail = True
    assert await memory_store.save("t1", ScriptProgress()) is False
    assert await memory_store.load("t1") is None


@pytest.mark.asyncio
async def test_промах_восстанавливается_из_состояния_треда():
    state = {
        "step_status": {"city": "closed"},
        "step_attempts": {"city": 1},
        "step_taken_turn": {"city": 2},
    }
    progress = progress_from_state(state)
    assert progress.status["city"] == "closed"
    assert progress.attempts["city"] == 1
    assert progress.taken_turn["city"] == 2


def test_слепок_ложится_в_поля_треда():
    progress = ScriptProgress(status={"name": "closed"}, attempts={"name": 2})
    patch = progress_to_state(progress)
    assert patch["step_status"]["name"] == "closed"
    assert patch["script_progress"]["attempts"]["name"] == 2


@pytest.mark.asyncio
async def test_наследие_v1_статусов_нормализуется():
    progress = ScriptProgress.from_mapping(
        {"status": {"city": "done", "name": "open", "price": "skipped"}}
    )
    assert progress.status["city"] == "closed"
    assert progress.status["name"] == "pending"
    assert progress.status["price"] == "closed"
