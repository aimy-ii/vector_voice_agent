"""Тесты точечной записи прогресса в кеш."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from graph import nodes as nodes_module
from script.store import (
    PROGRESS_FIELDS_CHECKER,
    PROGRESS_FIELDS_GENERATOR,
    MemoryScriptStore,
    ScriptProgress,
    merge_progress_fields,
)


@pytest.fixture()
def memory_store(monkeypatch) -> MemoryScriptStore:
    mem = MemoryScriptStore()
    monkeypatch.setattr(nodes_module, "script_store", mem)
    return mem


def test_merge_progress_fields_накладывает_только_свои():
    base = ScriptProgress(
        status={"name": "closed"},
        attempts={"name": 1, "city": 2},
        taken_turn={"name": 1},
        profile={"caller_name": "Андрей"},
    )
    overlay = ScriptProgress(
        status={"city": "closed"},
        attempts={"city": 99},
        taken_turn={"city": 3},
        profile={"city": "Пермь"},
    )
    checker = merge_progress_fields(base, overlay, PROGRESS_FIELDS_CHECKER)
    assert checker.status["name"] == "closed"
    assert checker.status["city"] == "closed"
    assert checker.attempts["city"] == 2  # attempts не трогали
    assert checker.profile["caller_name"] == "Андрей"
    assert checker.profile["city"] == "Пермь"

    generator = merge_progress_fields(base, overlay, PROGRESS_FIELDS_GENERATOR)
    assert generator.attempts["city"] == 99
    assert generator.taken_turn["city"] == 3
    assert generator.status.get("city") != "closed"  # status не трогали


@pytest.mark.asyncio
async def test_чекер_не_затирает_счётчик_генератора(memory_store: MemoryScriptStore):
    """Чекер пишет status — attempts генератора в кеше сохраняются."""
    await memory_store.save(
        "local",
        ScriptProgress(status={"name": "pending"}, attempts={"name": 1}, taken_turn={"name": 1}),
    )

    async def _fake_call_id() -> str:
        return "local"

    with patch.object(nodes_module, "_call_id", return_value="local"):
        # Генератор поставил вторую попытку.
        gen = ScriptProgress(
            status={"name": "pending"},
            attempts={"name": 2},
            taken_turn={"name": 1},
        )
        await nodes_module._save_progress(gen, fields=PROGRESS_FIELDS_GENERATOR)

        # Чекер закрыл шаг со старым attempts=1 в локальной копии.
        chk = ScriptProgress(
            status={"name": "closed"},
            attempts={"name": 1},
            taken_turn={"name": 1},
        )
        await nodes_module._save_progress(chk, fields=PROGRESS_FIELDS_CHECKER)

    loaded = await memory_store.load("local")
    assert loaded is not None
    assert loaded.status["name"] == "closed"
    assert loaded.attempts["name"] == 2


@pytest.mark.asyncio
async def test_генератор_не_затирает_статус_чекера(memory_store: MemoryScriptStore):
    """Генератор пишет attempts — closed от чекера сохраняется."""
    await memory_store.save(
        "local",
        ScriptProgress(status={"name": "closed"}, attempts={"name": 1}, taken_turn={"name": 1}),
    )
    with patch.object(nodes_module, "_call_id", return_value="local"):
        gen = ScriptProgress(
            status={"name": "pending"},
            attempts={"name": 2, "city": 1},
            taken_turn={"name": 1, "city": 2},
        )
        await nodes_module._save_progress(gen, fields=PROGRESS_FIELDS_GENERATOR)

    loaded = await memory_store.load("local")
    assert loaded is not None
    assert loaded.status["name"] == "closed"
    assert loaded.attempts["name"] == 2
    assert loaded.attempts["city"] == 1
