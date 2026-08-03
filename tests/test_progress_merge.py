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
    from graph.context_store import MemoryContextStore

    mem = MemoryScriptStore()
    monkeypatch.setattr(nodes_module, "script_store", mem)
    monkeypatch.setattr(nodes_module, "context_store", MemoryContextStore())
    return mem


def test_merge_progress_fields_накладывает_только_свои():
    base = ScriptProgress(
        status={"name": "closed"},
        attempts={"name": 1, "city": 2},
        taken_turn={"name": 1},
        in_work=["name", "city"],
        profile={"caller_name": "Андрей"},
    )
    overlay = ScriptProgress(
        status={"city": "closed"},
        attempts={"city": 99},
        taken_turn={"city": 3},
        in_work=["who_studies"],
        profile={"city": "Пермь"},
    )
    checker = merge_progress_fields(base, overlay, PROGRESS_FIELDS_CHECKER)
    assert checker.status["name"] == "closed"
    assert checker.status["city"] == "closed"
    assert checker.attempts["city"] == 2  # attempts не трогали
    assert checker.in_work == ["name", "city"]
    assert checker.profile["caller_name"] == "Андрей"
    assert checker.profile["city"] == "Пермь"

    generator = merge_progress_fields(base, overlay, PROGRESS_FIELDS_GENERATOR)
    assert generator.attempts["city"] == 99
    assert generator.taken_turn["city"] == 3
    assert "who_studies" in generator.in_work
    assert "name" in generator.in_work
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


@pytest.mark.asyncio
async def test_последовательные_точечные_записи_сохраняют_оба_набора(
    memory_store: MemoryScriptStore,
):
    """Как после параллельного check∥lookup: чекер, затем генератор в plan."""
    await memory_store.save(
        "local",
        ScriptProgress(status={"name": "pending"}, attempts={"name": 1}, taken_turn={"name": 1}),
    )
    with patch.object(nodes_module, "_call_id", return_value="local"):
        chk = ScriptProgress(
            status={"name": "closed"},
            attempts={"name": 1},
            taken_turn={"name": 1},
            profile={"caller_name": "Андрей"},
        )
        await nodes_module._save_progress(chk, fields=PROGRESS_FIELDS_CHECKER)
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
    assert loaded.profile.get("caller_name") == "Андрей"


def test_merge_dicts_для_параллельного_профиля():
    """Редьюсер профиля: правки check и lookup не затирают друг друга."""
    from graph.state import merge_dicts

    after_check = merge_dicts({"caller_name": "Андрей"}, {"caller_name": "Андрей"})
    after_both = merge_dicts(after_check, {"city": "Пермь"})
    assert after_both == {"caller_name": "Андрей", "city": "Пермь"}
    assert merge_dicts({"a": "1"}, None) == {"a": "1"}
    assert merge_dicts(None, {"b": "2"}) == {"b": "2"}
