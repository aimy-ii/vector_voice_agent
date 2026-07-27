"""Тесты идентификатора звонка и общего стека скрипта между каналами."""

from __future__ import annotations

import pytest

from graph import nodes as nodes_module
from script.store import MemoryScriptStore, ScriptProgress


def test_call_id_из_конфига(monkeypatch: pytest.MonkeyPatch) -> None:
    """Непустой ``call_id`` в configurable побеждает ``thread_id``."""
    monkeypatch.setattr(
        nodes_module,
        "get_config",
        lambda: {"configurable": {"call_id": "call-42", "thread_id": "other"}},
    )
    assert nodes_module._call_id() == "call-42"


def test_call_id_отбрасывает_суффикс_лайв_треда(monkeypatch: pytest.MonkeyPatch) -> None:
    """``abc-live`` → ``abc`` при суффиксе из настроек."""
    monkeypatch.setattr(nodes_module.settings, "live_thread_suffix", "-live")
    monkeypatch.setattr(
        nodes_module,
        "get_config",
        lambda: {"configurable": {"thread_id": "abc-live"}},
    )
    assert nodes_module._call_id() == "abc"


def test_call_id_без_суффикса_как_есть(monkeypatch: pytest.MonkeyPatch) -> None:
    """``thread_id`` без суффикса лайв-треда возвращается целиком."""
    monkeypatch.setattr(nodes_module.settings, "live_thread_suffix", "-live")
    monkeypatch.setattr(
        nodes_module,
        "get_config",
        lambda: {"configurable": {"thread_id": "abc"}},
    )
    assert nodes_module._call_id() == "abc"


def test_call_id_local_без_конфига(monkeypatch: pytest.MonkeyPatch) -> None:
    """Без конфига LangGraph — ``local``."""

    def _boom() -> dict:
        raise RuntimeError("no config")

    monkeypatch.setattr(nodes_module, "get_config", _boom)
    assert nodes_module._call_id() == "local"


@pytest.mark.asyncio
async def test_общий_стек_скрипта_между_каналами(
    monkeypatch: pytest.MonkeyPatch,
    memory_store: MemoryScriptStore,
) -> None:
    """Прогресс под ``abc`` виден под ``abc-live`` и наоборот."""
    monkeypatch.setattr(nodes_module, "script_store", memory_store)
    monkeypatch.setattr(nodes_module.settings, "live_thread_suffix", "-live")

    progress = ScriptProgress(status={"name": "closed"}, attempts={"name": 1})
    monkeypatch.setattr(
        nodes_module,
        "get_config",
        lambda: {"configurable": {"thread_id": "abc"}},
    )
    await nodes_module._save_progress(progress, persist_state=False)

    monkeypatch.setattr(
        nodes_module,
        "get_config",
        lambda: {"configurable": {"thread_id": "abc-live"}},
    )
    loaded = await nodes_module._load_progress({})
    assert loaded.status["name"] == "closed"
    assert loaded.attempts["name"] == 1
    assert "abc" in memory_store._data
    assert "abc-live" not in memory_store._data

    progress_live = ScriptProgress(status={"city": "pending"}, attempts={"city": 2})
    await nodes_module._save_progress(progress_live, persist_state=False)

    monkeypatch.setattr(
        nodes_module,
        "get_config",
        lambda: {"configurable": {"thread_id": "abc"}},
    )
    loaded_back = await nodes_module._load_progress({})
    assert loaded_back.status["city"] == "pending"
    assert loaded_back.attempts["city"] == 2
