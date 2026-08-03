"""Сдвиг ведущего в ``plan_node``, если шаг прошлого хода снова впереди."""

from __future__ import annotations

from typing import Any

import pytest
from langchain_core.messages import HumanMessage

from graph import nodes as nodes_module
from graph.state import new_state_defaults
from script.store import MemoryScriptStore, ScriptProgress


@pytest.fixture()
def store(monkeypatch) -> MemoryScriptStore:
    mem = MemoryScriptStore()
    monkeypatch.setattr(nodes_module, "script_store", mem)
    return mem


@pytest.fixture()
def use_v2(monkeypatch) -> None:
    monkeypatch.setattr(nodes_module.settings, "script_version", "2")


@pytest.fixture()
def plan_logs(monkeypatch) -> list[str]:
    """Собирает тексты ``[plan|done]`` без побочных эффектов stage."""
    texts: list[str] = []

    def _stage(name: str, text: str, phase: str = "done", **_kwargs: Any) -> None:
        if name == "plan" and phase == "done":
            texts.append(text)

    monkeypatch.setattr(nodes_module, "stage", _stage)
    return texts


def _base_state(**extra: Any) -> dict[str, Any]:
    return {
        **new_state_defaults(),
        "messages": [HumanMessage(content="ну не знаю")],
        "script_id": "vector_ru",
        "script_version": "2",
        "turn": 3,
        "turn_kind": "client",
        **extra,
    }


async def _seed_pending_head(store: MemoryScriptStore) -> None:
    """Шапка из двух висящих: name → city (и свежий who_studies при soft_cap≥3)."""
    await store.save(
        "local",
        ScriptProgress(
            status={"name": "pending", "city": "pending"},
            attempts={"name": 1, "city": 1},
            taken_turn={"name": 1, "city": 1},
            in_work=["name", "city"],
        ),
    )


async def test_сдвиг_ведущего_если_совпал_с_прошлым_ходом(store, use_v2, plan_logs, monkeypatch):
    """Совпал с прошлым ходом + реплика клиента — ведущий = следующий в шапке."""
    monkeypatch.setattr(nodes_module.settings, "pending_steps_soft_cap", 4)
    await _seed_pending_head(store)
    state = _base_state(current_step="name")
    out = await nodes_module.plan_node(state, None)  # type: ignore[arg-type]
    assert out["head_steps"][:2] == ["name", "city"]
    assert out["current_step"] == "city"


async def test_без_сдвига_если_ведущий_другой(store, use_v2, plan_logs):
    """Пересчитанный ведущий не совпал с прошлым — ничего не меняем."""
    await store.save(
        "local",
        ScriptProgress(
            status={"name": "closed", "city": "pending"},
            attempts={"name": 1, "city": 1},
            taken_turn={"name": 1, "city": 1},
            in_work=["name", "city"],
            profile={"caller_name": "Мария"},
        ),
    )
    state = _base_state(
        current_step="name",
        profile={"caller_name": "Мария"},
    )
    out = await nodes_module.plan_node(state, None)  # type: ignore[arg-type]
    assert out["current_step"] == "city"
    assert "сдвиг" not in (plan_logs[-1] if plan_logs else "")


async def test_один_открытый_в_шапке_ведущий_не_меняется(store, use_v2, plan_logs, monkeypatch):
    """В шапке только один открытый — сдвигать некуда."""
    monkeypatch.setattr(nodes_module.settings, "pending_steps_soft_cap", 1)
    await store.save(
        "local",
        ScriptProgress(
            status={"name": "pending"},
            attempts={"name": 1},
            taken_turn={"name": 1},
            in_work=["name"],
        ),
    )
    state = _base_state(current_step="name")
    out = await nodes_module.plan_node(state, None)  # type: ignore[arg-type]
    assert out["head_steps"] == ["name"]
    assert out["current_step"] == "name"
    assert "сдвиг" not in plan_logs[-1]


@pytest.mark.parametrize("turn_kind", ["continuation", "silence"])
async def test_без_сдвига_на_ходе_без_реплики_клиента(
    store, use_v2, plan_logs, monkeypatch, turn_kind: str
):
    """Ходы continuation / silence — ведущий не сдвигается."""
    monkeypatch.setattr(nodes_module.settings, "pending_steps_soft_cap", 4)
    await _seed_pending_head(store)
    state = _base_state(current_step="name", turn_kind=turn_kind)
    out = await nodes_module.plan_node(state, None)  # type: ignore[arg-type]
    assert out["current_step"] == "name"
    assert out["head_steps"][0] == "name"
    assert "сдвиг" not in plan_logs[-1]


async def test_лог_сдвига_содержит_прежний_шаг(store, use_v2, plan_logs, monkeypatch):
    """В ``[plan|done]`` есть пометка о сдвиге и шаг, который вёл раньше."""
    monkeypatch.setattr(nodes_module.settings, "pending_steps_soft_cap", 4)
    await _seed_pending_head(store)
    state = _base_state(current_step="name")
    out = await nodes_module.plan_node(state, None)  # type: ignore[arg-type]
    assert out["current_step"] == "city"
    assert plan_logs
    assert "сдвиг с name" in plan_logs[-1]
    assert "шаг city" in plan_logs[-1]
