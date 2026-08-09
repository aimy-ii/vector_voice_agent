"""Приём хода: история копится мозгом, снимок бота не сшивается."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from graph import nodes as nodes_module
from graph.context import ConversationContext
from graph.history import text_of
from graph.state import new_state_defaults
from graph.transcript import ROLE_AGENT, ROLE_CLIENT, TranscriptEntry


@pytest.fixture()
def ctx_store(monkeypatch):
    from graph.context_store import MemoryContextStore

    mem = MemoryContextStore()
    monkeypatch.setattr(nodes_module, "context_store", mem)
    return mem


@pytest.fixture()
def quiet(monkeypatch) -> None:
    monkeypatch.setattr(nodes_module, "stage", lambda *a, **k: None)


def _runtime(*, turn_kind: str = "client") -> SimpleNamespace:
    return SimpleNamespace(context={}, configurable={"turn_kind": turn_kind})


def _base_state(*, messages: list[Any], turn: int = 2) -> dict[str, Any]:
    return {
        **new_state_defaults(),
        "messages": messages,
        "script_id": "vector_ru",
        "script_version": "2",
        "turn": turn,
    }


@pytest.fixture()
def turn_kind(monkeypatch):
    """Подменяет ``_turn_kind`` значением из аргумента теста."""

    def _set(kind: str) -> None:
        monkeypatch.setattr(nodes_module, "_turn_kind", lambda: kind)

    return _set


async def test_ход_с_фразой_клиента_дописывает_её(ctx_store, quiet, turn_kind):
    """Ход с фразой клиента дописывает её в историю."""
    turn_kind("client")
    await ctx_store.save(
        "local",
        ConversationContext(
            transcript=[
                TranscriptEntry(entry_id="agent:1:0", role=ROLE_AGENT, text="Здравствуйте"),
            ],
        ),
    )
    state = _base_state(messages=[HumanMessage(content="алло")])
    out = await nodes_module.ingest_node(state, _runtime())  # type: ignore[arg-type]
    assert [text_of(m) for m in out["messages"]] == ["Здравствуйте", "алло"]
    assert isinstance(out["messages"][-1], HumanMessage)
    loaded = await ctx_store.load("local")
    assert loaded is not None
    assert [item.role for item in loaded.transcript] == [ROLE_AGENT, ROLE_CLIENT]
    assert loaded.transcript[-1].text == "алло"


async def test_ход_вытаскивания_историю_не_трогает(ctx_store, quiet, turn_kind):
    """Ход вытаскивания историю не трогает."""
    turn_kind("pull")
    prior = [
        TranscriptEntry(entry_id="agent:1:0", role=ROLE_AGENT, text="Здравствуйте"),
        TranscriptEntry(entry_id="client:2:1", role=ROLE_CLIENT, text="алло"),
    ]
    await ctx_store.save("local", ConversationContext(transcript=list(prior)))
    state = _base_state(messages=[HumanMessage(content="шум")])
    out = await nodes_module.ingest_node(state, _runtime(turn_kind="pull"))  # type: ignore[arg-type]
    assert [text_of(m) for m in out["messages"]] == ["Здравствуйте", "алло"]
    loaded = await ctx_store.load("local")
    assert loaded is not None
    assert [item.text for item in loaded.transcript] == ["Здравствуйте", "алло"]


async def test_ход_продолжения_историю_не_трогает(ctx_store, quiet, turn_kind):
    """Ход продолжения историю не трогает."""
    turn_kind("continuation")
    prior = [
        TranscriptEntry(entry_id="agent:1:0", role=ROLE_AGENT, text="Подождите"),
    ]
    await ctx_store.save("local", ConversationContext(transcript=list(prior)))
    state = _base_state(messages=[])
    out = await nodes_module.ingest_node(state, _runtime(turn_kind="continuation"))  # type: ignore[arg-type]
    assert [text_of(m) for m in out["messages"]] == ["Подождите"]
    loaded = await ctx_store.load("local")
    assert loaded is not None
    assert len(loaded.transcript) == 1


async def test_ход_молчания_историю_не_трогает(ctx_store, quiet, turn_kind):
    """Ход молчания историю не трогает."""
    turn_kind("silence")
    prior = [
        TranscriptEntry(entry_id="agent:1:0", role=ROLE_AGENT, text="Вы на связи?"),
    ]
    await ctx_store.save("local", ConversationContext(transcript=list(prior)))
    state = _base_state(messages=[])
    out = await nodes_module.ingest_node(state, _runtime(turn_kind="silence"))  # type: ignore[arg-type]
    assert [text_of(m) for m in out["messages"]] == ["Вы на связи?"]
    loaded = await ctx_store.load("local")
    assert loaded is not None
    assert len(loaded.transcript) == 1


async def test_история_из_кеша_целиком_даже_если_снимок_пустой(ctx_store, quiet, turn_kind):
    """История из кеша уходит в messages целиком, даже если снимок бота пустой."""
    turn_kind("client")
    await ctx_store.save(
        "local",
        ConversationContext(
            transcript=[
                TranscriptEntry(entry_id="agent:1:0", role=ROLE_AGENT, text="Здравствуйте"),
                TranscriptEntry(entry_id="client:2:1", role=ROLE_CLIENT, text="алло"),
                TranscriptEntry(entry_id="agent:2:2", role=ROLE_AGENT, text="Из какого вы города?"),
            ],
        ),
    )
    state = _base_state(messages=[])
    out = await nodes_module.ingest_node(state, _runtime())  # type: ignore[arg-type]
    assert [text_of(m) for m in out["messages"]] == [
        "Здравствуйте",
        "алло",
        "Из какого вы города?",
    ]
    assert isinstance(out["messages"][0], AIMessage)
    assert isinstance(out["messages"][1], HumanMessage)
    assert isinstance(out["messages"][2], AIMessage)
