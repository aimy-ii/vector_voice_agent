"""Приём хода: история из кеша переживает отстающий снимок бота."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from graph import nodes as nodes_module
from graph.context import ConversationContext
from graph.history import text_of
from graph.state import new_state_defaults
from graph.transcript import ROLE_AGENT, TranscriptEntry


@pytest.fixture()
def ctx_store(monkeypatch):
    from graph.context_store import MemoryContextStore

    mem = MemoryContextStore()
    monkeypatch.setattr(nodes_module, "context_store", mem)
    return mem


@pytest.fixture()
def quiet(monkeypatch) -> None:
    monkeypatch.setattr(nodes_module, "stage", lambda *a, **k: None)


def _runtime() -> SimpleNamespace:
    return SimpleNamespace(context={})


def _base_state(*, messages: list[Any]) -> dict[str, Any]:
    return {
        **new_state_defaults(),
        "messages": messages,
        "script_id": "vector_ru",
        "script_version": "2",
        "turn": 2,
    }


def _agent_texts(messages: list[Any]) -> list[str]:
    return [text_of(m) for m in messages if isinstance(m, AIMessage)]


async def test_снимок_без_реплики_бота_возвращает_её_из_кеша(ctx_store, quiet):
    """В снимке нет последней реплики бота — ingest поднимает её из transcript."""
    reply = "Из какого вы города?"
    await ctx_store.save(
        "local",
        ConversationContext(
            last_agent_reply=reply,
            transcript=[
                TranscriptEntry(
                    entry_id="agent:1:0",
                    role=ROLE_AGENT,
                    text="Здравствуйте",
                    spoken=True,
                ),
                TranscriptEntry(
                    entry_id="agent:2:1",
                    role=ROLE_AGENT,
                    text=reply,
                    spoken=False,
                ),
            ],
        ),
    )
    state = _base_state(
        messages=[
            HumanMessage(content="алло"),
            AIMessage(content="Здравствуйте"),
        ]
    )
    out = await nodes_module.ingest_node(state, _runtime())  # type: ignore[arg-type]
    assert _agent_texts(out["messages"]) == ["Здравствуйте", reply]
    assert text_of(out["messages"][-1]) == reply
    assert isinstance(out["messages"][-1], AIMessage)
    loaded = await ctx_store.load("local")
    assert loaded is not None
    assert reply in [item.text for item in loaded.transcript]
    assert any(
        item.role == ROLE_AGENT and item.text == reply and item.spoken is False
        for item in loaded.transcript
    )


async def test_снимок_с_репликой_помечает_spoken(ctx_store, quiet):
    """Снимок содержит реплику бота — она остаётся и становится spoken."""
    reply = "Из какого вы города?"
    await ctx_store.save(
        "local",
        ConversationContext(
            last_agent_reply=reply,
            transcript=[
                TranscriptEntry(
                    entry_id="agent:2:0",
                    role=ROLE_AGENT,
                    text=reply,
                    spoken=False,
                ),
            ],
        ),
    )
    messages = [HumanMessage(content="алло"), AIMessage(content=reply)]
    state = _base_state(messages=list(messages))
    out = await nodes_module.ingest_node(state, _runtime())  # type: ignore[arg-type]
    assert _agent_texts(out["messages"]) == [reply]
    loaded = await ctx_store.load("local")
    assert loaded is not None
    agent_entries = [item for item in loaded.transcript if item.role == ROLE_AGENT]
    assert len(agent_entries) == 1
    assert agent_entries[0].spoken is True
    assert agent_entries[0].text == reply


async def test_пустой_кеш_история_из_снимка(ctx_store, quiet):
    """Первый ход без накопленного — сообщения берутся из снимка бота."""
    await ctx_store.save("local", ConversationContext())
    messages = [HumanMessage(content="алло")]
    state = _base_state(messages=list(messages))
    out = await nodes_module.ingest_node(state, _runtime())  # type: ignore[arg-type]
    assert len(out["messages"]) == 1
    assert isinstance(out["messages"][0], HumanMessage)
    assert _agent_texts(out["messages"]) == []
