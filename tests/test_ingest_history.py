"""Восстановление last_agent_reply в хвосте истории на узле ingest."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from graph import nodes as nodes_module
from graph.context import ConversationContext
from graph.history import text_of
from graph.state import new_state_defaults


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


async def test_нет_в_истории_дописывает_в_конец(ctx_store, quiet):
    """Последней реплики бота нет в истории — после ingest она в конце."""
    reply = "Из какого вы города?"
    await ctx_store.save("local", ConversationContext(last_agent_reply=reply))
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


async def test_уже_есть_в_хвосте_без_дубля(ctx_store, quiet):
    """Реплика уже в хвосте — история не меняется, дубля нет."""
    reply = "Из какого вы города?"
    await ctx_store.save("local", ConversationContext(last_agent_reply=reply))
    messages = [HumanMessage(content="алло"), AIMessage(content=reply)]
    state = _base_state(messages=list(messages))
    out = await nodes_module.ingest_node(state, _runtime())  # type: ignore[arg-type]
    assert len(out["messages"]) == len(messages)
    assert _agent_texts(out["messages"]) == [reply]


async def test_отличается_пробелами_и_знаками_без_дубля(ctx_store, quiet):
    """В хвосте та же реплика с другими пробелами/знаками — дубля нет."""
    await ctx_store.save(
        "local",
        ConversationContext(last_agent_reply="Из какого вы города?"),
    )
    messages = [
        HumanMessage(content="алло"),
        AIMessage(content="Из  какого вы города !"),
    ]
    state = _base_state(messages=list(messages))
    out = await nodes_module.ingest_node(state, _runtime())  # type: ignore[arg-type]
    assert len(out["messages"]) == len(messages)
    assert _agent_texts(out["messages"]) == ["Из  какого вы города !"]


async def test_пустая_last_agent_reply_история_не_меняется(ctx_store, quiet):
    """Первый ход: last_agent_reply пуста — историю не трогаем."""
    await ctx_store.save("local", ConversationContext(last_agent_reply=""))
    messages = [HumanMessage(content="алло")]
    state = _base_state(messages=list(messages))
    out = await nodes_module.ingest_node(state, _runtime())  # type: ignore[arg-type]
    assert len(out["messages"]) == 1
    assert isinstance(out["messages"][0], HumanMessage)
    assert _agent_texts(out["messages"]) == []


async def test_встаёт_перед_репликой_клиента_этого_хода(ctx_store, quiet):
    """Есть human в хвосте — реплика бота встаёт перед ним."""
    reply = "Как к вам обращаться?"
    await ctx_store.save("local", ConversationContext(last_agent_reply=reply))
    state = _base_state(
        messages=[
            HumanMessage(content="алло"),
            HumanMessage(content="Павел"),
        ]
    )
    out = await nodes_module.ingest_node(state, _runtime())  # type: ignore[arg-type]
    assert [m.type for m in out["messages"]] == ["human", "ai", "human"]
    assert text_of(out["messages"][0]) == "алло"
    assert text_of(out["messages"][1]) == reply
    assert text_of(out["messages"][2]) == "Павел"
    assert _agent_texts(out["messages"]) == [reply]
