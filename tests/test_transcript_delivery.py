"""Офлайн-тесты сверки последней реплики бота с фактом доставки."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from graph import nodes as nodes_module
from graph.context import ConversationContext
from graph.history import text_of
from graph.state import new_state_defaults
from graph.transcript import (
    ROLE_AGENT,
    ROLE_CLIENT,
    TranscriptEntry,
    append_agent,
    reconcile_last_agent,
    to_messages,
)


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


def test_реплика_прозвучала_целиком_история_не_меняется():
    """Снимок совпал с последней записью — история без изменений."""
    entries = [
        TranscriptEntry(entry_id="c:1:0", role=ROLE_CLIENT, text="Алло"),
        TranscriptEntry(entry_id="a:1:1", role=ROLE_AGENT, text="Здравствуйте"),
    ]
    out = reconcile_last_agent(entries, spoken="Здравствуйте")
    assert out == entries
    assert out is not entries


def test_реплика_не_прозвучала_последняя_запись_бота_удалена():
    """Пустой снимок — последняя запись бота убрана, более ранние на месте."""
    earlier = TranscriptEntry(entry_id="c:1:0", role=ROLE_CLIENT, text="Алло")
    entries = [
        earlier,
        TranscriptEntry(entry_id="a:1:1", role=ROLE_AGENT, text="Здравствуйте"),
    ]
    out = reconcile_last_agent(entries, spoken="")
    assert out == [earlier]
    assert out[0].entry_id == "c:1:0"


def test_снимок_с_текстом_предыдущей_реплики_удаляет_последнюю():
    """Ход обогнал предыдущий: в снимке старый текст — последняя запись снята."""
    first = TranscriptEntry(entry_id="a:1:0", role=ROLE_AGENT, text="Первая")
    second = TranscriptEntry(entry_id="a:2:1", role=ROLE_AGENT, text="Вторая")
    out = reconcile_last_agent([first, second], spoken="Первая")
    assert out == [first]
    assert out[0].text == "Первая"


def test_частичная_доставка_урезает_до_префикса():
    """Прозвучал префикс — текст урезан, роль и entry_id сохранены."""
    entry = TranscriptEntry(
        entry_id="a:3:2",
        role=ROLE_AGENT,
        text="Добрый день, меня зовут Анна",
    )
    out = reconcile_last_agent([entry], spoken="Добрый день")
    assert len(out) == 1
    assert out[0].text == "Добрый день"
    assert out[0].role == ROLE_AGENT
    assert out[0].entry_id == "a:3:2"


def test_последняя_запись_клиента_не_меняется():
    """Последняя запись — клиент: история возвращается без правок."""
    entries = [
        TranscriptEntry(entry_id="a:1:0", role=ROLE_AGENT, text="Привет"),
        TranscriptEntry(entry_id="c:1:1", role=ROLE_CLIENT, text="Да"),
    ]
    out = reconcile_last_agent(entries, spoken="Привет")
    assert out == entries


def test_пустая_история_остаётся_пустой():
    """Пустой список — пустой результат."""
    assert reconcile_last_agent([], spoken="что угодно") == []


def test_две_реплики_бота_снимается_только_последняя():
    """Две записи бота подряд: снимок с первой — удаляется только последняя."""
    entries = append_agent([], turn=1, text="Первая")
    entries = append_agent(entries, turn=2, text="Вторая")
    out = reconcile_last_agent(entries, spoken="Первая")
    assert len(out) == 1
    assert out[0].text == "Первая"
    assert out[0].role == ROLE_AGENT
    messages = to_messages(out)
    assert len(messages) == 1
    assert isinstance(messages[0], AIMessage)
    assert messages[0].content == "Первая"


async def test_снимок_без_реплик_бота_историю_не_трогает(ctx_store, quiet, turn_kind):
    """Снимок только с клиентом — последняя реплика бота в истории остаётся."""
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


async def test_пустой_снимок_историю_не_трогает(ctx_store, quiet, turn_kind):
    """Пустой снимок — история из кеша без изменений."""
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


async def test_снимок_с_репликой_бота_сверку_включает(ctx_store, quiet, turn_kind):
    """В снимке старая реплика бота — более новая последняя запись снимается."""
    turn_kind("continuation")
    await ctx_store.save(
        "local",
        ConversationContext(
            transcript=[
                TranscriptEntry(entry_id="agent:1:0", role=ROLE_AGENT, text="Первая"),
                TranscriptEntry(entry_id="agent:2:1", role=ROLE_AGENT, text="Вторая"),
            ],
        ),
    )
    state = _base_state(messages=[AIMessage(content="Первая")])
    out = await nodes_module.ingest_node(state, _runtime(turn_kind="continuation"))  # type: ignore[arg-type]
    assert [text_of(m) for m in out["messages"]] == ["Первая"]
