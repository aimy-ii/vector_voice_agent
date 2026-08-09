"""Офлайн-тесты истории звонка без сшивания."""

from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage

from graph.transcript import (
    ROLE_AGENT,
    ROLE_CLIENT,
    TranscriptEntry,
    append_agent,
    append_client,
    to_messages,
)


def test_три_реплики_бота_подряд_без_клиента():
    """Три реплики бота подряд без хода клиента — все три в истории по порядку."""
    entries: list[TranscriptEntry] = []
    for turn, text in enumerate(("Раз.", "Два.", "Три."), start=1):
        entries = append_agent(entries, turn=turn, text=text)
    assert [item.text for item in entries] == ["Раз.", "Два.", "Три."]
    assert all(item.role == ROLE_AGENT for item in entries)


def test_две_одинаковые_фразы_клиента_две_записи():
    """Две одинаковые фразы клиента на разных ходах — две записи, не одна."""
    entries = append_client([], turn=1, text="Да.")
    entries = append_client(entries, turn=2, text="Да.")
    assert len(entries) == 2
    assert all(item.role == ROLE_CLIENT and item.text == "Да." for item in entries)


def test_пустой_текст_ничего_не_добавляет():
    """Пустой или пробельный текст историю не меняет."""
    entries = append_agent([], turn=1, text="Есть.")
    assert append_agent(entries, turn=2, text="") == entries
    assert append_client(entries, turn=2, text="   ") == entries


def test_to_messages_порядок_и_типы():
    """to_messages отдаёт AIMessage и HumanMessage в порядке разговора."""
    entries = [
        TranscriptEntry(entry_id="a:1:0", role=ROLE_AGENT, text="Привет"),
        TranscriptEntry(entry_id="c:1:1", role=ROLE_CLIENT, text="Алло"),
    ]
    messages = to_messages(entries)
    assert isinstance(messages[0], AIMessage)
    assert isinstance(messages[1], HumanMessage)
    assert messages[0].content == "Привет"
    assert messages[1].content == "Алло"


def test_идентификаторы_записей_уникальны():
    """Идентификаторы записей уникальны в пределах истории."""
    entries = append_agent([], turn=1, text="Раз.")
    entries = append_client(entries, turn=1, text="Да.")
    entries = append_agent(entries, turn=2, text="Два.")
    ids = [item.entry_id for item in entries]
    assert len(ids) == len(set(ids))
