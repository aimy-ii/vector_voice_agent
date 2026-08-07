"""Офлайн-тесты единой истории звонка."""

from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage

from graph.transcript import (
    ROLE_AGENT,
    ROLE_CLIENT,
    TranscriptEntry,
    append_agent,
    count_spoken_agent,
    merge_snapshot,
    to_messages,
)


def test_три_реплики_бота_без_снимка_все_в_истории():
    """Три подряд без подтверждения — все на месте, spoken ложь."""
    entries: list[TranscriptEntry] = []
    for turn, text in enumerate(("Раз.", "Два.", "Три."), start=1):
        entries = append_agent(entries, turn=turn, text=text)
    assert [item.text for item in entries] == ["Раз.", "Два.", "Три."]
    assert all(item.role == ROLE_AGENT for item in entries)
    assert all(item.spoken is False for item in entries)


def test_снимок_подтверждает_первую_из_трёх():
    """Снимок с первой репликой — она spoken, остальные на месте по порядку."""
    entries = append_agent([], turn=1, text="Раз.")
    entries = append_agent(entries, turn=2, text="Два.")
    entries = append_agent(entries, turn=3, text="Три.")
    merged = merge_snapshot(entries, [AIMessage(content="Раз.")], turn=4)
    assert [item.text for item in merged] == ["Раз.", "Два.", "Три."]
    assert merged[0].spoken is True
    assert merged[1].spoken is False
    assert merged[2].spoken is False


def test_две_одинаковые_реплики_человека_не_схлопываются():
    """«Да.» дважды подряд — две записи, не одна."""
    snapshot = [HumanMessage(content="Да."), HumanMessage(content="Да.")]
    merged = merge_snapshot([], snapshot, turn=1)
    assert len(merged) == 2
    assert all(item.role == ROLE_CLIENT and item.text == "Да." for item in merged)
    assert all(item.spoken is True for item in merged)


def test_перебивание_заменяет_текст_на_короткий():
    """Полная реплика в накопленном, начало в снимке — одна запись, короткий текст."""
    entries = append_agent([], turn=1, text="Расскажу про обучение подробно и по порядку")
    merged = merge_snapshot(
        entries,
        [AIMessage(content="Расскажу про обучение")],
        turn=2,
    )
    assert len(merged) == 1
    assert merged[0].text == "Расскажу про обучение"
    assert merged[0].spoken is True


def test_повторный_снимок_не_удлиняет_историю():
    """Та же история снимка второй раз — длина не растёт."""
    entries = append_agent([], turn=1, text="Здравствуйте")
    snapshot = [AIMessage(content="Здравствуйте"), HumanMessage(content="алло")]
    once = merge_snapshot(entries, snapshot, turn=2)
    twice = merge_snapshot(once, snapshot, turn=3)
    assert len(twice) == len(once) == 2
    assert [item.text for item in twice] == ["Здравствуйте", "алло"]


def test_новая_реплика_человека_дописывается():
    """Реплики человека не было в накопленном — появляется из снимка."""
    entries = append_agent([], turn=1, text="Как вас зовут?")
    merged = merge_snapshot(
        entries,
        [
            AIMessage(content="Как вас зовут?"),
            HumanMessage(content="Павел"),
        ],
        turn=2,
    )
    assert [item.role for item in merged] == [ROLE_AGENT, ROLE_CLIENT]
    assert merged[1].text == "Павел"
    assert merged[1].spoken is True


def test_count_spoken_agent_только_подтверждённые():
    """Счётчик берёт только agent с spoken=True."""
    entries = [
        TranscriptEntry(entry_id="a:1:0", role=ROLE_AGENT, text="Раз", spoken=True),
        TranscriptEntry(entry_id="c:1:1", role=ROLE_CLIENT, text="Да", spoken=True),
        TranscriptEntry(entry_id="a:2:2", role=ROLE_AGENT, text="Два", spoken=False),
    ]
    assert count_spoken_agent(entries) == 1


def test_to_messages_порядок_и_типы():
    """to_messages отдаёт AIMessage и HumanMessage в порядке разговора."""
    entries = [
        TranscriptEntry(entry_id="a:1:0", role=ROLE_AGENT, text="Привет", spoken=True),
        TranscriptEntry(entry_id="c:1:1", role=ROLE_CLIENT, text="Алло", spoken=True),
    ]
    messages = to_messages(entries)
    assert isinstance(messages[0], AIMessage)
    assert isinstance(messages[1], HumanMessage)
    assert messages[0].content == "Привет"
    assert messages[1].content == "Алло"
