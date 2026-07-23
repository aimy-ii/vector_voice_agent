"""Тесты сверки намеченного с произнесённым.

Шаг закрывается по произнесённому, а не по сгенерированному: правду о
перебивании знает только бот, и она приезжает в истории следующего хода.
"""

from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage

from graph.reconcile import (
    SPOKEN_ENOUGH,
    count_agent_messages,
    reopen_if_interrupted,
    spoken_ratio,
    was_delivered,
)


def test_счёт_реплик_бота():
    messages = [HumanMessage(content="а"), AIMessage(content="б"), HumanMessage(content="в")]
    assert count_agent_messages(messages) == 1
    assert count_agent_messages([]) == 0


def test_доля_произнесённого():
    assert spoken_ratio(100, 100) == 1.0
    assert spoken_ratio(100, 50) == 0.5
    assert spoken_ratio(0, 0) == 1.0
    assert spoken_ratio(100, 500) == 1.0


def test_реплику_дослушали():
    assert was_delivered(planned_len=100, spoken_len=95, ai_count_before=0, ai_count_now=1) is True


def test_перебили_на_середине():
    assert was_delivered(planned_len=100, spoken_len=20, ai_count_before=0, ai_count_now=1) is False


def test_перебили_до_первого_звука():
    """Записи в истории не появится вовсе — счётчик реплик не сдвинулся."""
    assert was_delivered(planned_len=100, spoken_len=99, ai_count_before=1, ai_count_now=1) is False


def test_порог_на_границе():
    length = 100
    ниже = int(length * SPOKEN_ENOUGH) - 1
    выше = int(length * SPOKEN_ENOUGH) + 1
    assert (
        was_delivered(planned_len=length, spoken_len=ниже, ai_count_before=0, ai_count_now=1)
        is False
    )
    assert (
        was_delivered(planned_len=length, spoken_len=выше, ai_count_before=0, ai_count_now=1)
        is True
    )


def test_перебитый_шаг_возвращается_в_работу():
    state = {
        "pending_step": "presentation",
        "pending_len": 200,
        "pending_ai_count": 0,
        "step_status": {"presentation": "done", "city": "done"},
    }
    messages = [AIMessage(content="Расскажу, как проходит")]
    patch = reopen_if_interrupted(
        state=state, messages=messages, last_spoken="Расскажу, как проходит"
    )

    assert patch["step_status"]["presentation"] == "open"
    assert patch["step_status"]["city"] == "done"
    assert patch["pending_step"] is None


def test_дослушанный_шаг_остаётся_закрытым():
    text = "Расскажу, как проходит обучение у нас в академии подробно и по порядку"
    state = {
        "pending_step": "presentation",
        "pending_len": len(text),
        "pending_ai_count": 0,
        "step_status": {"presentation": "done"},
    }
    patch = reopen_if_interrupted(state=state, messages=[AIMessage(content=text)], last_spoken=text)
    assert "step_status" not in patch
    assert patch["pending_step"] is None


def test_сверять_нечего():
    assert reopen_if_interrupted(state={}, messages=[], last_spoken="") == {}


def test_сверка_на_истории_из_словарей():
    """История с сервера — словари; после редьюсера reconcile видит AIMessage."""
    from graph.state import replace_messages

    messages = replace_messages(
        [],
        [{"role": "ai", "content": "Расскажу, как проходит"}],
    )
    state = {
        "pending_step": "presentation",
        "pending_len": 200,
        "pending_ai_count": 0,
        "step_status": {"presentation": "done", "city": "done"},
    }
    patch = reopen_if_interrupted(
        state=state,
        messages=messages,
        last_spoken="Расскажу, как проходит",
    )
    assert patch["step_status"]["presentation"] == "open"
    assert count_agent_messages(messages) == 1
