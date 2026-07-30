"""Тесты схемы ответа генератора."""

from __future__ import annotations

from graph.schemas import TurnResult


def test_turn_result_reply_первым_без_understood():
    """В TurnResult нет understood, первым полем идёт reply."""
    fields = list(TurnResult.model_fields)
    assert fields[0] == "reply"
    assert "understood" not in fields
    assert fields == ["reply", "aside_id", "resume_step"]
