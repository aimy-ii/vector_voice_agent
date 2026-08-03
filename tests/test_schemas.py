"""Тесты схемы ответа генератора."""

from __future__ import annotations

from graph.schemas import TurnResult


def test_turn_result_только_reply():
    """Схема принимает ответ с одним полем reply; признака завершения нет."""
    fields = list(TurnResult.model_fields)
    assert fields == ["reply"]
    assert "conversation_ended" not in TurnResult.model_fields

    parsed = TurnResult.model_validate({"reply": "Здравствуйте."})
    assert parsed.reply == "Здравствуйте."

    with_extra = TurnResult.model_validate(
        {
            "reply": "Здравствуйте.",
            "aside_id": "think",
            "resume_step": False,
            "understood": ["name"],
            "conversation_ended": True,
        }
    )
    assert with_extra.reply == "Здравствуйте."
    assert "conversation_ended" not in with_extra.model_dump()
