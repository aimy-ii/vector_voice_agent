"""Тесты схемы ответа генератора."""

from __future__ import annotations

from graph.schemas import TurnResult


def test_turn_result_только_reply():
    """В TurnResult одно поле reply; лишние поля ответа не роняют разбор."""
    fields = list(TurnResult.model_fields)
    assert fields == ["reply"]
    assert "understood" not in fields
    assert "aside_id" not in fields
    assert "resume_step" not in fields

    parsed = TurnResult.model_validate(
        {
            "reply": "Здравствуйте.",
            "aside_id": "think",
            "resume_step": False,
            "understood": ["name"],
        }
    )
    assert parsed.reply == "Здравствуйте."
    assert parsed.model_dump() == {"reply": "Здравствуйте."}
