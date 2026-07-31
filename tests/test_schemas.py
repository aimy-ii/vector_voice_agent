"""Тесты схемы ответа генератора."""

from __future__ import annotations

from graph.schemas import TurnResult


def test_turn_result_reply_обязателен_завершение_по_умолчанию_ложь():
    """Схема принимает ответ с одним полем reply; завершение необязательно и False."""
    fields = list(TurnResult.model_fields)
    assert fields == ["reply", "conversation_ended"]
    assert fields[-1] == "conversation_ended"

    parsed = TurnResult.model_validate({"reply": "Здравствуйте."})
    assert parsed.reply == "Здравствуйте."
    assert parsed.conversation_ended is False

    with_extra = TurnResult.model_validate(
        {
            "reply": "Здравствуйте.",
            "aside_id": "think",
            "resume_step": False,
            "understood": ["name"],
        }
    )
    assert with_extra.reply == "Здравствуйте."
    assert with_extra.conversation_ended is False


def test_turn_result_принимает_признак_завершения():
    """Явный True в conversation_ended сохраняется."""
    parsed = TurnResult.model_validate(
        {"reply": "До свидания, хорошего дня.", "conversation_ended": True}
    )
    assert parsed.conversation_ended is True
