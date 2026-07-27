"""Тесты склейки потока и логирования фолбэка LLM."""

from __future__ import annotations

import logging
from typing import Any
from unittest.mock import MagicMock

import pytest

from utils.llm_gen import LLMTurnFailed, astream_structured


@pytest.mark.asyncio
async def test_токены_слова_склеиваются_встык():
    """«внут»+«ренние»+« экзамены» → «внутренние экзамены» без пробела внутри слова."""
    deltas: list[str] = []

    class _Structured:
        async def astream(self, _messages: list[Any]):
            yield {"reply": "внут"}
            yield {"reply": "внутренние"}
            yield {"reply": "внутренние экзамены"}

    llm = MagicMock()
    llm.with_structured_output.return_value = _Structured()
    llm.model_name = "test-model"

    result = await astream_structured(
        llm,
        [],
        schema={"name": "t", "schema": {}, "strict": False},
        text_field="reply",
        on_delta=deltas.append,
        budget=5.0,
    )

    assert deltas == ["внут", "ренние", " экзамены"]
    assert "".join(deltas) == "внутренние экзамены"
    assert result["reply"] == "внутренние экзамены"


@pytest.mark.asyncio
async def test_односимвольные_токены_не_разбиваются_пробелами():
    """Поток «П»+«р»+…+«т» собирается в «Привет», а не «П р и в е т»."""
    deltas: list[str] = []
    word = "Привет"
    prefixes = [word[: i + 1] for i in range(len(word))]

    class _Structured:
        async def astream(self, _messages: list[Any]):
            for prefix in prefixes:
                yield {"reply": prefix}

    llm = MagicMock()
    llm.with_structured_output.return_value = _Structured()
    llm.model_name = "test-model"

    result = await astream_structured(
        llm,
        [],
        schema={"name": "t", "schema": {}, "strict": False},
        text_field="reply",
        on_delta=deltas.append,
        budget=5.0,
    )

    assert deltas == list(word)
    assert "".join(deltas) == "Привет"
    assert result["reply"] == "Привет"


@pytest.mark.asyncio
async def test_фолбэк_логирует_пустой_ответ(caplog: pytest.LogCaptureFixture):
    """При пустом reply в лог уходит причина подстановки фолбэка."""

    class _Structured:
        async def astream(self, _messages: list[Any]):
            yield {"reply": ""}

    llm = MagicMock()
    llm.with_structured_output.return_value = _Structured()
    llm.model_name = "test-model"

    with caplog.at_level(logging.WARNING, logger="utils.llm_gen"):
        with pytest.raises(LLMTurnFailed, match="Пустой ответ модели"):
            await astream_structured(
                llm,
                [],
                schema={"name": "t", "schema": {}, "strict": False},
                text_field="reply",
                budget=5.0,
            )

    assert any("Подстановка фолбэка" in rec.message for rec in caplog.records)
    assert any("Пустой ответ модели" in rec.message for rec in caplog.records)
