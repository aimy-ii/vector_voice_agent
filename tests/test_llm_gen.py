"""Тесты склейки потока и логирования фолбэка LLM."""

from __future__ import annotations

import logging
from typing import Any
from unittest.mock import MagicMock

import pytest

from utils.llm_gen import (
    LLMTurnFailed,
    astream_structured,
    glue_stream_delta,
    join_stream_chunks,
)


def test_склейка_кусков_не_теряет_пробел_на_стыке():
    """На стыке «внутренние»+«экзамены» появляется пробел."""
    assert glue_stream_delta("внутренние", "экзамены") == " экзамены"
    assert (
        join_stream_chunks(["все необходимые ", "внутренние", "экзамены"])
        == "все необходимые внутренние экзамены"
    )


def test_склейка_не_дублирует_пробел():
    """Если пробел уже на границе — второй не вставляется."""
    assert glue_stream_delta("внутренние ", "экзамены") == "экзамены"
    assert glue_stream_delta("внутренние", " экзамены") == " экзамены"
    assert join_stream_chunks(["все ", "необходимые внутренние ", "экзамены"]) == (
        "все необходимые внутренние экзамены"
    )


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
