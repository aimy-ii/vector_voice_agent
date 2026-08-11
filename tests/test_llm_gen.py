"""Тесты склейки потока и логирования фолбэка LLM."""

from __future__ import annotations

import logging
from typing import Any
from unittest.mock import MagicMock

import pytest

from utils.llm_gen import LLMTurnFailed, _empty_reply_note, astream_structured


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


def test_справка_о_пустом_ответе_различает_случаи():
    """Справка различает отсутствие объекта, пустой объект и пустое поле."""
    assert _empty_reply_note(None, "reply") == "объекта нет"
    assert _empty_reply_note({}, "reply") == "объект пустой"
    note = _empty_reply_note({"reply": ""}, "reply")
    assert "ключи [reply]" in note
    assert "'reply'" in note
    assert "str" in note
    assert "0 симв." in note
    note_none = _empty_reply_note({"meta": 1, "reply": None}, "reply")
    assert "ключи [meta, reply]" in note_none
    assert "NoneType" in note_none


@pytest.mark.asyncio
async def test_пустая_реплика_пишет_подробность_в_лог(caplog: pytest.LogCaptureFixture):
    """Пустой reply: WARNING с подробностью на каждую попытку."""

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

    detail = [
        rec
        for rec in caplog.records
        if "Пустая реплика от модели" in rec.message and "ключи [reply]" in rec.message
    ]
    assert len(detail) == 2


@pytest.mark.asyncio
async def test_непустая_реплика_ничего_не_пишет(caplog: pytest.LogCaptureFixture):
    """Непустой reply не пишет предупреждений про пустую реплику."""

    class _Structured:
        async def astream(self, _messages: list[Any]):
            yield {"reply": "текст"}

    llm = MagicMock()
    llm.with_structured_output.return_value = _Structured()
    llm.model_name = "test-model"

    with caplog.at_level(logging.WARNING, logger="utils.llm_gen"):
        result = await astream_structured(
            llm,
            [],
            schema={"name": "t", "schema": {}, "strict": False},
            text_field="reply",
            budget=5.0,
        )

    assert result["reply"] == "текст"
    assert not any("Пустая реплика" in rec.message for rec in caplog.records)
