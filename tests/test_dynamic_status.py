"""Тесты: генератор читает статус динамики."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any

import pytest
from langchain_core.messages import HumanMessage

from graph import nodes as nodes_module
from graph.context import DYN_MISSING, DYN_NONE, DYN_READY, DYN_SEARCHING
from graph.prompts import build_turn_messages, dynamic_status_block
from graph.situations import load_situations
from graph.state import new_state_defaults
from script.store import MemoryScriptStore


@pytest.fixture()
def spoken(monkeypatch) -> list[str]:
    chunks: list[str] = []
    monkeypatch.setattr(nodes_module, "say", chunks.append)
    monkeypatch.setattr(nodes_module, "stage", lambda *a, **k: None)
    return chunks


@pytest.fixture()
def use_v2(monkeypatch) -> None:
    monkeypatch.setattr(nodes_module.settings, "script_version", None)


@pytest.fixture()
def store(monkeypatch) -> MemoryScriptStore:
    mem = MemoryScriptStore()
    monkeypatch.setattr(nodes_module, "script_store", mem)
    return mem


@pytest.fixture()
def model(monkeypatch):
    @asynccontextmanager
    async def _fake_llm(**kwargs: Any):
        yield None

    monkeypatch.setattr(nodes_module, "get_llm", _fake_llm)
    holder: dict[str, Any] = {"result": {"reply": "Хорошо."}, "calls": 0, "messages": None}

    async def _fake_stream(
        llm, messages, *, schema, text_field=None, on_delta=None, budget=None, purpose=None
    ):
        holder["calls"] += 1
        holder["messages"] = messages
        result = holder["result"]
        if text_field and on_delta is not None and result.get(text_field):
            on_delta(result[text_field])
        return result

    monkeypatch.setattr(nodes_module, "astream_structured", _fake_stream)
    return holder


def test_промпт_не_нашлось_инструкция(script):
    block = dynamic_status_block(status=DYN_MISSING)
    assert "ничего нет" in block.lower()
    messages = build_turn_messages(
        script=script,
        steps=[],
        profile={},
        facts={},
        history=[],
        asides_done=[],
        context_text="Город: Пермь",
        dynamic_status=DYN_MISSING,
    )
    assert "ничего нет" in messages[0].content.lower()


def test_промпт_повторный_поиск_без_контекста(script):
    messages = build_turn_messages(
        script=script,
        steps=[],
        profile={},
        facts={},
        history=[],
        asides_done=[],
        context_text="Секретный факт про медкомиссию",
        dynamic_status=DYN_SEARCHING,
        searching_retry=True,
    )
    content = messages[0].content
    assert "медкомиссию" not in content
    assert "пауза-заглушка уже звучала" in content.lower()


def test_промпт_готово_и_не_требуется_как_обычно(script):
    for status in (DYN_NONE, DYN_READY, ""):
        messages = build_turn_messages(
            script=script,
            steps=[],
            profile={},
            facts={},
            history=[],
            asides_done=[],
            context_text="Город: Пермь",
            dynamic_status=status,
        )
        assert "Город: Пермь" in messages[0].content
        assert "пауза-заглушка уже звучала" not in messages[0].content.lower()


async def test_генератор_в_поиске_отдаёт_фразу_и_ставит_флаг(
    spoken, store, model, use_v2, monkeypatch
):
    model["result"] = {"understood": [], "reply": "Продолжаем."}
    catalog = load_situations()

    async def _keep_searching(context, **kwargs: Any):
        return context

    monkeypatch.setattr(nodes_module, "run_contexter", _keep_searching)
    state: dict[str, Any] = {
        **new_state_defaults(),
        "messages": [HumanMessage(content="а медкомиссия?")],
        "script_id": "vector_ru",
        "script_version": "2",
        "current_step": "name",
        "head_steps": ["name"],
        "conversation_context": {
            "static_text": "Город: Пермь",
            "dynamic_status": DYN_SEARCHING,
            "situation_slug": "default",
            "filler_spoken": False,
        },
    }
    out = await nodes_module.respond_node(state, None)  # type: ignore[arg-type]
    filler = out.get("spoken_filler") or ""
    assert filler
    assert filler in catalog["default"]
    assert out["conversation_context"]["filler_spoken"] is True
    assert any(filler in chunk for chunk in spoken)
    assert model["calls"] == 1


async def test_генератор_повторный_в_поиске_фразу_не_повторяет(
    spoken, store, model, use_v2, monkeypatch
):
    model["result"] = {"understood": [], "reply": "Давайте дальше."}
    calls: list[str] = []

    def _no_pick(*args: Any, **kwargs: Any) -> str:
        calls.append("pick")
        return "НЕ ДОЛЖНА ЗВУЧАТЬ"

    async def _keep_searching(context, **kwargs: Any):
        return context

    monkeypatch.setattr(nodes_module, "pick_filler", _no_pick)
    monkeypatch.setattr(nodes_module, "run_contexter", _keep_searching)
    state: dict[str, Any] = {
        **new_state_defaults(),
        "messages": [HumanMessage(content="ну что там?")],
        "script_id": "vector_ru",
        "script_version": "2",
        "current_step": "name",
        "head_steps": ["name"],
        "fillers_used": ["так… секунду"],
        "conversation_context": {
            "static_text": "Город: Пермь\nСекрет",
            "dynamic_status": DYN_SEARCHING,
            "situation_slug": "default",
            "filler_spoken": True,
        },
    }
    out = await nodes_module.respond_node(state, None)  # type: ignore[arg-type]
    assert calls == []
    assert "НЕ ДОЛЖНА ЗВУЧАТЬ" not in "".join(spoken)
    assert out["conversation_context"]["filler_spoken"] is True
    prompt = model["messages"][0].content
    assert "Секрет" not in prompt
    assert "пауза-заглушка уже звучала" in prompt.lower()


async def test_контекстер_до_respond_кладёт_справку_в_динамику(
    spoken, store, model, use_v2, script
):
    """Контекстер вызывается до генерации; справка уже в промпте."""
    model["result"] = {"understood": [], "aside_id": None, "reply": "Медкомиссия отдельно."}
    med = script.helps["medcheck"].text
    state: dict[str, Any] = {
        **new_state_defaults(),
        "messages": [HumanMessage(content="а когда медкомиссию проходить?")],
        "script_id": "vector_ru",
        "script_version": "2",
        "current_step": "name",
        "head_steps": ["name"],
        "conversation_context": {"static_text": "Город: Пермь"},
    }
    out = await nodes_module.respond_node(state, None)  # type: ignore[arg-type]
    ctx = out["conversation_context"]
    assert ctx["dynamic_status"] == DYN_READY
    assert med in ctx["dynamic_text"]
    prompt = model["messages"][0].content
    assert med in prompt
    assert model["calls"] == 1


async def test_контекстер_на_lookup_кладёт_справку_в_динамику(
    spoken, store, use_v2, script, monkeypatch
):
    """На маршруте lookup контекстер тоже наполняет динамику до respond."""
    from tests.conftest import FakeKB

    med = script.helps["medcheck"].text
    fake = FakeKB(cities=[], city=None, branches=[], branch=None)
    monkeypatch.setattr(nodes_module, "vector_kb", fake)
    state: dict[str, Any] = {
        **new_state_defaults(),
        "messages": [HumanMessage(content="а когда медкомиссию проходить?")],
        "script_id": "vector_ru",
        "script_version": "2",
        "current_step": "price",
        "head_steps": ["price"],
        "city_slug": "perm",
        "city_name": "Пермь",
        "branch_slug": "perm_chernyshevskogo",
        "profile": {"city": "Пермь", "branch": "perm_chernyshevskogo"},
        "conversation_context": {
            "static_text": "Город: Пермь\nСтоимость обучения — от 43900 рублей.",
            "city_slug": "perm",
            "city_name": "Пермь",
            "branch_slug": "perm_chernyshevskogo",
            "frozen": True,
        },
    }
    out = await nodes_module.lookup_node(state, None)  # type: ignore[arg-type]
    ctx = out["conversation_context"]
    assert ctx["dynamic_status"] == DYN_READY
    assert med in ctx["dynamic_text"]
