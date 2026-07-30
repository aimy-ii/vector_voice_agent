"""Тесты: генератор читает статус динамики."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any

import pytest
from langchain_core.messages import HumanMessage

from graph import nodes as nodes_module
from graph.context import DYN_MISSING, DYN_NONE, DYN_READY, DYN_SEARCHING, ConversationContext
from graph.context_store import MemoryContextStore
from graph.prompts import build_turn_messages, dynamic_status_block
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
def ctx_store(monkeypatch) -> MemoryContextStore:
    mem = MemoryContextStore()
    monkeypatch.setattr(nodes_module, "context_store", mem)
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


async def test_генератор_в_поиске_отдаёт_заглушку_с_предметом(
    spoken, store, ctx_store, model, use_v2, monkeypatch
):
    model["result"] = {"understood": [], "reply": "Продолжаем."}

    async def _spy(*args: Any, **kwargs: Any):
        raise AssertionError("контекстер не должен вызываться из respond")

    monkeypatch.setattr("graph.contexter.run_contexter", _spy)
    reply = "а медкомиссия?"
    ctx = ConversationContext(
        static_text="Город: Пермь",
        dynamic_status=DYN_SEARCHING,
        situation_slug="медкомиссия",
        filler_spoken=False,
        dynamic_reply=reply,
    )
    await ctx_store.save("local", ctx)
    state: dict[str, Any] = {
        **new_state_defaults(),
        "messages": [HumanMessage(content=reply)],
        "script_id": "vector_ru",
        "script_version": "2",
        "current_step": "name",
        "head_steps": ["name"],
        "conversation_context": ctx.model_dump(),
    }
    out = await nodes_module.respond_node(state, None)  # type: ignore[arg-type]
    filler = out.get("spoken_filler") or ""
    assert filler
    assert "медкомиссия" in filler
    assert out["conversation_context"]["filler_spoken"] is True
    assert any(filler in chunk for chunk in spoken)
    # Между заглушкой и репликой генератора — пробел в spoken.
    joined = "".join(spoken)
    assert "Продолжаем." in joined
    assert filler + " " in joined or filler.rstrip() + " " in joined
    assert model["calls"] == 1


async def test_генератор_повторный_в_поиске_фразу_не_повторяет(
    spoken, store, ctx_store, model, use_v2, monkeypatch
):
    model["result"] = {"understood": [], "reply": "Давайте дальше."}
    calls: list[str] = []

    def _no_pick(*args: Any, **kwargs: Any) -> str:
        calls.append("pick")
        return "НЕ ДОЛЖНА ЗВУЧАТЬ"

    async def _spy(*args: Any, **kwargs: Any):
        raise AssertionError("контекстер не должен вызываться из respond")

    monkeypatch.setattr(nodes_module, "pick_filler", _no_pick)
    monkeypatch.setattr("graph.contexter.run_contexter", _spy)
    reply = "ну что там?"
    ctx = ConversationContext(
        static_text="Город: Пермь\nСекрет",
        dynamic_status=DYN_SEARCHING,
        situation_slug="филиалы",
        filler_spoken=True,
        dynamic_reply=reply,
        dynamic_text="Секретный факт",
    )
    await ctx_store.save("local", ctx)
    state: dict[str, Any] = {
        **new_state_defaults(),
        "messages": [HumanMessage(content=reply)],
        "script_id": "vector_ru",
        "script_version": "2",
        "current_step": "name",
        "head_steps": ["name"],
        "fillers_used": ["так, филиалы… секундочку."],
        "conversation_context": ctx.model_dump(),
    }
    out = await nodes_module.respond_node(state, None)  # type: ignore[arg-type]
    assert calls == []
    assert "НЕ ДОЛЖНА ЗВУЧАТЬ" not in "".join(spoken)
    assert out["conversation_context"]["filler_spoken"] is True
    prompt = model["messages"][0].content
    assert "Секрет" not in prompt
    assert "пауза-заглушка уже звучала" in prompt.lower()


async def test_respond_читает_динамику_из_кеша_без_контекстера(
    spoken, store, ctx_store, model, use_v2, monkeypatch
):
    """Основной ход только читает готовую динамику из кеша."""
    model["result"] = {"understood": [], "aside_id": None, "reply": "Медкомиссия отдельно."}
    fact = "Медкомиссия понадобится к началу практики."
    reply = "а когда медкомиссию проходить?"

    async def _spy(*args: Any, **kwargs: Any):
        raise AssertionError("контекстер не должен вызываться из respond")

    monkeypatch.setattr("graph.contexter.run_contexter", _spy)
    ctx = ConversationContext(
        static_text="Город: Пермь",
        dynamic_text=fact,
        dynamic_status=DYN_READY,
        dynamic_reply=reply,
    )
    await ctx_store.save("local", ctx)

    state: dict[str, Any] = {
        **new_state_defaults(),
        "messages": [HumanMessage(content=reply)],
        "script_id": "vector_ru",
        "script_version": "2",
        "current_step": "name",
        "head_steps": ["name"],
        "conversation_context": ctx.model_dump(),
    }
    out = await nodes_module.respond_node(state, None)  # type: ignore[arg-type]
    prompt = model["messages"][0].content
    assert fact in prompt
    assert fact in out["conversation_context"]["dynamic_text"]
    assert model["calls"] == 1


async def test_lookup_не_зовёт_контекстер(spoken, store, ctx_store, use_v2, monkeypatch):
    """На маршруте lookup контекстер не вызывается."""
    from tests.conftest import FakeKB

    fake = FakeKB(cities=[], city=None, branches=[], branch=None)
    monkeypatch.setattr(nodes_module, "vector_kb", fake)

    async def _spy(*args: Any, **kwargs: Any):
        raise AssertionError("контекстер не должен вызываться из lookup")

    monkeypatch.setattr("graph.contexter.run_contexter", _spy)
    # run_contexter больше не импортируется в nodes — шпион на модуле contexter.
    if hasattr(nodes_module, "run_contexter"):
        monkeypatch.setattr(nodes_module, "run_contexter", _spy)

    closed = {
        "name": "closed",
        "city": "closed",
        "who_studies": "closed",
        "experience": "closed",
        "transmission": "closed",
        "terms": "closed",
        "theory_format": "closed",
        "included": "closed",
        "practice": "closed",
        "branch": "closed",
    }
    state: dict[str, Any] = {
        **new_state_defaults(),
        "messages": [HumanMessage(content="а когда медкомиссию проходить?")],
        "script_id": "vector_ru",
        "script_version": "2",
        "step_status": closed,
        "step_attempts": {sid: 1 for sid in closed},
        "city_slug": "perm",
        "city_name": "Пермь",
        "branch_slug": "perm_chernyshevskogo",
        "profile": {
            "city": "Пермь",
            "branch": "perm_chernyshevskogo",
            "caller_name": "Мария",
            "student_is_caller": "да",
            "experience": "впервые",
            "transmission": "механика",
            "theory_format": "очно",
        },
        "conversation_context": {
            "static_text": "Город: Пермь\nСтоимость обучения — от 43900 рублей.",
            "city_slug": "perm",
            "city_name": "Пермь",
            "branch_slug": "perm_chernyshevskogo",
            "frozen": True,
        },
    }
    out = await nodes_module.lookup_node(state, None)  # type: ignore[arg-type]
    # Lookup мог вернуть пустой патч или статику — динамики от контекстера нет.
    ctx = out.get("conversation_context") or {}
    assert (
        ctx.get("dynamic_text", "") == ""
        or "dynamic_text" not in ctx
        or not ctx.get("dynamic_status")
        or ctx.get("dynamic_status") in (DYN_NONE, None, "")
    )
