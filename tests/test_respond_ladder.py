"""Лестница ожидания в ``respond_node``: filler → waiting → full."""

from __future__ import annotations

from contextlib import asynccontextmanager
from types import SimpleNamespace
from typing import Any

import pytest
from langchain_core.messages import HumanMessage

from graph import nodes as nodes_module
from graph.context import (
    DYN_READY,
    DYN_SEARCHING,
    DYN_WORKING,
    ConversationContext,
)
from graph.contexter import reply_hash
from graph.state import new_state_defaults
from script.store import MemoryScriptStore


@pytest.fixture()
def spoken(monkeypatch) -> Any:
    chunks: list[str] = []
    kinds: list[str] = []

    def _stage(name: str, _msg: str, _status: str = "done", **kwargs: Any) -> None:
        if name == "prompt" and kwargs.get("prompt"):
            kinds.append(str(kwargs["prompt"]))

    monkeypatch.setattr(nodes_module, "say", chunks.append)
    monkeypatch.setattr(nodes_module, "stage", _stage)
    return SimpleNamespace(chunks=chunks, kinds=kinds)


@pytest.fixture()
def use_v2(monkeypatch) -> None:
    monkeypatch.setattr(nodes_module.settings, "script_version", "2")
    monkeypatch.setattr(nodes_module.settings, "ladder_deadline_seconds", 5.0)


@pytest.fixture()
def store(monkeypatch) -> MemoryScriptStore:
    from graph.context_agent import ContextDecision
    from graph.context_store import MemoryContextStore

    mem = MemoryScriptStore()
    monkeypatch.setattr(nodes_module, "script_store", mem)
    ctx_mem = MemoryContextStore()
    monkeypatch.setattr(nodes_module, "context_store", ctx_mem)

    async def _no_context(*_a, **_k):
        return ContextDecision(need=False)

    monkeypatch.setattr("graph.contexter.decide_context", _no_context)
    return mem


@pytest.fixture()
def ctx_store(monkeypatch):
    from graph.context_store import MemoryContextStore

    mem = MemoryContextStore()
    monkeypatch.setattr(nodes_module, "context_store", mem)
    return mem


@pytest.fixture()
def model(monkeypatch):
    @asynccontextmanager
    async def _fake_llm(**kwargs: Any):
        yield None

    monkeypatch.setattr(nodes_module, "get_llm", _fake_llm)

    holder: dict[str, Any] = {
        "result": {"reply": "Хорошо."},
        "calls": 0,
        "messages": None,
        "all_messages": [],
    }

    async def _fake_stream(
        llm, messages, *, schema, text_field=None, on_delta=None, budget=None, purpose=None
    ):
        holder["calls"] += 1
        holder["messages"] = messages
        holder["all_messages"].append(messages)
        result = holder["result"]
        if isinstance(result, list):
            result = result[holder["calls"] - 1]
        on_call = holder.get("on_call")
        if on_call is not None:
            await on_call(holder["calls"], messages)
        if isinstance(result, Exception):
            raise result
        if text_field and on_delta is not None and result.get(text_field):
            on_delta(result[text_field])
        return result

    monkeypatch.setattr(nodes_module, "astream_structured", _fake_stream)
    return holder


def _branch_state(reply: str, ctx: ConversationContext) -> dict[str, Any]:
    return {
        **new_state_defaults(),
        "messages": [HumanMessage(content=reply)],
        "script_id": "vector_ru",
        "script_version": "2",
        "current_step": "branch",
        "head_steps": ["branch"],
        "profile": {"city": "Пермь"},
        "turn": 2,
        "conversation_context": ctx.model_dump(),
    }


def _missing_branches_ctx(
    *,
    reply: str,
    status: str,
    dynamic_text: str = "",
) -> ConversationContext:
    return ConversationContext(
        city_slug="perm",
        city_name="Пермь",
        static_text="Статика разговора:\nГород: Пермь (слаг perm).",
        dynamic_text=dynamic_text,
        dynamic_status=status,
        dynamic_reply_hash=reply_hash(reply),
        dynamic_reply=reply,
    )


async def test_лестница_не_запускается_если_данные_на_месте(
    spoken, store, ctx_store, model, use_v2
):
    """Данные на месте — одна штатная генерация, без лестницы."""
    reply = "какие филиалы?"
    ctx = ConversationContext(
        city_slug="perm",
        city_name="Пермь",
        static_text="Статика разговора:\nГород: Пермь (слаг perm).",
        dynamic_text="Филиалы под запрос: ул. Ленина, 1.",
        dynamic_status=DYN_READY,
        dynamic_reply_hash=reply_hash(reply),
    )
    await ctx_store.save("local", ctx)
    model["result"] = {"reply": "Ближайший на Ленина."}
    out = await nodes_module.respond_node(_branch_state(reply, ctx), None)  # type: ignore[arg-type]
    assert model["calls"] == 1
    assert spoken.kinds == ["full"]
    assert out.get("expect_continuation") is False
    assert "В истории — весь разговор" in model["messages"][0].content


async def test_лестница_в_работе_затем_готово(spoken, store, ctx_store, model, use_v2):
    """«в работе» → «готово»: ровно две генерации — заглушка и штатная."""
    reply = "какие филиалы у Просвещения?"
    ctx = _missing_branches_ctx(reply=reply, status=DYN_WORKING)
    await ctx_store.save("local", ctx)

    async def _on_call(n: int, _messages: Any) -> None:
        if n == 1:
            updated = ctx.model_copy(
                update={
                    "dynamic_status": DYN_READY,
                    "dynamic_text": "Филиалы: ул. Ленина, 1.",
                }
            )
            await ctx_store.save("local", updated)

    model["on_call"] = _on_call
    model["result"] = [
        {"reply": "Секунду…"},
        {"reply": "Ближайший на Ленина."},
    ]
    out = await nodes_module.respond_node(_branch_state(reply, ctx), None)  # type: ignore[arg-type]
    assert model["calls"] == 2
    assert spoken.kinds == ["filler", "full"]
    assert out.get("expect_continuation") is False


async def test_лестница_в_работе_поиск_готово(spoken, store, ctx_store, model, use_v2):
    """«в работе» → «в поиске» → «готово»: три сборки по порядку."""
    reply = "какие филиалы у Просвещения?"
    ctx = _missing_branches_ctx(reply=reply, status=DYN_WORKING)
    await ctx_store.save("local", ctx)

    async def _on_call(n: int, _messages: Any) -> None:
        if n == 1:
            await ctx_store.save(
                "local",
                ctx.model_copy(
                    update={
                        "dynamic_status": DYN_SEARCHING,
                        "situation_slug": "филиалы",
                    }
                ),
            )
        elif n == 2:
            await ctx_store.save(
                "local",
                ctx.model_copy(
                    update={
                        "dynamic_status": DYN_READY,
                        "dynamic_text": "Филиалы: ул. Ленина, 1.",
                        "situation_slug": None,
                    }
                ),
            )

    model["on_call"] = _on_call
    model["result"] = [
        {"reply": "Секунду…"},
        {"reply": "Сейчас подберу филиалы."},
        {"reply": "Ближайший на Ленина."},
    ]
    out = await nodes_module.respond_node(_branch_state(reply, ctx), None)  # type: ignore[arg-type]
    assert model["calls"] == 3
    assert spoken.kinds == ["filler", "waiting", "full"]
    assert out.get("expect_continuation") is False


async def test_лестница_статус_не_меняется_штатная(spoken, store, ctx_store, model, use_v2):
    """Статус не меняется — та же сборка не повторяется, сразу штатная."""
    reply = "какие филиалы у Просвещения?"
    ctx = _missing_branches_ctx(reply=reply, status=DYN_WORKING)
    await ctx_store.save("local", ctx)
    model["result"] = [
        {"reply": "Секунду…"},
        {"reply": "Пока точных данных нет, давайте так."},
    ]
    out = await nodes_module.respond_node(_branch_state(reply, ctx), None)  # type: ignore[arg-type]
    assert model["calls"] == 2
    assert spoken.kinds == ["filler", "full"]
    assert out.get("expect_continuation") is False


async def test_лестница_поиск_не_повторяет_waiting(spoken, store, ctx_store, model, use_v2):
    """«в поиске» без смены — waiting один раз, затем штатная."""
    reply = "какие филиалы у Просвещения?"
    ctx = _missing_branches_ctx(reply=reply, status=DYN_SEARCHING)
    await ctx_store.save("local", ctx)
    model["result"] = [
        {"reply": "Сейчас подберу филиалы."},
        {"reply": "Пока точных адресов нет."},
    ]
    out = await nodes_module.respond_node(_branch_state(reply, ctx), None)  # type: ignore[arg-type]
    assert model["calls"] == 2
    assert spoken.kinds == ["waiting", "full"]
    assert out.get("expect_continuation") is False


async def test_лестница_вторая_ступень_слышит_первую(spoken, store, ctx_store, model, use_v2):
    """Промпт второй ступени содержит текст, произнесённый первой."""
    reply = "какие филиалы у Просвещения?"
    ctx = _missing_branches_ctx(reply=reply, status=DYN_WORKING)
    await ctx_store.save("local", ctx)
    first_reply = "Секунду, гляну."

    async def _on_call(n: int, _messages: Any) -> None:
        if n == 1:
            await ctx_store.save(
                "local",
                ctx.model_copy(update={"dynamic_status": DYN_READY}),
            )

    model["on_call"] = _on_call
    model["result"] = [
        {"reply": first_reply},
        {"reply": "Ближайший на Ленина."},
    ]
    await nodes_module.respond_node(_branch_state(reply, ctx), None)  # type: ignore[arg-type]
    assert model["calls"] == 2
    second_history = [m.content for m in model["all_messages"][1][1:]]
    assert first_reply in second_history


async def test_лестница_произнесённое_не_слипается(spoken, store, ctx_store, model, use_v2):
    """Между текстами ступеней в накопителе есть разделитель."""
    reply = "какие филиалы у Просвещения?"
    ctx = _missing_branches_ctx(reply=reply, status=DYN_WORKING)
    await ctx_store.save("local", ctx)
    first = "Секунду, уточню."
    second = "Сейчас подберу филиалы."
    third = "Ближайший на Ленина."

    async def _on_call(n: int, _messages: Any) -> None:
        if n == 1:
            await ctx_store.save(
                "local",
                ctx.model_copy(
                    update={
                        "dynamic_status": DYN_SEARCHING,
                        "situation_slug": "филиалы",
                    }
                ),
            )
        elif n == 2:
            await ctx_store.save(
                "local",
                ctx.model_copy(
                    update={
                        "dynamic_status": DYN_READY,
                        "dynamic_text": "Филиалы: ул. Ленина, 1.",
                        "situation_slug": None,
                    }
                ),
            )

    model["on_call"] = _on_call
    model["result"] = [
        {"reply": first},
        {"reply": second},
        {"reply": third},
    ]
    out = await nodes_module.respond_node(_branch_state(reply, ctx), None)  # type: ignore[arg-type]
    joined = "".join(out.get("spoken") or [])
    assert spoken.kinds == ["filler", "waiting", "full"]
    assert f"{first}{second}" not in joined
    assert f"{second}{third}" not in joined
    assert f"{first} {second}" in joined
    assert f"{second} {third}" in joined


async def test_лестница_смена_хеша_обрывает(spoken, store, ctx_store, model, use_v2):
    """Смена хеша реплики между ступенями обрывает лестницу."""
    reply = "какие филиалы у Просвещения?"
    ctx = _missing_branches_ctx(reply=reply, status=DYN_WORKING)
    await ctx_store.save("local", ctx)

    async def _on_call(n: int, _messages: Any) -> None:
        if n == 1:
            await ctx_store.save(
                "local",
                ctx.model_copy(
                    update={
                        "dynamic_reply_hash": reply_hash("совсем другая реплика"),
                        "dynamic_status": DYN_WORKING,
                    }
                ),
            )

    model["on_call"] = _on_call
    model["result"] = {"reply": "Секунду…"}
    out = await nodes_module.respond_node(_branch_state(reply, ctx), None)  # type: ignore[arg-type]
    assert model["calls"] == 1
    assert spoken.kinds == ["filler"]
    assert out.get("expect_continuation") is False


async def test_лестница_истёкший_дедлайн_штатная(
    spoken, store, ctx_store, model, use_v2, monkeypatch
):
    """Истёкший дедлайн даёт штатную генерацию сразу."""
    monkeypatch.setattr(nodes_module.settings, "ladder_deadline_seconds", 0.0)
    reply = "какие филиалы у Просвещения?"
    ctx = _missing_branches_ctx(reply=reply, status=DYN_WORKING)
    await ctx_store.save("local", ctx)
    model["result"] = {"reply": "Пока без точных адресов."}
    out = await nodes_module.respond_node(_branch_state(reply, ctx), None)  # type: ignore[arg-type]
    assert model["calls"] == 1
    assert spoken.kinds == ["full"]
    assert out.get("expect_continuation") is False
    assert "В истории — весь разговор" in model["messages"][0].content
