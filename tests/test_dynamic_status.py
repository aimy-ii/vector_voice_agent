"""Тесты: генератор читает статус динамики."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any

import pytest
from langchain_core.messages import HumanMessage

from graph import nodes as nodes_module
from graph.context import DYN_MISSING, DYN_NONE, DYN_READY, DYN_SEARCHING, ConversationContext
from graph.context_store import MemoryContextStore
from graph.prompts import build_turn_messages, build_waiting_messages, dynamic_status_block
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
        if text_field and on_delta is not None and result.get(text_field):
            on_delta(result[text_field])
        return result

    monkeypatch.setattr(nodes_module, "astream_structured", _fake_stream)
    return holder


#: Указание при «не нашлось»: не объявлять пробел, вести разговор дальше.
_MISSING_SOFT_MARKERS: tuple[str, ...] = (
    "не объявлять о пробеле",
    "вести разговор дальше",
    "вслух об этом не сообщать",
)


def test_промпт_не_нашлось_инструкция(script):
    block = dynamic_status_block(status=DYN_MISSING)
    lowered = block.lower()
    assert "ничего нет" in lowered
    for marker in _MISSING_SOFT_MARKERS:
        assert marker in lowered, marker
    assert "тактично сказать" not in lowered
    # Готовой фразы для произнесения нет — только указание модели.
    assert "«" not in block and "»" not in block
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
    content = messages[0].content.lower()
    assert "ничего нет" in content
    for marker in _MISSING_SOFT_MARKERS:
        assert marker in content, marker


def test_промпт_готово_поиск_не_требуется_без_указания_о_пробеле(script):
    """При готово / в поиске / не требуется указания «не объявлять пробел» нет."""
    for status in (DYN_NONE, DYN_READY, DYN_SEARCHING, ""):
        block = dynamic_status_block(status=status)
        lowered = block.lower()
        assert "не объявлять о пробеле" not in lowered
        assert "вслух об этом не сообщать" not in lowered
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
        content = messages[0].content.lower()
        assert "Город: Пермь".lower() in content
        assert "не объявлять о пробеле" not in content
        assert "вслух об этом не сообщать" not in content


def test_waiting_с_контекстом_короче_полного(script):
    """Укороченный промпт ожидания тянет контекст, но не факты и шапку."""
    history = [HumanMessage(content=f"реплика {i}") for i in range(6)]
    waiting = build_waiting_messages(
        script,
        messages=history,
        profile={"caller_name": "Мария"},
        pending_fields=["branch"],
        step=script.step("branch"),
        history_limit=4,
        context_text="Статика города: Пермь, филиалы по адресам…",
    )
    full = build_turn_messages(
        script=script,
        steps=[script.step("branch")],
        profile={"caller_name": "Мария"},
        facts={"price_line": "Стоимость — 10000"},
        history=history,
        asides_done=[],
        context_text="Статика города: Пермь, филиалы по адресам…",
        pending_fields=["branch"],
    )
    assert "Статика города" in waiting[0].content
    assert "Стоимость — 10000" not in waiting[0].content
    assert "Шапка скрипта" not in waiting[0].content
    assert len(waiting) - 1 <= 4
    assert len(waiting[0].content) < len(full[0].content)


async def test_генератор_без_ready_hash_берёт_waiting(
    spoken, store, ctx_store, model, use_v2, monkeypatch
):
    """Недостающие данные и «в поиске» — лестница со waiting, затем full."""
    kinds: list[str] = []

    def _stage(name: str, _msg: str, _status: str = "done", **kwargs: Any) -> None:
        if name == "prompt" and kwargs.get("prompt"):
            kinds.append(str(kwargs["prompt"]))

    monkeypatch.setattr(nodes_module, "stage", _stage)
    model["result"] = [
        {"reply": "Секунду, уточняю филиалы."},
        {"reply": "Ближайший на Ленина."},
    ]

    async def _spy(*args: Any, **kwargs: Any):
        raise AssertionError("контекстер не должен вызываться из respond")

    monkeypatch.setattr("graph.contexter.run_contexter", _spy)
    from graph.contexter import reply_hash

    reply = "а какие филиалы рядом?"
    ctx = ConversationContext(
        city_slug="perm",
        city_name="Пермь",
        static_text="Город: Пермь",
        dynamic_status=DYN_SEARCHING,
        dynamic_turn=1,
        pending_fields=["branch"],
        dynamic_reply=reply,
        dynamic_reply_hash=reply_hash(reply),
    )
    await ctx_store.save("local", ctx)

    async def _ready(n: int, _messages: Any) -> None:
        if n == 1:
            await ctx_store.save(
                "local",
                ctx.model_copy(
                    update={
                        "dynamic_status": DYN_READY,
                        "dynamic_text": "Филиалы: Ленина",
                    }
                ),
            )

    model["on_call"] = _ready
    state: dict[str, Any] = {
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
    out = await nodes_module.respond_node(state, None)  # type: ignore[arg-type]
    assert out.get("expect_continuation") is False
    assert kinds == ["waiting", "full"]
    prompt = model["all_messages"][0][0].content
    assert "Шапка скрипта" not in prompt
    assert "предмет" in prompt.lower() or "готовиш" in prompt.lower()
    assert model["calls"] == 2


async def test_respond_читает_динамику_из_кеша_без_контекстера(
    spoken, store, ctx_store, model, use_v2, monkeypatch
):
    """Основной ход только читает готовую динамику из кеша."""
    from graph.contexter import reply_hash

    model["result"] = {"aside_id": None, "reply": "Медкомиссия отдельно."}
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
        dynamic_reply_hash=reply_hash(reply),
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
    assert out.get("expect_continuation") is False
    assert model["calls"] == 1
