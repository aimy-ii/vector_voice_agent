"""Лестница ожидания в ``respond_node``: filler → waiting → full."""

from __future__ import annotations

from contextlib import asynccontextmanager
from types import SimpleNamespace
from typing import Any

import pytest
from langchain_core.messages import HumanMessage

from graph import nodes as nodes_module
from graph.context import (
    DYN_NONE,
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
    assert "# СЕЙЧАС ГОВОРИМ ОБ ЭТОМ" in model["messages"][0].content


async def test_статус_в_работе_даёт_одну_заглушку(spoken, store, ctx_store, model, use_v2):
    """«В работе» — ровно одна генерация-заглушка, и ход на этом кончен."""
    reply = "какие филиалы у Просвещения?"
    ctx = _missing_branches_ctx(reply=reply, status=DYN_WORKING)
    await ctx_store.save("local", ctx)
    model["result"] = {"reply": "Секунду, гляну."}
    out = await nodes_module.respond_node(_branch_state(reply, ctx), None)  # type: ignore[arg-type]
    assert model["calls"] == 1
    assert spoken.kinds == ["filler"]
    assert out.get("expect_continuation") is False


async def test_статус_поиска_даёт_одну_заглушку(spoken, store, ctx_store, model, use_v2):
    """«В поиске» — одна генерация ожидания, дальше ход не идёт."""
    reply = "какие филиалы у Просвещения?"
    ctx = _missing_branches_ctx(reply=reply, status=DYN_SEARCHING)
    await ctx_store.save("local", ctx)
    model["result"] = {"reply": "Сейчас подберу филиалы."}
    out = await nodes_module.respond_node(_branch_state(reply, ctx), None)  # type: ignore[arg-type]
    assert model["calls"] == 1
    assert spoken.kinds == ["waiting"]
    assert out.get("expect_continuation") is False


async def test_данные_подъехали_во_время_генерации_второй_ступени_нет(
    spoken, store, ctx_store, model, use_v2
):
    """Даже если фон успел с данными, второй генерации в этом ходе не будет.

    Продолжение — работа бота: он выдержит паузу после реплики без вопроса
    и заведёт следующий ход сам. Мозг за один ход говорит один раз.
    """
    reply = "какие филиалы у Просвещения?"
    ctx = _missing_branches_ctx(reply=reply, status=DYN_WORKING)
    await ctx_store.save("local", ctx)

    async def _on_call(n: int, _messages: Any) -> None:
        if n == 1:
            await ctx_store.save(
                "local",
                ctx.model_copy(
                    update={
                        "dynamic_status": DYN_READY,
                        "dynamic_text": "Филиалы: ул. Ленина, 1.",
                    }
                ),
            )

    model["on_call"] = _on_call
    model["result"] = [{"reply": "Секунду…"}, {"reply": "Ближайший на Ленина."}]
    out = await nodes_module.respond_node(_branch_state(reply, ctx), None)  # type: ignore[arg-type]
    assert model["calls"] == 1
    assert spoken.kinds == ["filler"]
    assert "Ближайший на Ленина." not in "".join(out.get("spoken") or [])


async def test_за_ход_склеивать_нечего(spoken, store, ctx_store, model, use_v2):
    """В эфир уходит ровно один текст: разделителей и склеек не бывает.

    Это и есть починка боевой жалобы «Сейчас уточню… Одну минуту. Поняла.
    По коробке уже определились?» — трёх ступеней одним вдохом.
    """
    reply = "какие филиалы у Просвещения?"
    ctx = _missing_branches_ctx(reply=reply, status=DYN_WORKING)
    await ctx_store.save("local", ctx)
    model["result"] = {"reply": "Секунду, уточню."}
    out = await nodes_module.respond_node(_branch_state(reply, ctx), None)  # type: ignore[arg-type]
    assert "".join(out.get("spoken") or []) == "Секунду, уточню."
    assert spoken.chunks == ["Секунду, уточню."]


async def test_без_статуса_фона_штатная_генерация(spoken, store, ctx_store, model, use_v2):
    """Фон ещё не взял реплику — говорим штатно, заглушка не нужна."""
    reply = "какие филиалы у Просвещения?"
    ctx = _missing_branches_ctx(reply=reply, status=DYN_NONE)
    await ctx_store.save("local", ctx)
    model["result"] = {"reply": "Пока без точных адресов."}
    out = await nodes_module.respond_node(_branch_state(reply, ctx), None)  # type: ignore[arg-type]
    assert model["calls"] == 1
    assert spoken.kinds == ["full"]
    assert "# СЕЙЧАС ГОВОРИМ ОБ ЭТОМ" in model["messages"][0].content
    assert out.get("expect_continuation") is False


def test_условие_входа_в_лестницу_только_нехватка_данных():
    """Заглушка включается только по нехватке данных ведущего шага."""
    import inspect

    source = inspect.getsource(nodes_module.respond_node)
    assert "use_ladder = bool(lead_missing) and not _no_client_reply(turn_kind)" in source
    assert "lead_missing = missing_needs(ctx, needs_of(lead), profile) if lead else []" in source


def test_за_ход_ровно_одна_генерация_в_коде():
    """В генераторе не осталось цикла ступеней: одна ступень — один ход.

    Проверка по исходнику намеренно: цикл было легко вернуть правкой
    «на всякий случай», а в трубке это снова даст склейку.
    """
    import inspect

    source = inspect.getsource(nodes_module.respond_node)
    assert "while True:" not in source
    assert 'spoken.append(" ")' not in source


async def test_вытаскивание_не_повторяет_заглушку(spoken, store, ctx_store, model, use_v2):
    """Ход, который бот завёл сам, говорит по делу, а не заглушкой снова.

    Боевой прогон: «Поняла, Приморский район рядом с домом» → пауза →
    та же фраза слово в слово. Заглушку слышат один раз; продолжение —
    содержательная реплика с тем, что есть.
    """
    reply = "какие филиалы у Просвещения?"
    ctx = _missing_branches_ctx(reply=reply, status=DYN_WORKING)
    await ctx_store.save("local", ctx)
    state = {**_branch_state(reply, ctx), "turn_kind": "pull"}
    model["result"] = {"reply": "Пока адресов нет, но филиалы есть в каждом районе."}
    out = await nodes_module.respond_node(state, None)  # type: ignore[arg-type]
    assert model["calls"] == 1
    assert spoken.kinds == ["full"]
    assert out.get("expect_continuation") is False


@pytest.mark.parametrize("kind", ["continuation", "silence", "pull"])
async def test_ход_без_реплики_человека_не_даёт_заглушку(
    spoken, store, ctx_store, model, use_v2, kind: str
):
    """Ни один ход без реплики человека не уходит в заглушку."""
    reply = "какие филиалы у Просвещения?"
    ctx = _missing_branches_ctx(reply=reply, status=DYN_SEARCHING)
    await ctx_store.save("local", ctx)
    state = {**_branch_state(reply, ctx), "turn_kind": kind}
    model["result"] = {"reply": "Скажу по тому, что есть."}
    await nodes_module.respond_node(state, None)  # type: ignore[arg-type]
    assert spoken.kinds == ["full"]
