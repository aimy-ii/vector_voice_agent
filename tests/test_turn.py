"""Тесты хода графа целиком — с заглушкой модели, без сети и ключей.

Проверяется связность: каким путём прошёл ход, что прозвучало в трубке, что
осело в состоянии. Модель подменяется функцией, справочник — заглушкой.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from graph import nodes as nodes_module
from graph.graph import graph
from utils.llm_gen import LLMTurnFailed


@pytest.fixture()
def spoken(monkeypatch) -> list[str]:
    """Перехватывает всё, что уходит в трубку."""
    chunks: list[str] = []
    monkeypatch.setattr(nodes_module, "say", chunks.append)
    monkeypatch.setattr(nodes_module, "stage", lambda *a, **k: None)
    return chunks


@pytest.fixture()
def kb(monkeypatch, fake_kb):
    """Подменяет справочник заглушкой."""
    monkeypatch.setattr(nodes_module, "vector_kb", fake_kb)
    return fake_kb


@pytest.fixture()
def model(monkeypatch):
    """Подменяет вызов модели: ответ задаётся тестом, дельты идут в эфир."""

    @asynccontextmanager
    async def _fake_llm(**kwargs: Any):
        yield None

    monkeypatch.setattr(nodes_module, "get_llm", _fake_llm)

    holder: dict[str, Any] = {"result": {"reply": "Хорошо."}, "calls": 0, "messages": None}

    async def _fake_stream(llm, messages, *, schema, text_field, on_delta=None, budget=None):
        holder["calls"] += 1
        holder["messages"] = messages
        result = holder["result"]
        if isinstance(result, Exception):
            raise result
        if on_delta is not None and result.get(text_field):
            on_delta(result[text_field])
        return result

    monkeypatch.setattr(nodes_module, "astream_structured", _fake_stream)
    return holder


async def test_первый_ход_спрашивает_город(spoken, kb, model):
    model["result"] = {
        "understood": [],
        "step_status": "unclear",
        "aside_id": None,
        "resume_step": True,
        "reply": "Давайте сориентирую. В каком городе планируете обучение?",
    }
    state = await graph.ainvoke({"messages": [HumanMessage(content="Здравствуйте, хочу учиться")]})

    assert state["current_step"] == "city"
    assert state["route"] == "lookup"
    assert "городе" in "".join(spoken)
    assert model["calls"] == 1


async def test_город_подтверждается_по_перечислению(spoken, kb, model):
    model["result"] = {
        "understood": [{"key": "city", "value": "perm"}],
        "step_status": "done",
        "aside_id": None,
        "resume_step": True,
        "reply": "Отлично, Пермь.",
    }
    state = await graph.ainvoke({"messages": [HumanMessage(content="Я из Перми")]})

    assert state["city_slug"] == "perm"
    assert state["profile"]["city"] == "perm"
    assert state["step_status"]["city"] == "done"


async def test_город_вне_сети_не_записывается(spoken, kb, model):
    model["result"] = {
        "understood": [{"key": "city", "value": "Москва"}],
        "step_status": "done",
        "aside_id": None,
        "resume_step": True,
        "reply": "Секунду.",
    }
    state = await graph.ainvoke({"messages": [HumanMessage(content="Я из Москвы")]})

    assert state["city_slug"] is None
    assert "city" not in state["profile"]


async def test_перечень_городов_уходит_модели_только_пока_город_неизвестен(spoken, kb, model):
    await graph.ainvoke({"messages": [HumanMessage(content="Здравствуйте")]})
    assert "list_cities" in kb.calls

    kb.calls.clear()
    await graph.ainvoke(
        {
            "messages": [HumanMessage(content="Механика")],
            "city_slug": "perm",
            "profile": {"city": "perm", "caller_name": "Мария", "experience": "впервые"},
            "step_status": {"city": "done", "name": "done", "experience": "done"},
        }
    )
    assert "list_cities" not in kb.calls


async def test_дословный_блок_идёт_мимо_модели(spoken, kb, model):
    """Клиент только поддакнул: питч выталкивается писателем, модель не зовётся."""
    state = await graph.ainvoke(
        {
            "messages": [HumanMessage(content="Да, конечно")],
            "city_slug": "perm",
            "profile": {
                "city": "perm",
                "caller_name": "Мария",
                "experience": "впервые",
                "transmission": "механика",
            },
            "step_status": {
                "city": "done",
                "name": "done",
                "experience": "done",
                "transmission": "done",
            },
        }
    )

    текст = "".join(spoken)
    assert state["current_step"] == "presentation"
    assert model["calls"] == 0
    assert "федеральная академия вождения" in текст.lower()
    assert "Как вам в целом такой подход" in текст
    assert state["step_status"]["presentation"] == "done"


async def test_справка_звучит_перед_дословным_блоком(spoken, kb, model):
    """«А медкомиссия?» — сначала ответ, потом возврат на место."""
    model["result"] = {
        "understood": [],
        "step_status": "unclear",
        "aside_id": "medcheck",
        "resume_step": True,
        "reply": "Медкомиссия нужна к началу практики.",
    }
    state = await graph.ainvoke(
        {
            "messages": [HumanMessage(content="А медкомиссию когда проходить?")],
            "city_slug": "perm",
            "profile": {
                "city": "perm",
                "caller_name": "Мария",
                "experience": "впервые",
                "transmission": "механика",
            },
            "step_status": {
                "city": "done",
                "name": "done",
                "experience": "done",
                "transmission": "done",
            },
        }
    )

    текст = "".join(spoken)
    assert текст.index("Медкомиссия") < текст.index("Расскажу, как проходит")
    assert "medcheck" in state["asides_done"]
    assert model["calls"] == 1


async def test_возражение_меняет_состояние(spoken, kb, model):
    model["result"] = {
        "understood": [],
        "step_status": "refused",
        "aside_id": "think",
        "resume_step": False,
        "reply": "Хорошо, спокойно подумайте.",
    }
    state = await graph.ainvoke(
        {
            "messages": [HumanMessage(content="Я подумаю")],
            "city_slug": "perm",
            "profile": {"city": "perm"},
            "step_status": {"city": "done"},
        }
    )

    assert state["profile"]["urgency"] == "думает"
    assert "think" in state["asides_done"]


async def test_возврат_на_шаг_после_вопроса(spoken, kb, model):
    model["result"] = {
        "understood": [],
        "step_status": "unclear",
        "aside_id": "practice_start",
        "resume_step": True,
        "reply": "Практика доступна после первого занятия.",
    }
    state = await graph.ainvoke(
        {
            "messages": [HumanMessage(content="А когда практика?")],
            "city_slug": "perm",
            "profile": {"city": "perm", "caller_name": "Мария"},
            "step_status": {"city": "done", "name": "done"},
        }
    )

    assert state["current_step"] == "experience"
    assert state["resume_step"] == "experience"
    assert state["step_attempts"]["experience"] == 1


async def test_модель_не_ответила_в_эфир_идёт_заглушка(spoken, kb, model, script):
    model["result"] = LLMTurnFailed("бюджет хода исчерпан")
    state = await graph.ainvoke({"messages": [HumanMessage(content="Здравствуйте")]})

    assert script.params.fallback in "".join(spoken)
    assert state["last_error"]


async def test_реплика_оседает_в_истории_и_запоминается_длина(spoken, kb, model):
    model["result"] = {
        "understood": [],
        "step_status": "unclear",
        "aside_id": None,
        "resume_step": True,
        "reply": "В каком городе планируете обучение?",
    }
    state = await graph.ainvoke({"messages": [HumanMessage(content="Здравствуйте")]})

    последнее = state["messages"][-1]
    assert isinstance(последнее, AIMessage)
    assert последнее.content == "В каком городе планируете обучение?"
    assert state["pending_len"] == len(последнее.content)


async def test_системный_промпт_бота_не_доезжает_до_модели(spoken, kb, model):
    from langchain_core.messages import SystemMessage

    model["result"] = {"understood": [], "step_status": "unclear", "reply": "Слушаю."}
    await graph.ainvoke(
        {
            "messages": [
                SystemMessage(content="Ты менеджер автошколы", id="lk.agent_task.instructions"),
                HumanMessage(content="Здравствуйте"),
            ]
        }
    )
    отправленные = model["messages"]
    assert sum(1 for m in отправленные if m.type == "system") == 1
    assert "Ты менеджер автошколы" not in отправленные[0].content


async def test_перебитый_питч_возвращается_в_работу(spoken, kb, model):
    """Следующий ход видит короткое произнесённое и открывает шаг заново."""
    model["result"] = {"understood": [], "step_status": "unclear", "reply": "Да."}
    state = await graph.ainvoke(
        {
            "messages": [HumanMessage(content="Стоп, а сколько стоит?")],
            "city_slug": "perm",
            "profile": {"city": "perm"},
            "step_status": {"city": "done", "presentation": "done"},
            "pending_step": "presentation",
            "pending_len": 400,
            "pending_ai_count": 0,
        }
    )
    assert state["step_status"]["presentation"] == "open"


async def test_версия_скрипта_фиксируется_в_состоянии(spoken, kb, model):
    model["result"] = {"understood": [], "step_status": "unclear", "reply": "Слушаю."}
    state = await graph.ainvoke({"messages": [HumanMessage(content="Здравствуйте")]})

    assert state["script_id"] == "vector_ru"
    assert state["script_version"] == "1"


async def test_город_из_контекста_подхватывается(spoken, kb, model):
    """Необязательный вход на будущее: сейчас рабочий путь — вопрос клиенту."""
    model["result"] = {"understood": [], "step_status": "unclear", "reply": "Слушаю."}
    state = await graph.ainvoke(
        {"messages": [HumanMessage(content="Здравствуйте")]},
        context={"city_slug": "perm"},
    )
    assert state["city_slug"] == "perm"
    assert state["current_step"] != "city"
