"""Тесты хода графа целиком — с заглушками модели, Redis, чекера и резолверов."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any

import pytest
from langchain_core.messages import BaseMessage, HumanMessage

from graph import nodes as nodes_module
from graph.checker import CheckerVerdict
from graph.graph import graph
from graph.resolvers import BranchResolution, CityResolution
from script.store import MemoryScriptStore
from utils.llm_gen import LLMTurnFailed


class SilentChecker:
    """Чекер, который ничего не закрывает."""

    def __init__(self) -> None:
        self.calls = 0

    async def judge(self, **kwargs: Any) -> CheckerVerdict:
        self.calls += 1
        return CheckerVerdict(reply_usable=True, step_closed=False)


class ClosingChecker:
    """Чекер, закрывающий переданные шаги по порядку."""

    def __init__(self, close_ids: set[str]) -> None:
        self.close_ids = close_ids

    async def judge(self, *, step, **kwargs: Any) -> CheckerVerdict:
        return CheckerVerdict(reply_usable=True, step_closed=step.id in self.close_ids)


class ExactCity:
    """Резолвер: делегирует exact через resolve_city (вызов не нужен для Перми)."""

    def __init__(self) -> None:
        self.calls = 0

    async def resolve(self, text: str, cities: Any) -> CityResolution:
        self.calls += 1
        return CityResolution(slug=None, name=None)


class SelectBranch:
    """Резолвер филиала с фиксированным ответом."""

    def __init__(self, result: BranchResolution) -> None:
        self.result = result
        self.calls = 0

    async def resolve(self, text: str, branches: Any) -> BranchResolution:
        self.calls += 1
        return self.result


@pytest.fixture()
def spoken(monkeypatch) -> list[str]:
    chunks: list[str] = []
    monkeypatch.setattr(nodes_module, "say", chunks.append)
    monkeypatch.setattr(nodes_module, "stage", lambda *a, **k: None)
    return chunks


@pytest.fixture()
def use_v2(monkeypatch) -> None:
    """Локальный .env может держать SCRIPT_VERSION=1 для идущих звонков."""
    monkeypatch.setattr(nodes_module.settings, "script_version", None)


@pytest.fixture()
def store(monkeypatch) -> MemoryScriptStore:
    mem = MemoryScriptStore()
    monkeypatch.setattr(nodes_module, "script_store", mem)
    return mem


@pytest.fixture()
def checker(monkeypatch) -> SilentChecker:
    client = SilentChecker()
    monkeypatch.setattr(nodes_module, "_checker_client", client)
    return client


@pytest.fixture()
def kb(monkeypatch, fake_kb):
    monkeypatch.setattr(nodes_module, "vector_kb", fake_kb)
    return fake_kb


@pytest.fixture()
def resolvers(monkeypatch):
    city = ExactCity()
    branch = SelectBranch(BranchResolution(slugs=[], selected=None))
    monkeypatch.setattr(nodes_module, "_city_resolver", city)
    monkeypatch.setattr(nodes_module, "_branch_resolver", branch)
    return city, branch


@pytest.fixture()
def model(monkeypatch):
    @asynccontextmanager
    async def _fake_llm(**kwargs: Any):
        yield None

    monkeypatch.setattr(nodes_module, "get_llm", _fake_llm)

    holder: dict[str, Any] = {"result": {"reply": "Хорошо."}, "calls": 0, "messages": None}

    async def _fake_stream(llm, messages, *, schema, text_field=None, on_delta=None, budget=None):
        holder["calls"] += 1
        holder["messages"] = messages
        result = holder["result"]
        if isinstance(result, Exception):
            raise result
        if text_field and on_delta is not None and result.get(text_field):
            on_delta(result[text_field])
        return result

    monkeypatch.setattr(nodes_module, "astream_structured", _fake_stream)
    return holder


async def test_первый_ход_спрашивает_имя(spoken, store, checker, kb, resolvers, model, use_v2):
    model["result"] = {
        "understood": [],
        "aside_id": None,
        "resume_step": True,
        "reply": "Подскажите, как я могу к вам обращаться?",
    }
    state = await graph.ainvoke({"messages": [HumanMessage(content="Здравствуйте, хочу учиться")]})

    assert state["current_step"] == "name"
    assert state["step_attempts"]["name"] == 1
    assert (
        "обращать" in "".join(spoken).lower()
        or "зовут" in "".join(spoken).lower()
        or model["calls"] == 1
    )


async def test_город_фиксируется_резолвером(
    spoken, store, checker, kb, resolvers, model, monkeypatch, use_v2
):
    model["result"] = {
        "understood": [],
        "aside_id": None,
        "resume_step": True,
        "reply": "Отлично, Пермь. Учиться будете сами?",
    }
    # Имя уже закрыто, на шаге города.
    state = await graph.ainvoke(
        {
            "messages": [HumanMessage(content="Пермь")],
            "step_status": {"name": "closed"},
            "step_attempts": {"name": 1},
            "profile": {"caller_name": "Мария"},
        }
    )
    assert state["city_slug"] == "perm"
    assert state["city_name"] == "Пермь"
    prompt = model["messages"][0].content
    assert "city_choices" not in prompt
    assert "list_cities" not in prompt


async def test_перечень_городов_не_в_промпте_ни_на_одном_ходу(
    spoken, store, checker, kb, resolvers, model, use_v2
):
    model["result"] = {"understood": [], "reply": "Слушаю."}
    await graph.ainvoke({"messages": [HumanMessage(content="Здравствуйте")]})
    assert "city_choices" not in model["messages"][0].content


async def test_дословный_practice_с_проверкой(spoken, store, checker, kb, resolvers, model, use_v2):
    state = await graph.ainvoke(
        {
            "messages": [HumanMessage(content="Да, понятно")],
            "city_slug": "perm",
            "city_name": "Пермь",
            "profile": {
                "city": "Пермь",
                "caller_name": "Мария",
                "student_is_caller": "да",
                "experience": "впервые",
                "transmission": "механика",
                "theory_format": "очно",
            },
            "step_status": {
                "name": "closed",
                "city": "closed",
                "who_studies": "closed",
                "experience": "closed",
                "transmission": "closed",
                "terms": "closed",
                "theory_format": "closed",
                "included": "closed",
            },
            "conversation_context": {
                "static_text": "Город: Пермь",
                "city_slug": "perm",
                "city_name": "Пермь",
            },
        }
    )
    текст = "".join(spoken)
    assert state["current_step"] == "practice"
    assert "Как вам в целом такой подход" in текст
    assert model["calls"] == 0


async def test_дословный_price_задаёт_следующий_вопрос(
    spoken, store, checker, kb, resolvers, model, use_v2
):
    state = await graph.ainvoke(
        {
            "messages": [HumanMessage(content="Да")],
            "city_slug": "perm",
            "city_name": "Пермь",
            "branch_slug": "perm_chernyshevskogo",
            "profile": {
                "city": "Пермь",
                "caller_name": "Мария",
                "student_is_caller": "да",
                "experience": "впервые",
                "transmission": "механика",
                "theory_format": "очно",
                "branch": "perm_chernyshevskogo",
            },
            "step_status": {
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
            },
            "conversation_context": {
                "static_text": "Город: Пермь\nфилиал",
                "city_slug": "perm",
                "city_name": "Пермь",
                "branch_slug": "perm_chernyshevskogo",
                "frozen": True,
            },
        }
    )
    текст = "".join(spoken)
    assert state["current_step"] == "price"
    assert model["calls"] == 0
    assert "43900" in текст or "стоимость" in текст.lower()
    assert "частями" in текст.lower() or "целиком" in текст.lower() or "оплатить" in текст.lower()


async def test_генератор_плюсует_счётчик_в_момент_взятия(
    spoken, store, checker, kb, resolvers, model, use_v2
):
    model["result"] = {"understood": [], "reply": "Как к вам обращаться?"}
    state = await graph.ainvoke({"messages": [HumanMessage(content="Здравствуйте")]})
    assert state["step_attempts"]["name"] == 1
    loaded = await store.load("local")
    assert loaded is not None
    assert loaded.attempts["name"] == 1


async def test_возражение_меняет_состояние(spoken, store, checker, kb, resolvers, model, use_v2):
    model["result"] = {
        "understood": [],
        "aside_id": "think",
        "resume_step": False,
        "reply": "Хорошо, спокойно подумайте.",
    }
    state = await graph.ainvoke(
        {
            "messages": [HumanMessage(content="Я подумаю")],
            "city_slug": "perm",
            "city_name": "Пермь",
            "profile": {"city": "Пермь", "caller_name": "Мария"},
            "step_status": {"name": "closed", "city": "closed"},
        }
    )
    assert state["profile"]["urgency"] == "думает"
    assert "think" in state["asides_done"]


async def test_модель_не_ответила_в_эфир_идёт_заглушка(
    spoken, store, checker, kb, resolvers, model, script, use_v2
):
    model["result"] = LLMTurnFailed("бюджет хода исчерпан")
    state = await graph.ainvoke({"messages": [HumanMessage(content="Здравствуйте")]})
    assert script.params.fallback in "".join(spoken)
    assert state["last_error"]


async def test_версия_скрипта_фиксируется_в_состоянии(
    spoken, store, checker, kb, resolvers, model, use_v2
):
    model["result"] = {"understood": [], "reply": "Слушаю."}
    state = await graph.ainvoke({"messages": [HumanMessage(content="Здравствуйте")]})
    assert state["script_id"] == "vector_ru"
    assert state["script_version"] == "2"


async def test_город_из_контекста_подхватывается(
    spoken, store, checker, kb, resolvers, model, use_v2
):
    model["result"] = {"understood": [], "reply": "Слушаю."}
    state = await graph.ainvoke(
        {"messages": [HumanMessage(content="Здравствуйте")]},
        context={"city_slug": "perm"},
    )
    assert state["city_slug"] == "perm"
    assert state["current_step"] != "city"


async def test_вход_словарями_как_у_сервера(spoken, store, checker, kb, resolvers, model, use_v2):
    model["result"] = {
        "understood": [],
        "aside_id": None,
        "resume_step": True,
        "reply": "Как я могу к вам обращаться?",
    }
    state = await graph.ainvoke(
        {"messages": [{"role": "human", "content": "Здравствуйте, хочу на механику"}]}
    )
    assert state["current_step"] == "name"
    assert all(isinstance(m, BaseMessage) for m in state["messages"])


async def test_системный_промпт_бота_не_доезжает_до_модели(
    spoken, store, checker, kb, resolvers, model, use_v2
):
    from langchain_core.messages import SystemMessage

    model["result"] = {"understood": [], "reply": "Слушаю."}
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


async def test_пустая_версия_из_env_берёт_последнюю(
    spoken, store, checker, kb, resolvers, model, monkeypatch
):
    from core.config import Settings

    monkeypatch.setattr(
        nodes_module.settings,
        "script_version",
        Settings(script_version="").script_version,
    )
    model["result"] = {"understood": [], "reply": "Слушаю."}
    state = await graph.ainvoke({"messages": [{"role": "human", "content": "Здравствуйте"}]})
    assert nodes_module.settings.script_version is None
    assert state["script_version"] == "2"


async def test_заглушка_города_без_модели_и_видна_генератору(
    spoken, store, checker, kb, resolvers, model, monkeypatch, use_v2
):
    monkeypatch.setattr(nodes_module.settings, "lookup_fillers_enabled", True)
    model["result"] = {"understood": [], "reply": "Учиться будете сами?"}
    state = await graph.ainvoke(
        {
            "messages": [HumanMessage(content="Пермь")],
            "step_status": {"name": "closed"},
            "profile": {"caller_name": "Мария"},
        }
    )
    assert state.get("spoken_filler")
    assert (
        "Пермь" in (state["spoken_filler"] or "")
        or "перм" in (state["spoken_filler"] or "").lower()
    )
    # Резолвер LLM не звался для точного совпадения.
    assert resolvers[0].calls == 0
    prompt = model["messages"][0].content
    assert "уже ушла фраза" in prompt or state["spoken_filler"] in prompt


async def test_филиал_не_определился_контекст_пуст_шаг_ждёт(
    spoken, store, checker, kb, resolvers, model, monkeypatch, use_v2
):
    monkeypatch.setattr(
        nodes_module,
        "_branch_resolver",
        SelectBranch(
            BranchResolution(
                slugs=["perm_chernyshevskogo", "perm_ekaterininskaya"],
                selected=None,
            )
        ),
    )
    model["result"] = {
        "understood": [],
        "reply": "Могу предложить Чернышевского или Екатерининскую. Что ближе?",
    }
    state = await graph.ainvoke(
        {
            "messages": [HumanMessage(content="а какие есть?")],
            "city_slug": "perm",
            "city_name": "Пермь",
            "profile": {
                "city": "Пермь",
                "caller_name": "Мария",
                "student_is_caller": "да",
                "experience": "впервые",
                "transmission": "механика",
                "theory_format": "очно",
            },
            "step_status": {
                "name": "closed",
                "city": "closed",
                "who_studies": "closed",
                "experience": "closed",
                "transmission": "closed",
                "terms": "closed",
                "theory_format": "closed",
                "included": "closed",
                "practice": "closed",
            },
            "conversation_context": {
                "static_text": "Город: Пермь",
                "city_slug": "perm",
                "city_name": "Пермь",
            },
        }
    )
    assert state.get("branch_slug") in (None, "")
    ctx = state.get("conversation_context") or {}
    assert not ctx.get("branch_slug")
    assert state["current_step"] == "branch"
    assert "branch_options" in (state.get("facts") or {}) or "Чернышевского" in "".join(spoken)


async def test_слепок_в_конце_звонка(spoken, store, checker, kb, resolvers, model, use_v2):
    model["result"] = {
        "understood": [{"key": "messenger", "value": "Telegram"}],
        "reply": "Отправлю.",
    }
    state = await graph.ainvoke(
        {
            "messages": [HumanMessage(content="Telegram")],
            "city_slug": "perm",
            "city_name": "Пермь",
            "branch_slug": "perm_chernyshevskogo",
            "profile": {
                "city": "Пермь",
                "caller_name": "Мария",
                "outcome": "оформлю дистанционно",
                "messenger": "Telegram",
            },
            "step_status": {
                sid: "closed"
                for sid in (
                    "name",
                    "city",
                    "who_studies",
                    "experience",
                    "transmission",
                    "terms",
                    "theory_format",
                    "included",
                    "practice",
                    "branch",
                    "price",
                    "payment",
                    "tax_deduction",
                    "closing",
                )
            },
            "conversation_context": {
                "static_text": "x",
                "city_slug": "perm",
                "city_name": "Пермь",
                "branch_slug": "perm_chernyshevskogo",
                "frozen": True,
            },
        }
    )
    assert state.get("script_progress")
    assert state.get("call_finished") is True or state["current_step"] in (None, "messenger")
