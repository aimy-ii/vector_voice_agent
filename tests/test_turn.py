"""Тесты хода графа целиком — с заглушками модели, Redis, чекера и резолверов."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any

import pytest
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from graph import nodes as nodes_module
from graph.checker import CheckerVerdict
from graph.graph import graph
from graph.resolvers import BranchResolution, CityResolution
from script.store import MemoryScriptStore, ScriptProgress
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
    """Ход-тесты идут на v2; .env и «последняя» не должны подменять версию."""
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
def checker(monkeypatch) -> SilentChecker:
    client = SilentChecker()
    monkeypatch.setattr(nodes_module, "_checker_client", client)
    return client


@pytest.fixture()
def kb(monkeypatch, fake_kb):
    monkeypatch.setattr("graph.checker_graph.vector_kb", fake_kb)
    monkeypatch.setattr("kb.client.vector_kb", fake_kb)
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

    async def _fake_stream(
        llm, messages, *, schema, text_field=None, on_delta=None, budget=None, purpose=None
    ):
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


async def test_перечень_городов_не_в_промпте_ни_на_одном_ходу(
    spoken, store, checker, kb, resolvers, model, use_v2
):
    model["result"] = {"understood": [], "reply": "Слушаю."}
    await graph.ainvoke({"messages": [HumanMessage(content="Здравствуйте")]})
    assert "city_choices" not in model["messages"][0].content


async def test_шаг_с_образцом_идёт_через_модель(
    spoken, store, checker, kb, resolvers, model, use_v2
):
    """Шаг practice с образцом — respond, модель зовётся, образец в промпте."""
    from graph.prompts import _SAMPLE_PREFIX

    model["result"] = {
        "understood": [],
        "aside_id": None,
        "resume_step": True,
        "reply": "Практику подбираем под ваш график. Как вам в целом такой подход?",
    }
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
    assert state["current_step"] == "practice"
    assert model["calls"] == 1
    prompt = model["messages"][0].content
    assert _SAMPLE_PREFIX in prompt
    assert "Практика по вашему графику" in prompt
    assert "".join(spoken) == model["result"]["reply"]


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
    spoken, store, checker, kb, resolvers, model, script, use_v2, caplog
):
    model["result"] = LLMTurnFailed("бюджет хода исчерпан")
    with caplog.at_level("WARNING"):
        state = await graph.ainvoke({"messages": [HumanMessage(content="Здравствуйте")]})
    assert script.params.fallback in "".join(spoken)
    assert state["last_error"]
    assert any("Подстановка фолбэка" in rec.message for rec in caplog.records)
    assert any("бюджет хода исчерпан" in rec.message for rec in caplog.records)


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


async def test_слаг_города_доживает_до_следующего_хода(
    spoken, store, checker, kb, resolvers, model, use_v2
):
    """Слаг города из кеша/состояния сохраняется на следующем ходу без резолвера."""
    model["result"] = {"understood": [], "reply": "Хорошо."}
    first_ctx = {
        "static_text": "Город: Пермь",
        "city_slug": "perm",
        "city_name": "Пермь",
    }
    second = await graph.ainvoke(
        {
            "messages": [HumanMessage(content="У меня проспект Просвещения, адрес")],
            "city_slug": "perm",
            "city_name": "Пермь",
            "profile": {
                "caller_name": "Мария",
                "city": "Пермь",
                "student_is_caller": "да",
                "experience": "впервые",
                "transmission": "механика",
            },
            "step_status": {
                "name": "closed",
                "city": "closed",
                "who_studies": "closed",
                "experience": "closed",
                "transmission": "closed",
            },
            "conversation_context": first_ctx,
            "turn": 1,
        }
    )
    assert second["city_slug"] == "perm"
    assert resolvers[0].calls == 0


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
    assert state["script_version"] == "4"


async def test_образец_в_промпте_склейки_мимо_модели_нет(
    spoken, store, checker, kb, resolvers, model, use_v2
):
    """Шаг terms: образец в промпте, в эфир только реплика модели."""
    from graph.contexter import reply_hash
    from graph.prompts import _SAMPLE_PREFIX

    model["result"] = {
        "understood": [],
        "aside_id": None,
        "resume_step": True,
        "reply": "Обучение занимает два с половиной месяца. Как вам такой срок?",
    }
    state = await graph.ainvoke(
        {
            "messages": [HumanMessage(content="механика")],
            "city_slug": "perm",
            "city_name": "Пермь",
            "profile": {
                "city": "Пермь",
                "caller_name": "Мария",
                "student_is_caller": "да",
                "experience": "впервые",
                "transmission": "механика",
            },
            "step_status": {
                "name": "closed",
                "city": "closed",
                "who_studies": "closed",
                "experience": "closed",
                "transmission": "closed",
            },
            "conversation_context": {
                "static_text": "Город: Пермь",
                "city_slug": "perm",
                "city_name": "Пермь",
                "dynamic_reply_hash": reply_hash("механика"),
            },
        }
    )
    assert state["current_step"] == "terms"
    assert model["calls"] == 1
    prompt = model["messages"][0].content
    assert _SAMPLE_PREFIX in prompt
    assert "два с половиной месяца" in prompt or "Расскажу, как всё устроено" in prompt
    assert "".join(spoken) == model["result"]["reply"]


async def test_подтверждение_на_шаге_образца_зовёт_модель(
    spoken, store, checker, kb, resolvers, model, use_v2
):
    """Подтверждение на шаге с образцом — модель всё равно генерит реплику."""
    model["result"] = {
        "understood": [],
        "aside_id": None,
        "resume_step": True,
        "reply": "Расскажу про сроки. Обучение — два с половиной месяца.",
    }
    state = await graph.ainvoke(
        {
            "messages": [HumanMessage(content="да, конечно")],
            "city_slug": "perm",
            "city_name": "Пермь",
            "profile": {
                "city": "Пермь",
                "caller_name": "Мария",
                "student_is_caller": "да",
                "experience": "впервые",
                "transmission": "механика",
            },
            "step_status": {
                "name": "closed",
                "city": "closed",
                "who_studies": "closed",
                "experience": "closed",
                "transmission": "closed",
            },
            "conversation_context": {
                "static_text": "Город: Пермь",
                "city_slug": "perm",
                "city_name": "Пермь",
            },
        }
    )
    assert state["current_step"] == "terms"
    assert model["calls"] == 1


async def test_реплика_по_существу_на_шаге_образца_зовёт_модель(
    spoken, store, checker, kb, resolvers, model, use_v2
):
    """Шаг с образцом + реплика по существу → модель."""
    model["result"] = {
        "understood": [],
        "aside_id": None,
        "resume_step": True,
        "reply": "Хорошо, на механике.",
    }
    state = await graph.ainvoke(
        {
            "messages": [HumanMessage(content="механика")],
            "city_slug": "perm",
            "city_name": "Пермь",
            "profile": {
                "city": "Пермь",
                "caller_name": "Мария",
                "student_is_caller": "да",
                "experience": "впервые",
                "transmission": "механика",
            },
            "step_status": {
                "name": "closed",
                "city": "closed",
                "who_studies": "closed",
                "experience": "closed",
                "transmission": "closed",
            },
            "conversation_context": {
                "static_text": "Город: Пермь",
                "city_slug": "perm",
                "city_name": "Пермь",
            },
        }
    )
    assert state["current_step"] == "terms"
    assert model["calls"] == 1


async def test_счётчик_растёт_у_всех_шагов_шапки(
    spoken, store, checker, kb, resolvers, model, use_v2
):
    """В шапке два висящих и один свежий: плюс каждому из шапки."""
    model["result"] = {
        "understood": [],
        "aside_id": None,
        "resume_step": True,
        "reply": "Как я могу к вам обращаться?",
    }
    state = await graph.ainvoke(
        {
            "messages": [HumanMessage(content="ну не знаю пока")],
            "step_status": {"name": "pending", "city": "pending"},
            "step_attempts": {"name": 1, "city": 1},
            "step_taken_turn": {"name": 1, "city": 1},
            "turn": 3,
        }
    )
    attempts = state.get("step_attempts") or {}
    assert attempts.get("name") == 2
    assert attempts.get("city") == 2
    # Свежий who_studies тоже в шапке — счётчик с нуля.
    assert attempts.get("who_studies") == 1
    assert state["head_steps"] == ["name", "city", "who_studies"]
    assert state.get("head_new_step") == "who_studies"


async def test_head_new_step_none_когда_все_висящие(
    spoken, store, checker, kb, resolvers, model, use_v2, monkeypatch
):
    """Мягкий потолок: шапка из висящих — head_new_step is None."""
    monkeypatch.setattr(nodes_module.settings, "pending_steps_soft_cap", 2)
    model["result"] = {
        "understood": [],
        "aside_id": None,
        "resume_step": True,
        "reply": "Как я могу к вам обращаться?",
    }
    state = await graph.ainvoke(
        {
            "messages": [HumanMessage(content="ну не знаю пока")],
            "step_status": {"name": "pending", "city": "pending"},
            "step_attempts": {"name": 1, "city": 1},
            "step_taken_turn": {"name": 1, "city": 1},
            "turn": 3,
        }
    )
    assert state["head_steps"] == ["name", "city"]
    assert state.get("head_new_step") is None


async def test_taken_turn_всем_шапке_без_перезаписи(
    spoken, store, checker, kb, resolvers, model, use_v2
):
    """taken_turn проставляется всем шагам шапки и не перезаписывается."""
    model["result"] = {
        "understood": [],
        "aside_id": None,
        "resume_step": True,
        "reply": "Как я могу к вам обращаться?",
    }
    state1 = await graph.ainvoke(
        {
            "messages": [HumanMessage(content="ну не знаю")],
            "step_status": {"name": "pending"},
            "step_attempts": {"name": 1},
            "step_taken_turn": {"name": 1},
            "turn": 1,
        }
    )
    taken1 = state1.get("step_taken_turn") or {}
    assert taken1.get("name") == 1
    assert taken1.get("city") == 2

    model["result"] = {
        "understood": [],
        "aside_id": None,
        "resume_step": True,
        "reply": "В каком городе планируете учиться?",
    }
    state2 = await graph.ainvoke(
        {
            "messages": [HumanMessage(content="пока думаю")],
            "turn": state1["turn"],
        }
    )
    taken2 = state2.get("step_taken_turn") or {}
    assert taken2.get("name") == 1
    assert taken2.get("city") == 2


async def test_pending_всем_шагам_шапки(spoken, store, checker, kb, resolvers, model, use_v2):
    """Статус pending проставляется каждому шагу шапки, в том числе свежему."""
    model["result"] = {
        "understood": [],
        "aside_id": None,
        "resume_step": True,
        "reply": "Как я могу к вам обращаться?",
    }
    state = await graph.ainvoke(
        {
            "messages": [HumanMessage(content="здравствуйте")],
            "step_status": {"name": "pending"},
            "step_attempts": {"name": 1},
            "step_taken_turn": {"name": 1},
        }
    )
    status = state.get("step_status") or {}
    assert status.get("name") == "pending"
    assert status.get("city") == "pending"
    assert "city" in state["head_steps"]


async def test_шаг_шапки_на_следующем_ходу_у_чекера(
    spoken, store, checker, kb, resolvers, model, use_v2, script
):
    """Шаг, попавший в шапку свежим, на следующем ходу — в висящих чекера."""
    from graph.checker import run_checker

    model["result"] = {
        "understood": [],
        "aside_id": None,
        "resume_step": True,
        "reply": "В каком городе планируете учиться?",
    }
    state = await graph.ainvoke(
        {
            "messages": [HumanMessage(content="Андрей")],
            "step_status": {"name": "closed"},
            "step_attempts": {"name": 1},
            "step_taken_turn": {"name": 1},
            "profile": {"caller_name": "Андрей"},
        }
    )
    assert int(state["step_attempts"].get("city", 0)) >= 1
    assert "city" in state["head_steps"]

    class CapturingChecker:
        def __init__(self) -> None:
            self.step_ids: list[str] = []

        async def judge(self, *, step, **kwargs: Any) -> CheckerVerdict:
            self.step_ids.append(step.id)
            return CheckerVerdict(reply_usable=True, step_closed=False)

    client = CapturingChecker()
    progress = ScriptProgress.from_mapping(
        {
            "status": state.get("step_status") or {},
            "attempts": state.get("step_attempts") or {},
            "taken_turn": state.get("step_taken_turn") or {},
        }
    )
    await run_checker(
        script=script,
        progress=progress,
        messages=[
            HumanMessage(content="Андрей"),
            AIMessage(content=model["result"]["reply"]),
            HumanMessage(content="В городе Санкт-Петербург"),
        ],
        profile={"caller_name": "Андрей"},
        turn=int(state["turn"]) + 1,
        client=client,
    )
    assert "city" in client.step_ids


async def test_просьба_повторить_идёт_через_модель(
    spoken, store, checker, kb, resolvers, model, use_v2
):
    """Просьба повторить — обычный ход через модель, обхода нет."""
    model["result"] = {
        "understood": [],
        "aside_id": None,
        "resume_step": True,
        "reply": "Ещё раз: срок обучения — два с половиной месяца.",
    }
    state = await graph.ainvoke(
        {
            "messages": [HumanMessage(content="повторите")],
            "city_slug": "perm",
            "city_name": "Пермь",
            "profile": {
                "city": "Пермь",
                "caller_name": "Мария",
                "student_is_caller": "да",
                "experience": "впервые",
                "transmission": "механика",
            },
            "step_status": {
                "name": "closed",
                "city": "closed",
                "who_studies": "closed",
                "experience": "closed",
                "transmission": "closed",
                "terms": "pending",
            },
            "step_attempts": {
                "name": 1,
                "city": 1,
                "who_studies": 1,
                "experience": 1,
                "transmission": 1,
                "terms": 1,
            },
            "conversation_context": {
                "static_text": "Город: Пермь",
                "city_slug": "perm",
                "city_name": "Пермь",
            },
        }
    )
    assert model["calls"] == 1
    assert state.get("step_attempts", {}).get("terms") == 2
    assert "".join(spoken) == model["result"]["reply"]


async def test_вопрос_на_шаге_образца_зовёт_модель(
    spoken, store, checker, kb, resolvers, model, use_v2
):
    """Шаг с образцом + вопрос → модель отвечает."""
    model["result"] = {
        "understood": [],
        "aside_id": None,
        "resume_step": True,
        "reply": "Сейчас расскажу про сроки.",
    }
    state = await graph.ainvoke(
        {
            "messages": [HumanMessage(content="а сколько стоит?")],
            "city_slug": "perm",
            "city_name": "Пермь",
            "profile": {
                "city": "Пермь",
                "caller_name": "Мария",
                "student_is_caller": "да",
                "experience": "впервые",
                "transmission": "механика",
            },
            "step_status": {
                "name": "closed",
                "city": "closed",
                "who_studies": "closed",
                "experience": "closed",
                "transmission": "closed",
            },
            "conversation_context": {
                "static_text": "Город: Пермь",
                "city_slug": "perm",
                "city_name": "Пермь",
            },
        }
    )
    assert state["current_step"] == "terms"
    assert model["calls"] == 1


async def test_просьба_рассказать_на_шаге_образца_зовёт_модель(
    spoken, store, checker, kb, resolvers, model, use_v2
):
    """Шаг с образцом + просьба рассказать → модель."""
    model["result"] = {
        "understood": [],
        "aside_id": None,
        "resume_step": True,
        "reply": "На автомате педаль сцепления не нужна.",
    }
    state = await graph.ainvoke(
        {
            "messages": [
                HumanMessage(content="а-а, хотел бы обучаться на механике. Расскажите про автомат")
            ],
            "city_slug": "perm",
            "city_name": "Пермь",
            "profile": {
                "city": "Пермь",
                "caller_name": "Мария",
                "student_is_caller": "да",
                "experience": "впервые",
                "transmission": "механика",
            },
            "step_status": {
                "name": "closed",
                "city": "closed",
                "who_studies": "closed",
                "experience": "closed",
                "transmission": "closed",
            },
            "conversation_context": {
                "static_text": "Город: Пермь",
                "city_slug": "perm",
                "city_name": "Пермь",
            },
        }
    )
    assert state["current_step"] == "terms"
    assert model["calls"] == 1


async def test_граф_без_узла_check_порядок_ingest_plan_respond_commit():
    """Топология: ingest → plan → respond → commit; check и lookup нет."""
    from graph.graph import build_graph

    built = build_graph()
    nodes = set(built.nodes)
    assert "check" not in nodes
    assert "lookup" not in nodes
    assert "verbatim" not in nodes
    assert nodes == {"ingest", "plan", "respond", "commit"}

    edges = built.edges
    assert ("ingest", "plan") in edges
    assert ("plan", "respond") in edges
    assert ("respond", "commit") in edges
    assert not any(src == "check" or tgt == "check" for src, tgt in edges)
    assert not any(src == "lookup" or tgt == "lookup" for src, tgt in edges)
    assert not built.waiting_edges


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


async def test_plan_читает_закрытия_лайв_из_кеша(
    spoken, store, checker, kb, resolvers, model, use_v2
):
    """Plan берёт статусы из Redis: закрытое лайвом не в шапке, следующий шаг."""
    from script.store import ScriptProgress

    await store.save(
        "local",
        ScriptProgress(
            status={"name": "closed"},
            attempts={"name": 1},
            taken_turn={"name": 1},
            profile={"caller_name": "Мария"},
        ),
    )
    model["result"] = {
        "understood": [],
        "reply": "В каком городе планируете обучение?",
    }
    state = await graph.ainvoke(
        {
            "messages": [HumanMessage(content="Мария")],
            "profile": {"caller_name": "Мария"},
        }
    )
    assert state.get("step_status", {}).get("name") == "closed"
    assert state["current_step"] == "city"
    assert "name" not in (state.get("head_steps") or [])


async def test_основной_ход_не_ждёт_лайв_и_не_падает(
    spoken, store, checker, kb, resolvers, model, use_v2
):
    """Лайв ещё не закрыл шаг — ход всё равно отрабатывает без ошибки."""
    model["result"] = {"understood": [], "reply": "Как к вам обращаться?"}
    state = await graph.ainvoke({"messages": [HumanMessage(content="Здравствуйте")]})
    assert state["current_step"] == "name"
    assert model["calls"] == 1
    assert "".join(spoken) == model["result"]["reply"]
    # Чекер основного хода не зовётся (узла check нет).
    assert checker.calls == 0


async def test_respond_не_зовёт_контекстер_если_dynamic_reply_совпал(
    spoken, store, checker, kb, resolvers, model, monkeypatch, use_v2
):
    """Основной ход не зовёт контекстер: читает динамику из кеша при совпадении."""
    from graph.context import DYN_READY, ConversationContext
    from graph.context_store import MemoryContextStore
    from graph.contexter import reply_hash
    from graph.state import new_state_defaults

    reply = "какие филиалы у Ленина?"
    fact = "Филиалы под запрос: ул. Ленина, 1."
    digest = reply_hash(reply)
    ctx_mem = MemoryContextStore()
    await ctx_mem.save(
        "local",
        ConversationContext(
            static_text="Город: Пермь",
            city_slug="perm",
            dynamic_text=fact,
            dynamic_status=DYN_READY,
            dynamic_reply=reply,
            dynamic_reply_hash=digest,
        ),
    )
    monkeypatch.setattr(nodes_module, "context_store", ctx_mem)

    async def _spy(*args: Any, **kwargs: Any):
        raise AssertionError("контекстер не должен вызываться")

    monkeypatch.setattr("graph.contexter.run_contexter", _spy)
    model["result"] = {"understood": [], "aside_id": None, "reply": "Рядом с Ленина есть филиал."}

    state: dict[str, Any] = {
        **new_state_defaults(),
        "messages": [HumanMessage(content=reply)],
        "script_id": "vector_ru",
        "script_version": "2",
        "current_step": "name",
        "head_steps": ["name"],
        "conversation_context": {
            "static_text": "Город: Пермь",
            "city_slug": "perm",
            "dynamic_text": fact,
            "dynamic_status": DYN_READY,
            "dynamic_reply": reply,
            "dynamic_reply_hash": digest,
        },
    }
    out = await nodes_module.respond_node(state, None)  # type: ignore[arg-type]
    assert fact in out["conversation_context"]["dynamic_text"]
    assert model["calls"] == 1
    prompt = model["messages"][0].content
    assert fact in prompt
    assert "готово" in prompt.lower() or "Город: Пермь" in prompt


async def test_respond_при_чужом_searching_hash_полная_сборка(
    spoken, store, checker, kb, resolvers, model, monkeypatch, use_v2
):
    """SEARCHING с чужим хешем — полная сборка; накопленная динамика остаётся."""
    from graph.context import DYN_READY, DYN_SEARCHING, ConversationContext
    from graph.context_store import MemoryContextStore
    from graph.contexter import reply_hash
    from graph.state import new_state_defaults

    fact = "Секретный факт про филиалы"
    ctx_mem = MemoryContextStore()
    await ctx_mem.save(
        "local",
        ConversationContext(
            static_text="Город: Пермь",
            city_slug="perm",
            dynamic_text=fact,
            dynamic_status=DYN_SEARCHING,
            dynamic_reply_hash=reply_hash("старая реплика"),
            dynamic_reply="старая реплика",
        ),
    )
    monkeypatch.setattr(nodes_module, "context_store", ctx_mem)

    async def _spy(*args: Any, **kwargs: Any):
        raise AssertionError("контекстер не должен вызываться")

    monkeypatch.setattr("graph.contexter.run_contexter", _spy)

    model["result"] = {"understood": [], "aside_id": None, "reply": "Хорошо."}

    state: dict[str, Any] = {
        **new_state_defaults(),
        "messages": [HumanMessage(content="новая реплика")],
        "script_id": "vector_ru",
        "script_version": "2",
        "current_step": "name",
        "head_steps": ["name"],
        "conversation_context": {
            "static_text": "Город: Пермь",
            "city_slug": "perm",
            "dynamic_text": fact,
            "dynamic_status": DYN_SEARCHING,
            "dynamic_reply_hash": reply_hash("старая реплика"),
            "dynamic_reply": "старая реплика",
        },
    }
    out = await nodes_module.respond_node(state, None)  # type: ignore[arg-type]
    assert out.get("expect_continuation") is False
    prompt = model["messages"][0].content
    assert "Шапка скрипта" in prompt
    assert fact in prompt
    assert "Город: Пермь" in prompt
    assert model["calls"] == 1

    # DYN_READY — тоже полная, динамика в промпте.
    ctx = ConversationContext(
        static_text="Город: Пермь",
        city_slug="perm",
        dynamic_text=fact,
        dynamic_status=DYN_READY,
        dynamic_reply="старая реплика",
    )
    await ctx_mem.save("local", ctx)
    state["conversation_context"] = ctx.model_dump()
    await nodes_module.respond_node(state, None)  # type: ignore[arg-type]
    assert fact in model["messages"][0].content


async def test_commit_кладёт_last_agent_reply_в_кеш(
    spoken, store, checker, kb, resolvers, model, monkeypatch, use_v2, ctx_store
):
    """commit_node пишет произнесённую реплику в last_agent_reply."""
    from graph.context import ConversationContext
    from graph.state import new_state_defaults

    await ctx_store.save(
        "local",
        ConversationContext(static_text="Город: Пермь", city_slug="perm"),
    )
    spoken_text = "Рядом с Ленина есть филиал."
    state: dict[str, Any] = {
        **new_state_defaults(),
        "messages": [HumanMessage(content="какие филиалы?")],
        "script_id": "vector_ru",
        "script_version": "2",
        "current_step": "name",
        "spoken": [spoken_text],
        "turn_result": {"understood": [], "aside_id": None, "reply": spoken_text},
        "conversation_context": {"static_text": "Город: Пермь", "city_slug": "perm"},
    }
    out = await nodes_module.commit_node(state, None)  # type: ignore[arg-type]
    assert out["conversation_context"]["last_agent_reply"] == spoken_text
    loaded = await ctx_store.load("local")
    assert loaded is not None
    assert loaded.last_agent_reply == spoken_text
    assert loaded.static_text == "Город: Пермь"


def _closed_through(script_ids: list[str], *, until: str) -> dict[str, str]:
    """Статусы closed для всех шагов до ``until`` включительно."""
    out: dict[str, str] = {}
    for sid in script_ids:
        out[sid] = "closed"
        if sid == until:
            break
    return out


async def test_ход_не_ходит_в_справочник(
    spoken, store, checker, kb, resolvers, model, use_v2, monkeypatch
):
    """За основной ход нет обращений к справочнику и резолверам."""
    calls: list[str] = []

    class _KB:
        async def list_cities(self):
            calls.append("list_cities")
            return []

        async def get_city(self, *a, **k):
            calls.append("get_city")
            return None

        async def list_branches(self, *a, **k):
            calls.append("list_branches")
            return []

        async def get_branch(self, *a, **k):
            calls.append("get_branch")
            return None

    monkeypatch.setattr("graph.checker_graph.vector_kb", _KB())
    model["result"] = {"understood": [], "reply": "Слушаю."}
    await graph.ainvoke({"messages": [HumanMessage(content="Здравствуйте")]})
    assert calls == []
    assert resolvers[0].calls == 0
    assert resolvers[1].calls == 0


async def test_ready_hash_полная_и_короткая_сборка(
    spoken, store, checker, kb, resolvers, model, use_v2, ctx_store
):
    """Ветвление: waiting / filler / full по статусу и недостающим данным."""
    from graph.context import DYN_MISSING, DYN_READY, DYN_SEARCHING, ConversationContext
    from graph.contexter import reply_hash
    from graph.state import new_state_defaults

    reply = "какие филиалы у Просвещения?"
    model["result"] = {"reply": "Сейчас уточню филиалы."}
    digest = reply_hash(reply)

    # Статус в поиске по этой реплике — сборка ожидания.
    ctx = ConversationContext(
        city_slug="perm",
        city_name="Пермь",
        static_text="Статика разговора:\nГород: Пермь (слаг perm).",
        dynamic_status=DYN_SEARCHING,
        pending_fields=["branch"],
        dynamic_reply_hash=digest,
        dynamic_reply=reply,
    )
    await ctx_store.save("local", ctx)
    state = {
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
    assert out.get("expect_continuation") is True
    assert "Шапка скрипта" not in model["messages"][0].content
    assert "предмет" in model["messages"][0].content.lower()

    # Чужой хеш при всём на месте — полная сборка, статус не влияет.
    ctx = ConversationContext(
        city_slug="perm",
        city_name="Пермь",
        static_text="Статика разговора:\nГород: Пермь (слаг perm).",
        dynamic_text="Филиалы под запрос: ул. Ленина, 1.",
        dynamic_status=DYN_SEARCHING,
        pending_fields=["branch"],
        dynamic_reply_hash=reply_hash("другая реплика"),
        dynamic_reply=reply,
    )
    await ctx_store.save("local", ctx)
    state["conversation_context"] = ctx.model_dump()
    model["result"] = {"reply": "Ближайшие на Ленина."}
    out = await nodes_module.respond_node(state, None)  # type: ignore[arg-type]
    assert out.get("expect_continuation") is False
    assert "Шапка скрипта" in model["messages"][0].content

    # DYN_READY и DYN_MISSING — полная сборка.
    for status in (DYN_READY, DYN_MISSING):
        ctx = ConversationContext(
            city_slug="perm",
            static_text="Статика разговора:\nГород: Пермь (слаг perm).",
            dynamic_text="Филиалы под запрос: ул. Ленина, 1.",
            dynamic_status=status,
            dynamic_reply_hash=digest,
            dynamic_reply=reply,
        )
        await ctx_store.save("local", ctx)
        state["conversation_context"] = ctx.model_dump()
        model["result"] = {"reply": "Ок."}
        out = await nodes_module.respond_node(state, None)  # type: ignore[arg-type]
        assert out.get("expect_continuation") is False
        assert "Шапка скрипта" in model["messages"][0].content


async def test_respond_без_знаний_полная_сборка(
    spoken, store, checker, kb, resolvers, model, use_v2, ctx_store
):
    """Ведущий шаг без объявленных знаний и без статуса поиска — полная сборка."""
    from graph.context import ConversationContext
    from graph.state import new_state_defaults

    reply = "Меня Андрей зовут"
    await ctx_store.save("local", ConversationContext())
    model["result"] = {"reply": "Приятно познакомиться, Андрей."}
    state = {
        **new_state_defaults(),
        "messages": [HumanMessage(content=reply)],
        "script_id": "vector_ru",
        "script_version": "2",
        "current_step": "name",
        "head_steps": ["name"],
        "conversation_context": {},
    }
    out = await nodes_module.respond_node(state, None)  # type: ignore[arg-type]
    assert out.get("expect_continuation") is False
    assert "Шапка скрипта" in model["messages"][0].content


async def test_respond_недостающие_факты_живая_реакция(
    spoken, store, checker, kb, resolvers, model, use_v2, ctx_store
):
    """Ведущий шаг с недостающими фактами — filler и expect_continuation."""
    from graph.context import ConversationContext
    from graph.state import new_state_defaults

    reply = "расскажите про сроки"
    # terms на v2 требует city_meta; город известен, статики нет.
    ctx = ConversationContext(city_name="Пермь")
    await ctx_store.save("local", ctx)
    model["result"] = {"reply": "Секунду…"}
    state = {
        **new_state_defaults(),
        "messages": [HumanMessage(content=reply)],
        "script_id": "vector_ru",
        "script_version": "2",
        "current_step": "terms",
        "head_steps": ["terms"],
        "profile": {"city": "Пермь"},
        "conversation_context": ctx.model_dump(),
    }
    out = await nodes_module.respond_node(state, None)  # type: ignore[arg-type]
    assert out.get("expect_continuation") is True
    prompt = model["messages"][0].content
    assert "Шапка скрипта" not in prompt
    assert "думает вслух" in prompt.lower() or "паузу" in prompt.lower()


async def test_respond_searching_даже_если_данные_на_месте(
    spoken, store, checker, kb, resolvers, model, use_v2, ctx_store
):
    """DYN_SEARCHING с хешем текущей реплики — waiting, даже если у шага всё есть."""
    from graph.context import DYN_SEARCHING, ConversationContext
    from graph.contexter import reply_hash
    from graph.state import new_state_defaults

    reply = "ок"
    digest = reply_hash(reply)
    ctx = ConversationContext(
        city_slug="perm",
        city_name="Пермь",
        static_text=(
            "Статика разговора:\nГород: Пермь (слаг perm).\n"
            "Цена (готовая фраза, произносить только так): от 10000"
        ),
        dynamic_status=DYN_SEARCHING,
        dynamic_reply_hash=digest,
        dynamic_reply=reply,
    )
    await ctx_store.save("local", ctx)
    model["result"] = {"reply": "Сейчас подберу стоимость."}
    state = {
        **new_state_defaults(),
        "messages": [HumanMessage(content=reply)],
        "script_id": "vector_ru",
        "script_version": "2",
        "current_step": "price",
        "head_steps": ["price"],
        "profile": {"city": "Пермь"},
        "conversation_context": ctx.model_dump(),
    }
    out = await nodes_module.respond_node(state, None)  # type: ignore[arg-type]
    assert out.get("expect_continuation") is True
    assert "Шапка скрипта" not in model["messages"][0].content
    assert "предмет" in model["messages"][0].content.lower()


async def test_continuation_всегда_полная_сборка(
    spoken, store, checker, kb, resolvers, model, use_v2, ctx_store, monkeypatch
):
    """На turn_kind=continuation всегда полная сборка, даже при SEARCHING."""
    from graph.context import DYN_SEARCHING, ConversationContext
    from graph.contexter import reply_hash
    from graph.state import new_state_defaults

    monkeypatch.setattr(
        "graph.nodes.get_config",
        lambda: {"configurable": {"turn_kind": "continuation", "thread_id": "local"}},
    )
    reply = "Просвещения"
    ctx = ConversationContext(
        static_text="Город: Пермь",
        dynamic_text="Филиалы: Ленина",
        dynamic_status=DYN_SEARCHING,
        dynamic_reply_hash=reply_hash(reply),
        dynamic_reply=reply,
    )
    await ctx_store.save("local", ctx)
    model["result"] = {"reply": "Ближайшие — на Ленина."}
    state = {
        **new_state_defaults(),
        "messages": [
            HumanMessage(content=reply),
            AIMessage(content="Сейчас уточню филиалы."),
        ],
        "script_id": "vector_ru",
        "script_version": "2",
        "current_step": "branch",
        "head_steps": ["branch"],
        "turn": 3,
        "turn_kind": "continuation",
        "conversation_context": ctx.model_dump(),
    }
    out = await nodes_module.respond_node(state, None)  # type: ignore[arg-type]
    assert out.get("expect_continuation") is False
    assert "Шапка скрипта" in model["messages"][0].content


async def test_continuation_не_растит_счётчики(
    spoken, store, checker, kb, resolvers, model, use_v2, monkeypatch
):
    monkeypatch.setattr(
        "graph.nodes.get_config",
        lambda: {"configurable": {"turn_kind": "continuation", "thread_id": "local"}},
    )
    model["result"] = {"reply": "Ближайшие — на Ленина."}
    before_attempts = {"name": 1, "city": 1}
    state = await graph.ainvoke(
        {
            "messages": [
                HumanMessage(content="Пермь"),
                AIMessage(content="Сейчас уточню филиалы."),
            ],
            "step_status": {"name": "closed", "city": "pending"},
            "step_attempts": dict(before_attempts),
            "profile": {"caller_name": "Мария", "city": "Пермь"},
            "city_slug": "perm",
            "city_name": "Пермь",
            "turn": 2,
            "conversation_context": {
                "static_text": "Город: Пермь",
                "city_slug": "perm",
                "city_name": "Пермь",
                "dynamic_status": "готово",
                "dynamic_reply": "Пермь",
                "dynamic_reply_hash": "",
            },
        }
    )
    assert state["turn_kind"] == "continuation"
    # Попытки не выросли относительно входа (plan не плюсует на continuation).
    for sid, count in before_attempts.items():
        assert int(state["step_attempts"].get(sid, 0)) == count
    assert state["step_status"].get("city") != "closed"
