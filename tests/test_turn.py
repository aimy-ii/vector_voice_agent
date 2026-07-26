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
    """Ход-тесты идут на v2; .env и «последняя» не должны подменять версию."""
    monkeypatch.setattr(nodes_module.settings, "script_version", "2")


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


async def test_город_фиксируется_резолвером(
    spoken, store, checker, kb, resolvers, model, monkeypatch, use_v2
):
    model["result"] = {
        "understood": [],
        "aside_id": None,
        "resume_step": True,
        "reply": "Отлично, Пермь. Учиться будете сами?",
    }
    # Имя уже закрыто, шаг city уже задавали.
    state = await graph.ainvoke(
        {
            "messages": [HumanMessage(content="Пермь")],
            "step_status": {"name": "closed"},
            "step_attempts": {"name": 1, "city": 1},
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
    assert state["route"] == "respond"
    assert model["calls"] == 1
    prompt = model["messages"][0].content
    assert _SAMPLE_PREFIX in prompt
    assert "Практика по вашему графику" in prompt
    assert "".join(spoken) == model["result"]["reply"]


async def test_шаг_price_lookup_затем_respond(spoken, store, checker, kb, resolvers, model, use_v2):
    """Шаг price нуждается в справочнике — lookup → respond, факт и образец в промпте."""
    from graph.prompts import _SAMPLE_PREFIX

    model["result"] = {
        "understood": [],
        "aside_id": None,
        "resume_step": True,
        "reply": "Стоимость — от 43900 рублей. Оплатить можно частями или целиком.",
    }
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
    assert state["current_step"] == "price"
    assert state["route"] == "lookup"
    assert model["calls"] == 1
    prompt = model["messages"][0].content
    assert _SAMPLE_PREFIX in prompt
    assert "43900" in prompt
    текст = "".join(spoken)
    assert "43900" in текст or "стоимость" in текст.lower()


async def test_маршрут_respond_без_справочника(
    spoken, store, checker, kb, resolvers, model, use_v2
):
    """Шаг без needs и без сбора city/branch — сразу respond."""
    model["result"] = {"understood": [], "reply": "Как к вам обращаться?"}
    state = await graph.ainvoke({"messages": [HumanMessage(content="Здравствуйте")]})
    assert state["current_step"] == "name"
    assert state["route"] == "respond"
    assert model["calls"] == 1


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


async def test_слаг_в_состоянии_резолвер_не_зовётся(
    spoken, store, checker, kb, resolvers, model, monkeypatch, use_v2
):
    """На шаге с needs при уже известном слаге resolve_city не вызывается."""
    resolve_calls: list[str] = []
    real = nodes_module.resolve_city

    async def _spy(text: str, cities: Any, *, resolver: Any = None):
        resolve_calls.append(text)
        return await real(text, cities, resolver=resolver)

    monkeypatch.setattr(nodes_module, "resolve_city", _spy)
    model["result"] = {"understood": [], "reply": "Хорошо."}
    state = await graph.ainvoke(
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
            "conversation_context": {
                "static_text": "Город: Пермь",
                "city_slug": "perm",
                "city_name": "Пермь",
            },
        }
    )
    assert state["city_slug"] == "perm"
    assert resolve_calls == []
    assert resolvers[0].calls == 0


async def test_город_из_профиля_точное_совпадение_без_модели(
    spoken, store, checker, kb, resolvers, model, monkeypatch, use_v2
):
    """Имя города уже в профиле, слага нет — ищем по профилю, не по реплике."""
    resolve_calls: list[str] = []
    real = nodes_module.resolve_city

    async def _spy(text: str, cities: Any, *, resolver: Any = None):
        resolve_calls.append(text)
        return await real(text, cities, resolver=resolver)

    monkeypatch.setattr(nodes_module, "resolve_city", _spy)
    model["result"] = {"understood": [], "reply": "Хорошо."}
    state = await graph.ainvoke(
        {
            "messages": [HumanMessage(content="У меня проспект Просвещения, адрес")],
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
            "step_attempts": {
                "name": 1,
                "city": 1,
                "who_studies": 1,
                "experience": 1,
                "transmission": 1,
            },
        }
    )
    assert state["city_slug"] == "perm"
    assert state["city_name"] == "Пермь"
    assert resolve_calls == ["Пермь"]
    assert resolvers[0].calls == 0


async def test_ведущий_шаг_city_ищет_в_реплике(
    spoken, store, checker, kb, resolvers, model, monkeypatch, use_v2
):
    """Ведущий шаг заполняет city — текст для резолва берётся из реплики."""
    resolve_calls: list[str] = []
    real = nodes_module.resolve_city

    async def _spy(text: str, cities: Any, *, resolver: Any = None):
        resolve_calls.append(text)
        return await real(text, cities, resolver=resolver)

    monkeypatch.setattr(nodes_module, "resolve_city", _spy)
    model["result"] = {
        "understood": [],
        "aside_id": None,
        "resume_step": True,
        "reply": "Отлично, Пермь. Учиться будете сами?",
    }
    state = await graph.ainvoke(
        {
            "messages": [HumanMessage(content="Пермь")],
            "step_status": {"name": "closed"},
            "step_attempts": {"name": 1, "city": 1},
            "profile": {"caller_name": "Мария"},
        }
    )
    assert state["city_slug"] == "perm"
    assert resolve_calls == ["Пермь"]
    assert resolvers[0].calls == 0


async def test_заполнитель_города_не_зовётся_при_нулевых_попытках(
    spoken, store, checker, kb, resolvers, model, monkeypatch, use_v2
):
    """Шаг city ещё не задавался — resolve_city не вызывается."""
    resolve_calls: list[str] = []
    real = nodes_module.resolve_city

    async def _spy(text: str, cities: Any, *, resolver: Any = None):
        resolve_calls.append(text)
        return await real(text, cities, resolver=resolver)

    monkeypatch.setattr(nodes_module, "resolve_city", _spy)
    model["result"] = {
        "understood": [],
        "aside_id": None,
        "resume_step": True,
        "reply": "Из какого города?",
    }
    await graph.ainvoke(
        {
            "messages": [HumanMessage(content="Да, хорошо")],
            "step_status": {"name": "closed"},
            "step_attempts": {"name": 1},
            "profile": {"caller_name": "Мария"},
        }
    )
    assert resolve_calls == []
    assert resolvers[0].calls == 0


async def test_заполнитель_города_зовётся_когда_шаг_уже_задавали(
    spoken, store, checker, kb, resolvers, model, monkeypatch, use_v2
):
    """Счётчик city > 0 и поле пустое — резолвер ищет в реплике."""
    resolve_calls: list[str] = []
    real = nodes_module.resolve_city

    async def _spy(text: str, cities: Any, *, resolver: Any = None):
        resolve_calls.append(text)
        return await real(text, cities, resolver=resolver)

    monkeypatch.setattr(nodes_module, "resolve_city", _spy)
    model["result"] = {
        "understood": [],
        "aside_id": None,
        "resume_step": True,
        "reply": "Учиться будете сами?",
    }
    state = await graph.ainvoke(
        {
            "messages": [HumanMessage(content="Пермь")],
            "step_status": {"name": "closed", "city": "pending"},
            "step_attempts": {"name": 1, "city": 1},
            "profile": {"caller_name": "Мария"},
        }
    )
    assert resolve_calls == ["Пермь"]
    assert state["city_slug"] == "perm"


async def test_слаг_города_доживает_до_следующего_хода(
    spoken, store, checker, kb, resolvers, model, monkeypatch, use_v2
):
    """Два хода подряд: после резолва второй lookup не зовёт resolve_city."""
    resolve_calls: list[str] = []
    real = nodes_module.resolve_city

    async def _spy(text: str, cities: Any, *, resolver: Any = None):
        resolve_calls.append(text)
        return await real(text, cities, resolver=resolver)

    monkeypatch.setattr(nodes_module, "resolve_city", _spy)
    model["result"] = {
        "understood": [],
        "aside_id": None,
        "resume_step": True,
        "reply": "Отлично, Пермь. Учиться будете сами?",
    }
    first = await graph.ainvoke(
        {
            "messages": [HumanMessage(content="Пермь")],
            "step_status": {"name": "closed"},
            "step_attempts": {"name": 1, "city": 1},
            "profile": {"caller_name": "Мария"},
        }
    )
    assert first["city_slug"] == "perm"
    assert first.get("conversation_context", {}).get("city_slug") == "perm"
    assert resolve_calls == ["Пермь"]

    model["result"] = {"understood": [], "reply": "Хорошо."}
    second = await graph.ainvoke(
        {
            "messages": [HumanMessage(content="У меня проспект Просвещения, адрес")],
            "city_slug": first["city_slug"],
            "city_name": first["city_name"],
            "profile": {
                **dict(first.get("profile") or {}),
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
            "conversation_context": first.get("conversation_context") or {},
            "turn": first.get("turn") or 1,
        }
    )
    assert second["city_slug"] == "perm"
    assert resolve_calls == ["Пермь"]
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


async def test_заглушка_города_без_модели_и_видна_генератору(
    spoken, store, checker, kb, resolvers, model, monkeypatch, use_v2
):
    monkeypatch.setattr(nodes_module.settings, "lookup_fillers_enabled", True)
    model["result"] = {"understood": [], "reply": "Учиться будете сами?"}
    state = await graph.ainvoke(
        {
            "messages": [HumanMessage(content="Пермь")],
            "step_status": {"name": "closed"},
            "step_attempts": {"name": 1, "city": 1},
            "profile": {"caller_name": "Мария"},
        }
    )
    assert state.get("spoken_filler")
    filler = state["spoken_filler"] or ""
    assert "город" in filler
    assert "Пермь" not in filler
    assert "перм" not in filler.lower()
    # Резолвер LLM не звался для точного совпадения.
    assert resolvers[0].calls == 0
    prompt = model["messages"][0].content
    assert "уже ушла фраза" in prompt or filler in prompt


async def test_заглушка_не_берёт_текст_клиента(
    spoken, store, checker, kb, resolvers, model, monkeypatch, use_v2
):
    monkeypatch.setattr(nodes_module.settings, "lookup_fillers_enabled", True)
    model["result"] = {"understood": [], "reply": "В каком городе?"}
    state = await graph.ainvoke(
        {
            "messages": [HumanMessage(content="Для себя")],
            "step_status": {"name": "closed", "city": "closed"},
            "city_slug": "perm",
            "city_name": "Пермь",
            "profile": {"caller_name": "Мария", "city": "Пермь"},
        }
    )
    # Ход без похода в справочник за городом/филиалом — заглушки нет.
    assert not state.get("spoken_filler")


async def test_заглушка_не_два_хода_подряд(
    spoken, store, checker, kb, resolvers, model, monkeypatch, use_v2
):
    monkeypatch.setattr(nodes_module.settings, "lookup_fillers_enabled", True)
    model["result"] = {"understood": [], "reply": "Учиться будете сами?"}
    state = await graph.ainvoke(
        {
            "messages": [HumanMessage(content="Пермь")],
            "step_status": {"name": "closed"},
            "step_attempts": {"name": 1, "city": 1},
            "profile": {"caller_name": "Мария"},
            "turn": 1,
            "last_filler_turn": 1,
        }
    )
    # turn станет 2, last_filler_turn=1 → подряд, молчим.
    assert not state.get("spoken_filler")


async def test_образец_в_промпте_склейки_мимо_модели_нет(
    spoken, store, checker, kb, resolvers, model, use_v2
):
    """Шаг terms: образец в промпте, в эфир только реплика модели."""
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
            },
        }
    )
    assert state["current_step"] == "terms"
    assert state["route"] == "lookup"
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
    assert state["route"] == "lookup"
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
    assert state["route"] in ("lookup", "respond")


async def test_счётчик_только_у_ведущего_висящие_не_тратят(
    spoken, store, checker, kb, resolvers, model, use_v2
):
    """В шапке два висящих: плюс только ведущему, второй и свежий не растут."""
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
    assert attempts.get("city") == 1
    assert attempts.get("who_studies", 0) == 0


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
    assert state["route"] != "verbatim"


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
    assert state["route"] != "verbatim"
    assert model["calls"] == 1


async def test_граф_без_узла_verbatim_всегда_respond_commit():
    """Топология: check∥lookup → plan → respond → commit; verbatim нет."""
    from graph.graph import build_graph

    built = build_graph()
    nodes = set(built.nodes)
    assert "verbatim" not in nodes
    assert {"ingest", "check", "plan", "lookup", "respond", "commit"} <= nodes

    edges = built.edges
    assert ("ingest", "check") in edges
    assert ("ingest", "lookup") in edges
    assert ("plan", "respond") in edges
    assert ("respond", "commit") in edges
    assert not any(src == "verbatim" or tgt == "verbatim" for src, tgt in edges)
    # Барьер: plan ждёт оба параллельных узла.
    assert ((("check", "lookup"), "plan") in built.waiting_edges) or (
        (("lookup", "check"), "plan") in built.waiting_edges
    )


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


async def test_check_и_lookup_идут_параллельно(
    spoken, store, checker, kb, resolvers, model, monkeypatch, use_v2
):
    """Чекер и резолвер стартуют одновременно, а не по очереди."""
    import asyncio
    import time

    marks: dict[str, float] = {}
    real_check = nodes_module.check_node
    real_lookup = nodes_module.lookup_node

    async def slow_check(state, runtime):
        marks["check_start"] = time.perf_counter()
        await asyncio.sleep(0.08)
        result = await real_check(state, runtime)
        marks["check_end"] = time.perf_counter()
        return result

    async def slow_lookup(state, runtime):
        marks["lookup_start"] = time.perf_counter()
        await asyncio.sleep(0.08)
        result = await real_lookup(state, runtime)
        marks["lookup_end"] = time.perf_counter()
        return result

    # Скомпилированный граф держит ссылки на функции — подменяем узлы через
    # пересборку на лету нельзя; патчим сами корутины в nodes и пересоберём.
    from graph.graph import build_graph

    monkeypatch.setattr(nodes_module, "check_node", slow_check)
    monkeypatch.setattr(nodes_module, "lookup_node", slow_lookup)
    # graph.graph импортировал узлы по имени на момент загрузки модуля.
    import graph.graph as graph_module

    monkeypatch.setattr(graph_module, "check_node", slow_check)
    monkeypatch.setattr(graph_module, "lookup_node", slow_lookup)
    fresh = build_graph().compile(name="vector_agent_parallel_test")

    model["result"] = {"understood": [], "reply": "Отлично, Пермь."}
    state = await fresh.ainvoke(
        {
            "messages": [HumanMessage(content="Пермь")],
            "step_status": {"name": "closed"},
            "step_attempts": {"name": 1, "city": 1},
            "profile": {"caller_name": "Мария"},
        }
    )
    assert "check_start" in marks and "lookup_start" in marks
    # Перекрытие интервалов: каждый стартовал до чужого финиша.
    assert marks["check_start"] < marks["lookup_end"]
    assert marks["lookup_start"] < marks["check_end"]
    assert state["city_slug"] == "perm"


async def test_plan_получает_результаты_check_и_lookup(
    spoken, store, checker, kb, resolvers, model, use_v2
):
    """После барьера plan видит закрытия чекера и слаг из lookup."""
    model["result"] = {
        "understood": [],
        "aside_id": None,
        "resume_step": True,
        "reply": "Отлично, Пермь. Учиться будете сами?",
    }
    state = await graph.ainvoke(
        {
            "messages": [HumanMessage(content="Пермь")],
            "step_status": {"name": "closed"},
            "step_attempts": {"name": 1, "city": 1},
            "profile": {"caller_name": "Мария"},
        }
    )
    assert state["city_slug"] == "perm"
    assert state["current_step"] in ("city", "who_studies")
    assert state.get("step_status", {}).get("name") == "closed"
    # Счётчик ведущего шага выставлен plan после обоих параллельных узлов.
    assert int(state.get("step_attempts", {}).get(state["current_step"], 0)) >= 1


async def test_ошибка_lookup_не_роняет_ход(
    spoken, store, checker, kb, resolvers, model, monkeypatch, use_v2
):
    """Исключение в резолвере глотается; чекер и генератор отрабатывают."""

    async def _boom(*args: Any, **kwargs: Any):
        raise RuntimeError("справочник недоступен")

    monkeypatch.setattr(nodes_module, "_lookup_body", _boom)
    model["result"] = {"understood": [], "reply": "Как к вам обращаться?"}
    state = await graph.ainvoke({"messages": [HumanMessage(content="Здравствуйте")]})
    assert state["current_step"] == "name"
    assert model["calls"] == 1
    assert "".join(spoken) == model["result"]["reply"]
