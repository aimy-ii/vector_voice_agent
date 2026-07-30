"""Тесты контекстера на агенте с подменой решения и инструментов."""

from __future__ import annotations

import asyncio
import inspect

import pytest

from graph.context import (
    DYN_MISSING,
    DYN_NONE,
    DYN_READY,
    DYN_SEARCHING,
    ConversationContext,
)
from graph.context_agent import ContextDecision
from graph.context_store import MemoryContextStore
from graph.contexter import reply_hash, run_contexter
from script.models import Objection


@pytest.fixture(autouse=True)
def _offline_contexter_store(monkeypatch):
    """Офлайн: кеш контекста в памяти, без Redis."""
    from graph import contexter as contexter_module

    mem = MemoryContextStore()
    monkeypatch.setattr(contexter_module, "context_store", mem)
    monkeypatch.setattr(contexter_module, "_call_id", lambda: "local")
    return mem


class _FakeAgent:
    """Фейковый агент: возвращает заранее заданное решение или падает."""

    def __init__(
        self,
        decision: ContextDecision | None = None,
        *,
        error: Exception | None = None,
    ) -> None:
        self.decision = decision or ContextDecision()
        self.error = error
        self.calls: list[str] = []
        self.branches: list = []

    async def decide(self, reply, context, tools, faq_questions, branches=()) -> ContextDecision:
        self.calls.append(reply)
        self.branches = list(branches)
        if self.error is not None:
            raise self.error
        return self.decision


class _FakeTool:
    """Фейковый инструмент: отдаёт заданный ответ или зависает."""

    def __init__(
        self,
        name: str,
        answer: str = "",
        *,
        delay: float = 0.0,
        hang: bool = False,
    ) -> None:
        self.name = name
        self.description = f"тест {name}"
        self.answer = answer
        self.delay = delay
        self.hang = hang
        self.calls: list[str] = []
        self.slugs_seen: list[list[str]] = []
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    async def run(self, query: str, context: ConversationContext, *, slugs=()) -> str:
        self.calls.append(query)
        self.slugs_seen.append(list(slugs))
        self.started.set()
        if self.hang:
            await self.release.wait()
            return self.answer
        if self.delay:
            await asyncio.sleep(self.delay)
        return self.answer


def test_run_contexter_без_allow_searching():
    """Аргумента allow_searching больше нет; needs и branches есть."""
    params = inspect.signature(run_contexter).parameters
    assert "allow_searching" not in params
    assert "branches" in params
    assert "needs" in params


async def test_need_false_не_требуется():
    agent = _FakeAgent(ContextDecision(need=False))
    seeded = ConversationContext(static_text="Город: Пермь", dynamic_status=DYN_READY)
    out = await run_contexter(
        seeded,
        reply="да, хорошо",
        tools=[],
        agent=agent,
    )
    # Нет потребностей и агент не зовёт инструмент — статус не трогаем.
    assert out.dynamic_status == DYN_READY
    assert out.situation_slug is None
    assert out.dynamic_reply == "да, хорошо"
    assert agent.calls == ["да, хорошо"]


async def test_инструмент_вернул_текст_готово():
    tool = _FakeTool("branches", "Филиалы под запрос: ул. Ленина, 1.")
    agent = _FakeAgent(
        ContextDecision(
            need=True,
            tool="branches",
            query="Ленина",
            subject="филиалы",
            branch_slugs=["perm_lenina"],
        )
    )
    out = await run_contexter(
        ConversationContext(static_text="статика"),
        reply="какие у Ленина?",
        tools=[tool],
        agent=agent,
        branches=[{"slug": "perm_lenina", "address": "ул. Ленина, 1"}],
    )
    assert out.dynamic_status == DYN_READY
    assert "ул. Ленина" in out.dynamic_text
    assert out.dynamic_reply == "какие у Ленина?"
    assert out.static_text == "статика"
    assert out.filler_spoken is False
    assert tool.calls == ["Ленина"]
    assert tool.slugs_seen == [["perm_lenina"]]


async def test_пустой_ответ_не_нашлось():
    tool = _FakeTool("city_faq", "")
    agent = _FakeAgent(
        ContextDecision(need=True, tool="city_faq", query="дайвинг?", subject="дайвинг")
    )
    out = await run_contexter(
        ConversationContext(),
        reply="а как на дайвинг?",
        tools=[tool],
        agent=agent,
    )
    assert out.dynamic_status == DYN_MISSING
    assert out.dynamic_text == ""
    assert out.dynamic_reply == "а как на дайвинг?"


async def test_долгий_инструмент_дожидается_готово():
    """Таймаута нет: долгий инструмент отрабатывает до конца."""
    tool = _FakeTool("branches", "Филиалы под запрос: ул. Мира, 2.", delay=0.05)
    agent = _FakeAgent(
        ContextDecision(
            need=True,
            tool="branches",
            query="Мира",
            subject="филиалы",
            branch_slugs=["perm_mira"],
        )
    )
    out = await run_contexter(
        ConversationContext(city_slug="perm"),
        reply="а на Мира?",
        tools=[tool],
        agent=agent,
        branches=[{"slug": "perm_mira", "address": "ул. Мира, 2"}],
    )
    assert out.dynamic_status == DYN_READY
    assert "Мира" in out.dynamic_text
    assert out.situation_slug is None
    assert tool.slugs_seen == [["perm_mira"]]


async def test_branches_без_валидных_слагов_не_требуется():
    """Инструмент branches без валидных слагов → need=False, статус не требуется."""
    tool = _FakeTool("branches", "не должен ответить")
    agent = _FakeAgent(
        ContextDecision(
            need=True,
            tool="branches",
            query="центр",
            subject="филиалы",
            branch_slugs=["чужой_слаг"],
        )
    )
    out = await run_contexter(
        ConversationContext(city_slug="perm"),
        reply="какие в центре?",
        tools=[tool],
        agent=agent,
        branches=[{"slug": "perm_lenina", "address": "ул. Ленина, 1"}],
    )
    assert out.dynamic_status == DYN_NONE
    assert tool.calls == []


async def test_возражение_не_требуется_без_вызова_агента():
    objections = {
        "think": Objection(
            id="think",
            triggers=["подумаю", "надо подумать"],
            text="Конечно, подумайте.",
            sets={"urgency": "high"},
        )
    }
    agent = _FakeAgent(ContextDecision(need=True, tool="branches", subject="филиалы"))
    tool = _FakeTool("branches", "не должен ответить")
    out = await run_contexter(
        ConversationContext(static_text="Город: Пермь", dynamic_text="было"),
        reply="я ещё подумаю",
        tools=[tool],
        objections=objections,
        agent=agent,
    )
    assert out.dynamic_status == DYN_NONE
    assert out.dynamic_text == "было"
    assert agent.calls == []
    assert tool.calls == []


async def test_агент_упал_не_требуется_исключение_не_летит():
    agent = _FakeAgent(error=RuntimeError("модель недоступна"))
    out = await run_contexter(
        ConversationContext(),
        reply="сколько стоит?",
        tools=[_FakeTool("city_faq", "ответ")],
        agent=agent,
    )
    assert out.dynamic_status == DYN_NONE
    assert out.dynamic_reply == "сколько стоит?"


@pytest.mark.parametrize(
    "decision",
    [
        ContextDecision(need=True, tool="unknown", subject="x"),
        ContextDecision(need=True, tool=None, subject="x"),
    ],
)
async def test_инструмент_вне_реестра_не_требуется(decision: ContextDecision):
    out = await run_contexter(
        ConversationContext(),
        reply="вопрос",
        tools=[_FakeTool("branches", "текст")],
        agent=_FakeAgent(decision),
    )
    assert out.dynamic_status == DYN_NONE


def test_reply_hash_нормализует_регистр_и_пробелы():
    from graph.contexter import reply_hash

    assert reply_hash("  Привет   Мир ") == reply_hash("привет мир")
    assert reply_hash("А") != reply_hash("Б")


async def test_повтор_реплики_не_зовёт_агента():
    agent = _FakeAgent(ContextDecision(need=True, tool="faq", query="медкомиссия"))
    tool = _FakeTool("faq", "Ответ про медкомиссию.")
    ctx = ConversationContext()
    first = await run_contexter(ctx, reply="А медкомиссия?", tools=[tool], agent=agent)
    assert agent.calls == ["А медкомиссия?"]
    assert tool.calls == ["медкомиссия"]
    assert first.last_reply_hash

    agent2 = _FakeAgent(ContextDecision(need=True, tool="faq", query="медкомиссия"))
    tool2 = _FakeTool("faq", "Другой ответ.")
    second = await run_contexter(
        first,
        reply="  а   МЕДКОМИССИЯ? ",
        tools=[tool2],
        agent=agent2,
    )
    assert agent2.calls == []
    assert tool2.calls == []
    assert second.dynamic_text == first.dynamic_text


async def test_изменённая_реплика_зовёт_агента():
    agent = _FakeAgent(ContextDecision(need=True, tool="faq", query="медкомиссия"))
    tool = _FakeTool("faq", "Ответ.")
    first = await run_contexter(
        ConversationContext(),
        reply="А медкомиссия?",
        tools=[tool],
        agent=agent,
    )
    agent2 = _FakeAgent(ContextDecision(need=True, tool="faq", query="пересдача"))
    tool2 = _FakeTool("faq", "Про пересдачу.")
    second = await run_contexter(
        first,
        reply="А пересдача?",
        tools=[tool2],
        agent=agent2,
    )
    assert agent2.calls == ["А пересдача?"]
    assert tool2.calls == ["пересдача"]
    assert second.last_reply_hash != first.last_reply_hash


async def test_needs_без_агента_и_searching_до_справочника(monkeypatch):
    """Потребности шага исполняются без агента; SEARCHING в кеше до KB."""
    from graph import contexter as contexter_module

    mem = MemoryContextStore()
    monkeypatch.setattr(contexter_module, "context_store", mem)
    monkeypatch.setattr(contexter_module, "_call_id", lambda: "local")

    class _NoBranches:
        async def list_branches(self, city_slug):
            raise AssertionError("list_branches не должен зваться до facts")

    monkeypatch.setattr(contexter_module, "vector_kb", _NoBranches())

    statuses_at_kb: list[str] = []

    class _Facts:
        name = "facts"
        description = "факты"

        def __init__(self) -> None:
            self.needs: list[str] = []
            self.calls = 0

        async def run(self, query, context, *, slugs=()):
            self.calls += 1
            loaded = await mem.load("local")
            assert loaded is not None
            statuses_at_kb.append(loaded.dynamic_status)
            assert loaded.dynamic_reply_hash == reply_hash("сколько стоит?")
            assert self.needs == ["price"]
            return 'Факты справочника:\n{"price_line": "от 10000"}'

    tool = _Facts()
    agent = _FakeAgent(ContextDecision(need=False))
    # Без city_slug — _load_branches не ходит в справочник.
    out = await run_contexter(
        ConversationContext(dynamic_turn=2),
        reply="сколько стоит?",
        tools=[tool],
        needs=["price"],
        agent=agent,
    )
    assert agent.calls == ["сколько стоит?"]
    assert tool.calls == 1
    assert statuses_at_kb == [DYN_SEARCHING]
    assert out.dynamic_status == DYN_READY
    assert "price_line" in out.dynamic_text
    assert out.dynamic_reply_hash == reply_hash("сколько стоит?")


async def test_needs_пустой_ответ_missing(monkeypatch):
    """Поход был, данных нет — DYN_MISSING."""
    from graph import contexter as contexter_module

    mem = MemoryContextStore()
    monkeypatch.setattr(contexter_module, "context_store", mem)
    monkeypatch.setattr(contexter_module, "_call_id", lambda: "local")

    class _EmptyFacts:
        name = "facts"
        description = "факты"
        needs: list[str] = []

        async def run(self, query, context, *, slugs=()):
            return ""

    out = await run_contexter(
        ConversationContext(),
        reply="цена?",
        tools=[_EmptyFacts()],
        needs=["price"],
        agent=_FakeAgent(ContextDecision(need=False)),
    )
    assert out.dynamic_status == DYN_MISSING
    assert out.dynamic_reply_hash == reply_hash("цена?")


async def test_без_needs_и_без_агента_статус_не_трогается():
    """Нет потребностей и агент не зовёт инструмент — статус как был."""
    seeded = ConversationContext(
        dynamic_status=DYN_READY,
        dynamic_text="было",
        dynamic_reply_hash="old",
    )
    out = await run_contexter(
        seeded,
        reply="угу",
        tools=[],
        needs=(),
        agent=_FakeAgent(ContextDecision(need=False)),
    )
    assert out.dynamic_status == DYN_READY
    assert out.dynamic_text == "было"
    assert out.dynamic_reply_hash == "old"


async def test_возражение_не_отменяет_потребности_шага():
    """При needs и триггере возражения инструменты шага работают, агент молчит."""
    objections = {
        "price": Objection(
            id="price",
            triggers=["дороговато", "дорого"],
            text="Понимаю, давайте сравним.",
            sets={},
        )
    }

    class _Facts:
        name = "facts"
        description = "факты"
        needs: list[str] = []
        calls = 0

        async def run(self, query, context, *, slugs=()):
            self.calls += 1
            return 'Факты справочника:\n{"price_line": "от 10000"}'

    facts = _Facts()
    agent = _FakeAgent(ContextDecision(need=True, tool="branches", subject="филиалы"))
    branches_tool = _FakeTool("branches", "не должен ответить")
    out = await run_contexter(
        ConversationContext(static_text="Город: Пермь"),
        reply="дороговато",
        tools=[facts, branches_tool],
        needs=["price"],
        objections=objections,
        agent=agent,
    )
    assert facts.calls == 1
    assert agent.calls == []
    assert branches_tool.calls == []
    assert out.dynamic_status == DYN_READY
    assert "price_line" in out.dynamic_text


async def test_возражение_без_needs_не_требуется():
    """Пустые потребности и возражение — DYN_NONE, агент не зовётся."""
    objections = {
        "think": Objection(
            id="think",
            triggers=["подумаю"],
            text="Конечно.",
            sets={},
        )
    }
    agent = _FakeAgent(ContextDecision(need=True, tool="branches", subject="филиалы"))
    tool = _FakeTool("branches", "не должен")
    out = await run_contexter(
        ConversationContext(),
        reply="я ещё подумаю",
        tools=[tool],
        needs=(),
        objections=objections,
        agent=agent,
    )
    assert out.dynamic_status == DYN_NONE
    assert agent.calls == []
    assert tool.calls == []


async def test_филиалы_не_грузятся_если_агент_не_за_ними(monkeypatch):
    """Без инструмента branches справочник филиалов не зовём."""
    from graph import contexter as contexter_module

    calls: list[str] = []

    class _KB:
        async def list_branches(self, city_slug: str):
            calls.append(city_slug)
            return [{"slug": "perm_lenina", "address": "ул. Ленина, 1"}]

    monkeypatch.setattr(contexter_module, "vector_kb", _KB())
    out = await run_contexter(
        ConversationContext(city_slug="perm"),
        reply="угу",
        tools=[_FakeTool("faq", "ответ")],
        agent=_FakeAgent(ContextDecision(need=False)),
    )
    assert calls == []
    assert out.dynamic_status == DYN_NONE


async def test_филиалы_грузятся_когда_агент_выбрал_branches(monkeypatch):
    """При решении tool=branches список подгружается из справочника."""
    from graph import contexter as contexter_module

    calls: list[str] = []

    class _KB:
        async def list_branches(self, city_slug: str):
            calls.append(city_slug)
            return [{"slug": "perm_lenina", "address": "ул. Ленина, 1"}]

    monkeypatch.setattr(contexter_module, "vector_kb", _KB())
    tool = _FakeTool("branches", "Филиалы: ул. Ленина, 1.")
    agent = _FakeAgent(
        ContextDecision(
            need=True,
            tool="branches",
            query="Ленина",
            subject="филиалы",
            branch_slugs=["perm_lenina"],
        )
    )
    out = await run_contexter(
        ConversationContext(city_slug="perm"),
        reply="какие у Ленина?",
        tools=[tool],
        agent=agent,
    )
    assert calls  # хотя бы один поход: для агента и/или для инструмента
    assert all(c == "perm" for c in calls)
    assert tool.calls == ["Ленина"]
    assert out.dynamic_status == DYN_READY
