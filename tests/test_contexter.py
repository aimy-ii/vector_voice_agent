"""Тесты контекстера на агенте с подменой решения и инструментов."""

from __future__ import annotations

import asyncio

import pytest

from graph.context import (
    DYN_MISSING,
    DYN_NONE,
    DYN_READY,
    DYN_SEARCHING,
    ConversationContext,
)
from graph.context_agent import ContextDecision
from graph.contexter import run_contexter
from script.models import Objection


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

    async def decide(self, reply, context, tools, faq_questions) -> ContextDecision:
        self.calls.append(reply)
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
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    async def run(self, query: str, context: ConversationContext) -> str:
        self.calls.append(query)
        self.started.set()
        if self.hang:
            await self.release.wait()
            return self.answer
        if self.delay:
            await asyncio.sleep(self.delay)
        return self.answer


async def test_need_false_не_требуется():
    agent = _FakeAgent(ContextDecision(need=False))
    out = await run_contexter(
        ConversationContext(static_text="Город: Пермь"),
        reply="да, хорошо",
        tools=[],
        agent=agent,
    )
    assert out.dynamic_status == DYN_NONE
    assert out.situation_slug is None
    assert out.dynamic_reply == "да, хорошо"
    assert agent.calls == ["да, хорошо"]


async def test_инструмент_вернул_текст_готово():
    tool = _FakeTool("branches", "Филиалы под запрос: ул. Ленина, 1.")
    agent = _FakeAgent(
        ContextDecision(need=True, tool="branches", query="Ленина", subject="филиалы")
    )
    out = await run_contexter(
        ConversationContext(static_text="статика"),
        reply="какие у Ленина?",
        tools=[tool],
        agent=agent,
    )
    assert out.dynamic_status == DYN_READY
    assert "ул. Ленина" in out.dynamic_text
    assert out.dynamic_reply == "какие у Ленина?"
    assert out.static_text == "статика"
    assert out.filler_spoken is False
    assert tool.calls == ["Ленина"]


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


async def test_таймаут_allow_searching_в_поиске(monkeypatch):
    monkeypatch.setattr("graph.contexter.settings.context_tool_timeout", 0.05)
    tool = _FakeTool("branches", "поздно", hang=True)
    agent = _FakeAgent(
        ContextDecision(need=True, tool="branches", query="центр", subject="филиалы")
    )
    out = await run_contexter(
        ConversationContext(city_slug="perm"),
        reply="какие в центре?",
        tools=[tool],
        agent=agent,
        allow_searching=True,
    )
    assert out.dynamic_status == DYN_SEARCHING
    assert out.situation_slug == "филиалы"
    assert out.dynamic_reply == "какие в центре?"
    tool.release.set()


async def test_таймаут_без_allow_searching_дожидается(monkeypatch):
    monkeypatch.setattr("graph.contexter.settings.context_tool_timeout", 0.01)
    tool = _FakeTool("branches", "Филиалы под запрос: ул. Мира, 2.", delay=0.05)
    agent = _FakeAgent(ContextDecision(need=True, tool="branches", query="Мира", subject="филиалы"))
    out = await run_contexter(
        ConversationContext(city_slug="perm"),
        reply="а на Мира?",
        tools=[tool],
        agent=agent,
        allow_searching=False,
    )
    assert out.dynamic_status == DYN_READY
    assert "Мира" in out.dynamic_text
    assert out.situation_slug is None


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
