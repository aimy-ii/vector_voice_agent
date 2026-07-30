"""Тесты агента контекста: валидация решения без сети."""

from __future__ import annotations

from graph.context import ConversationContext
from graph.context_agent import ContextDecision, decide_context
from graph.tools_registry import BranchesTool, CityFaqTool


class _FakeAgent:
    """Возвращает заданное решение или падает."""

    def __init__(
        self,
        decision: ContextDecision | None = None,
        *,
        error: Exception | None = None,
    ) -> None:
        self.decision = decision or ContextDecision()
        self.error = error
        self.seen_branches: list = []

    async def decide(self, reply, context, tools, faq_questions, branches=()):
        self.seen_branches = list(branches)
        if self.error is not None:
            raise self.error
        return self.decision


async def test_валидация_слагов_только_из_перечня_не_больше_трёх():
    agent = _FakeAgent(
        ContextDecision(
            need=True,
            tool="branches",
            query="центр",
            subject="филиалы",
            branch_slugs=["a", "чужой", "b", "c", "d", "a"],
        )
    )
    branches = [
        {"slug": "a", "address": "А"},
        {"slug": "b", "address": "Б"},
        {"slug": "c", "address": "В"},
        {"slug": "d", "address": "Г"},
    ]
    decision = await decide_context(
        "какие в центре?",
        ConversationContext(city_slug="perm"),
        [BranchesTool()],
        branches=branches,
        agent=agent,
    )
    assert decision.need is True
    assert decision.tool == "branches"
    assert decision.branch_slugs == ["a", "b", "c"]


async def test_branches_без_валидных_слагов_need_false():
    agent = _FakeAgent(
        ContextDecision(
            need=True,
            tool="branches",
            subject="филиалы",
            branch_slugs=["чужой"],
        )
    )
    decision = await decide_context(
        "филиалы?",
        ConversationContext(city_slug="perm"),
        [BranchesTool()],
        branches=[{"slug": "a", "address": "А"}],
        agent=agent,
    )
    assert decision.need is False
    assert decision.branch_slugs == []


async def test_subject_длиннее_трёх_слов_обрезается():
    agent = _FakeAgent(
        ContextDecision(
            need=True,
            tool="city_faq",
            query="Медкомиссия?",
            subject="медкомиссия для записи на практику завтра",
        )
    )
    decision = await decide_context(
        "медкомиссия?",
        ConversationContext(city_faq=[{"question": "Медкомиссия?", "answer": "да"}]),
        [CityFaqTool()],
        agent=agent,
    )
    assert decision.subject == "медкомиссия для записи"
    assert decision.need is True


async def test_пустой_subject_остаётся_пустым():
    agent = _FakeAgent(ContextDecision(need=False, subject=""))
    decision = await decide_context(
        "угу",
        ConversationContext(),
        [CityFaqTool()],
        agent=agent,
    )
    assert decision.subject == ""
    assert decision.need is False


async def test_сбой_агента_пустое_решение_без_исключения():
    agent = _FakeAgent(error=RuntimeError("модель недоступна"))
    decision = await decide_context(
        "вопрос",
        ConversationContext(),
        [BranchesTool()],
        agent=agent,
    )
    assert decision == ContextDecision()
