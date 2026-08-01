"""Тесты агента контекста: валидация решения без сети."""

from __future__ import annotations

import logging

from graph.context import ConversationContext
from graph.context_agent import ContextDecision, decide_context
from graph.tools_registry import BranchesTool, CityFaqTool, CityTool


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
        self.seen_step_needs: list[str] = []

    async def decide(
        self,
        reply,
        context,
        tools,
        faq_questions,
        branches=(),
        step_needs=(),
    ):
        self.seen_branches = list(branches)
        self.seen_step_needs = list(step_needs)
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


def test_описание_query_упоминает_название_города():
    """Поле query: для инструмента города — название из реплики."""
    description = ContextDecision.model_fields["query"].description or ""
    lowered = description.lower()
    assert "город" in lowered
    assert "назван" in lowered
    assert "пустым" in lowered or "пуст" in lowered


async def test_решение_агента_целиком_в_логе_info(caplog):
    """INFO: инструмент, запрос, предмет, слаги, потребности и реплика в логе."""
    reply = "подскажите филиалы в центре, я из Перми " + ("x" * 50)
    agent = _FakeAgent(
        ContextDecision(
            need=True,
            tool="branches",
            query="центр",
            subject="филиалы",
            branch_slugs=["a", "b"],
        )
    )
    with caplog.at_level(logging.INFO, logger="graph.context_agent"):
        await decide_context(
            reply,
            ConversationContext(city_slug="perm"),
            [BranchesTool()],
            branches=[{"slug": "a"}, {"slug": "b"}],
            step_needs=["филиалы города с адресами"],
            agent=agent,
        )
    info = [r.message for r in caplog.records if r.levelno == logging.INFO]
    assert info
    msg = info[0]
    assert "need=True" in msg
    assert "tool='branches'" in msg
    assert "query='центр'" in msg
    assert "subject='филиалы'" in msg
    assert "branch_slugs=['a', 'b']" in msg
    assert "step_needs=['филиалы города с адресами']" in msg
    assert f"реплика={reply[:80]!r}" in msg
    assert f"реплика={reply!r}" not in msg
    assert len(reply) > 80


async def test_пустой_query_виден_в_логе_как_пустой(caplog):
    """Пустой query в логе как '', а не пропускается."""
    agent = _FakeAgent(
        ContextDecision(
            need=True,
            tool="city",
            query="",
            subject="город",
            branch_slugs=[],
        )
    )
    with caplog.at_level(logging.INFO, logger="graph.context_agent"):
        await decide_context(
            "Пермь",
            ConversationContext(),
            [CityTool()],
            agent=agent,
        )
    info = [r.message for r in caplog.records if r.levelno == logging.INFO]
    assert info
    assert "query=''" in info[0]
    assert "tool='city'" in info[0]
    assert "step_needs=[]" in info[0]


async def test_вызов_без_step_needs_как_раньше():
    """Без нового параметра решение работает; агент видит пустой список."""
    agent = _FakeAgent(ContextDecision(need=False))
    decision = await decide_context(
        "угу",
        ConversationContext(),
        [CityFaqTool()],
        agent=agent,
    )
    assert decision.need is False
    assert agent.seen_step_needs == []


async def test_step_needs_доходят_до_агента():
    """Потребности шага передаются в decide агента."""
    agent = _FakeAgent(ContextDecision(need=False))
    needs = ["стоимость обучения в городе", "срок обучения по городу"]
    await decide_context(
        "хорошо",
        ConversationContext(city_slug="perm"),
        [CityTool()],
        step_needs=needs,
        agent=agent,
    )
    assert agent.seen_step_needs == needs


async def test_llm_агент_кладёт_потребности_в_системное(monkeypatch):
    """В системном сообщении есть раздел с потребностями шага."""
    from contextlib import asynccontextmanager

    import graph.context_agent as agent_module
    from graph.context_agent import LlmContextAgent

    seen: dict[str, str] = {}

    @asynccontextmanager
    async def _fake_llm(**kwargs):
        yield None

    async def _fake_stream(llm, messages, **kwargs):
        seen["system"] = str(messages[0].content)
        return {
            "need": False,
            "tool": None,
            "query": "",
            "subject": "",
            "branch_slugs": [],
        }

    monkeypatch.setattr(agent_module, "get_llm", _fake_llm)
    monkeypatch.setattr(agent_module, "astream_structured", _fake_stream)

    await LlmContextAgent().decide(
        "угу",
        ConversationContext(),
        [CityTool()],
        [],
        [],
        step_needs=["стоимость обучения в городе"],
    )
    system = seen["system"]
    assert "Потребности текущего шага" in system
    assert "стоимость обучения в городе" in system
    assert "даже если клиент об этом не спрашивал" in system


async def test_llm_агент_без_потребностей_раздела_нет(monkeypatch):
    """Пустые потребности — раздела в системном сообщении нет."""
    from contextlib import asynccontextmanager

    import graph.context_agent as agent_module
    from graph.context_agent import LlmContextAgent

    seen: dict[str, str] = {}

    @asynccontextmanager
    async def _fake_llm(**kwargs):
        yield None

    async def _fake_stream(llm, messages, **kwargs):
        seen["system"] = str(messages[0].content)
        return {
            "need": False,
            "tool": None,
            "query": "",
            "subject": "",
            "branch_slugs": [],
        }

    monkeypatch.setattr(agent_module, "get_llm", _fake_llm)
    monkeypatch.setattr(agent_module, "astream_structured", _fake_stream)

    await LlmContextAgent().decide(
        "угу",
        ConversationContext(),
        [CityTool()],
        [],
        [],
    )
    system = seen["system"]
    assert "Потребности текущего шага" not in system
    assert "даже если клиент об этом не спрашивал" not in system


async def test_лог_содержит_список_потребностей(caplog):
    """Строка лога решения содержит список потребностей."""
    agent = _FakeAgent(ContextDecision(need=False))
    with caplog.at_level(logging.INFO, logger="graph.context_agent"):
        await decide_context(
            "ок",
            ConversationContext(),
            [CityTool()],
            step_needs=["срок обучения по городу"],
            agent=agent,
        )
    info = [r.message for r in caplog.records if r.levelno == logging.INFO]
    assert info
    assert "step_needs=['срок обучения по городу']" in info[0]
