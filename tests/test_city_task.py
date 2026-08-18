"""Долг на данные города переживает обрыв захода и повторяется."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import pytest

from graph.context import (
    ConversationContext,
    city_data_missing,
    record_empty_needs,
)
from graph.context_agent import ContextDecision
from graph.context_store import MemoryContextStore
from graph.contexter import run_contexter
from graph.tools_registry import ContextTool


class _IdleAgent:
    """Агент контекста, который никогда не идёт в справочник."""

    async def decide(
        self,
        reply: str,
        context: ConversationContext,
        tools: Sequence[ContextTool],
        faq_questions: Sequence[str],
        branches: Sequence[Mapping[str, Any]],
        step_needs: Sequence[str] = (),
    ) -> ContextDecision:
        _ = (reply, context, tools, faq_questions, branches, step_needs)
        return ContextDecision(need=False)


class _CityStub:
    """Заглушка инструмента города: запоминает запросы, в сеть не ходит."""

    name = "city"
    description = "город"

    def __init__(self) -> None:
        self.queries: list[str] = []
        self.replies: list[str] = []

    async def run(
        self,
        query: str,
        context: ConversationContext,
        *,
        slugs: Sequence[str] = (),
        reply: str = "",
    ) -> str:
        _ = (context, slugs)
        self.queries.append(query)
        self.replies.append(reply)
        return "Город: Пермь."


@pytest.fixture
def _patch_context_store(monkeypatch: pytest.MonkeyPatch) -> MemoryContextStore:
    store = MemoryContextStore()
    monkeypatch.setattr("graph.contexter.context_store", store)
    return store


def test_city_data_missing_cases() -> None:
    """Проверить признак «город назван, данных нет»."""
    assert city_data_missing(ConversationContext(), {"city": "Пермь"}) is True
    assert city_data_missing(ConversationContext(city_slug="perm"), {"city": "Пермь"}) is False
    assert city_data_missing(ConversationContext(), None) is False
    assert city_data_missing(ConversationContext(), {}) is False
    assert (
        city_data_missing(
            ConversationContext(empty_needs=["city_choices"]),
            {"city": "Пермь"},
        )
        is False
    )


async def test_run_contexter_fetches_city_from_profile_when_reply_has_none(
    _patch_context_store: MemoryContextStore,
) -> None:
    """Проверить поход за городом по анкете, когда в реплике города нет."""
    stub = _CityStub()
    await run_contexter(
        ConversationContext(),
        reply="Сам.",
        tools=[stub],
        needs=(),
        profile={"city": "Пермь"},
        agent=_IdleAgent(),
    )
    assert stub.queries == ["Пермь"]
    assert stub.replies == ["Пермь"]


def test_record_empty_needs_city_retries_then_marks_empty() -> None:
    """Проверить, что город попадает в empty_needs только с третьей неудачи."""
    context = ConversationContext()
    record_empty_needs(context, ["city_choices"], found=False)
    assert context.city_attempts == 1
    assert "city_choices" not in context.empty_needs
    record_empty_needs(context, ["city_choices"], found=False)
    assert context.city_attempts == 2
    assert "city_choices" not in context.empty_needs
    record_empty_needs(context, ["city_choices"], found=False)
    assert context.city_attempts == 3
    assert "city_choices" in context.empty_needs


def test_record_empty_needs_city_success_resets_attempts() -> None:
    """Проверить, что успех обнуляет счётчик и не кладёт город в empty_needs."""
    context = ConversationContext(city_attempts=2)
    record_empty_needs(context, ["city_choices"], found=True)
    assert context.city_attempts == 0
    assert "city_choices" not in context.empty_needs


def test_record_empty_needs_other_need_marks_empty_immediately() -> None:
    """Проверить, что фактовая потребность пустеет с первой неудачи."""
    context = ConversationContext()
    record_empty_needs(context, ["price"], found=False)
    assert "price" in context.empty_needs


async def test_run_contexter_skips_city_tool_when_slug_fixed(
    _patch_context_store: MemoryContextStore,
) -> None:
    """Проверить, что при зафиксированном слага инструмент города не зовут."""
    stub = _CityStub()
    await run_contexter(
        ConversationContext(city_slug="perm"),
        reply="Сам.",
        tools=[stub],
        needs=(),
        profile={"city": "Пермь"},
        agent=_IdleAgent(),
    )
    assert stub.queries == []
    assert stub.replies == []


def test_city_attempts_survives_dump_and_legacy_snapshot() -> None:
    """Проверить сериализацию city_attempts и старый слепок без поля."""
    restored = ConversationContext.model_validate(ConversationContext(city_attempts=2).model_dump())
    assert restored.city_attempts == 2
    payload = ConversationContext().model_dump()
    payload.pop("city_attempts")
    legacy = ConversationContext.model_validate(payload)
    assert legacy.city_attempts == 0
