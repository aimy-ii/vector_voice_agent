"""Тесты реестра инструментов контекстера."""

from __future__ import annotations

from typing import Any

import pytest

from graph.context import ConversationContext
from graph.tools_registry import BranchesTool, CityFaqTool


class _FakeKB:
    """Заглушка справочника: филиалы без вызова модели."""

    def __init__(self, branches: list[dict[str, Any]]) -> None:
        self._branches = branches
        self.calls: list[str] = []
        self.model_calls = 0

    async def list_branches(self, city_slug: str) -> list[dict[str, Any]]:
        self.calls.append(f"list_branches:{city_slug}")
        return list(self._branches)


@pytest.fixture()
def branches_kb(monkeypatch) -> _FakeKB:
    kb = _FakeKB(
        [
            {"slug": "a", "address": "ул. А, 1", "landmark": "парк"},
            {"slug": "b", "address": "ул. Б, 2", "landmark": ""},
            {"slug": "c", "address": "ул. В, 3", "landmark": "метро"},
            {"slug": "d", "address": "ул. Г, 4", "landmark": "рынок"},
        ]
    )
    monkeypatch.setattr("graph.tools_registry.vector_kb", kb)
    return kb


async def test_branches_tool_без_вызова_модели(branches_kb: _FakeKB):
    tool = BranchesTool()
    ctx = ConversationContext(city_slug="perm")
    text = await tool.run("что угодно", ctx, slugs=["b", "a"])
    assert branches_kb.model_calls == 0
    assert branches_kb.calls == ["list_branches:perm"]
    assert text.startswith("Филиалы под запрос:")
    assert "ул. Б, 2" in text
    assert "ул. А, 1 (парк)" in text
    # Порядок слагов агента: сначала b, потом a.
    assert text.index("ул. Б, 2") < text.index("ул. А, 1")


async def test_branches_tool_чужие_слаги_игнорирует_лимит_три(branches_kb: _FakeKB):
    tool = BranchesTool()
    ctx = ConversationContext(city_slug="perm")
    text = await tool.run("", ctx, slugs=["чужой", "d", "a", "b", "c"])
    assert "ул. Г, 4" in text
    assert "ул. А, 1" in text
    assert "ул. Б, 2" in text
    assert "ул. В, 3" not in text  # четвёртый валидный не берём


async def test_branches_tool_пустые_слаги_или_город(branches_kb: _FakeKB):
    tool = BranchesTool()
    assert await tool.run("x", ConversationContext(city_slug="perm"), slugs=()) == ""
    assert await tool.run("x", ConversationContext(), slugs=["a"]) == ""
    assert branches_kb.calls == []


async def test_city_faq_игнорирует_slugs():
    tool = CityFaqTool()
    ctx = ConversationContext(
        city_faq=[{"question": "Медкомиссия?", "answer": "Нужна к практике."}]
    )
    assert await tool.run("Медкомиссия?", ctx, slugs=["ignored"]) == "Нужна к практике."
