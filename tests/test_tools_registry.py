"""Тесты реестра инструментов контекстера."""

from __future__ import annotations

from typing import Any

import pytest

from graph.context import ConversationContext
from graph.resolvers import CityResolution
from graph.tools_registry import BranchesTool, CityFaqTool, CityTool, FactsTool
from script.build import build_script
from script.source import JsonScriptSource


class _FakeKB:
    """Заглушка справочника: филиалы без вызова модели."""

    def __init__(
        self,
        *,
        branches: list[dict[str, Any]] | None = None,
        cities: list[dict[str, Any]] | None = None,
        city: dict[str, Any] | None = None,
    ) -> None:
        self._branches = branches or []
        self._cities = cities or []
        self._city = city
        self.calls: list[str] = []
        self.model_calls = 0
        self.collect_needs: list[str] | None = None

    async def list_branches(self, city_slug: str) -> list[dict[str, Any]]:
        self.calls.append(f"list_branches:{city_slug}")
        return list(self._branches)

    async def list_cities(self) -> list[dict[str, Any]]:
        self.calls.append("list_cities")
        return list(self._cities)

    async def get_city(self, city_slug: str) -> dict[str, Any] | None:
        self.calls.append(f"get_city:{city_slug}")
        return self._city

    async def get_branch(self, branch_slug: str) -> dict[str, Any] | None:
        self.calls.append(f"get_branch:{branch_slug}")
        return None


@pytest.fixture()
def branches_kb(monkeypatch) -> _FakeKB:
    kb = _FakeKB(
        branches=[
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


async def test_city_tool_слаг_из_перечня_и_статика(monkeypatch, data_dir, caplog):
    """Инструмент города возвращает слаг из перечня и кладёт статику."""
    kb = _FakeKB(
        cities=[{"slug": "perm", "name": "Пермь"}],
        city={
            "slug": "perm",
            "name": "Пермь",
            "categories": [],
            "vehicles": {},
            "price": {"amount": 10000, "is_from": True},
        },
    )
    monkeypatch.setattr("graph.tools_registry.vector_kb", kb)

    class _Resolver:
        async def resolve(self, text, cities):
            return CityResolution(slug="perm", name="Пермь", is_district=False)

    raw = JsonScriptSource(data_dir).fetch("vector_ru", "2")
    script = build_script(raw)
    tool = CityTool(script, resolver=_Resolver())
    ctx = ConversationContext()
    with caplog.at_level("INFO"):
        text = await tool.run("Пермь", ctx)
    assert "Пермь" in text
    assert ctx.city_slug == "perm"
    assert ctx.city_name == "Пермь"
    assert "Статика разговора" in ctx.static_text
    assert kb.calls == ["list_cities", "get_city:perm"]
    assert any("город найден" in rec.message and "perm" in rec.message for rec in caplog.records)


def _city_script(data_dir):
    """Скрипт для офлайн-тестов CityTool."""
    raw = JsonScriptSource(data_dir).fetch("vector_ru", "2")
    return build_script(raw)


def _spb_kb() -> _FakeKB:
    """Справочник с Санкт-Петербургом."""
    return _FakeKB(
        cities=[{"slug": "spb", "name": "Санкт-Петербург"}],
        city={
            "slug": "spb",
            "name": "Санкт-Петербург",
            "categories": [],
            "vehicles": {},
            "price": {"amount": 20000, "is_from": True},
        },
    )


async def test_city_tool_пустой_query_берёт_реплику(monkeypatch, data_dir):
    """Пустой запрос: реплика целиком уходит в резолвер, слаг в контекст."""
    kb = _spb_kb()
    monkeypatch.setattr("graph.tools_registry.vector_kb", kb)
    seen: list[str] = []

    class _Resolver:
        async def resolve(self, text, cities):
            seen.append(text)
            return CityResolution(slug="spb", name="Санкт-Петербург", is_district=False)

    tool = CityTool(_city_script(data_dir), resolver=_Resolver())
    ctx = ConversationContext()
    text = await tool.run("", ctx, reply="В городе Санкт-Петербург.")
    assert "Санкт-Петербург" in text
    assert ctx.city_slug == "spb"
    assert seen == ["В городе Санкт-Петербург."]


async def test_city_tool_непустой_query_реплику_не_подставляет(monkeypatch, data_dir):
    """Непустой запрос идёт в резолвер как есть; реплика не подставляется."""
    kb = _spb_kb()
    monkeypatch.setattr("graph.tools_registry.vector_kb", kb)
    seen: list[str] = []

    class _Resolver:
        async def resolve(self, text, cities):
            seen.append(text)
            return CityResolution(slug="spb", name="Санкт-Петербург", is_district=False)

    tool = CityTool(_city_script(data_dir), resolver=_Resolver())
    ctx = ConversationContext()
    await tool.run("Пермь", ctx, reply="В городе Санкт-Петербург.")
    assert seen == ["Пермь"]


async def test_city_tool_пустой_query_без_реплики(monkeypatch, data_dir, caplog):
    """Пустой запрос и нет реплики — пустая строка и прежний лог."""
    monkeypatch.setattr("graph.tools_registry.vector_kb", _FakeKB())
    with caplog.at_level("INFO"):
        empty = await CityTool(_city_script(data_dir)).run("", ConversationContext())
    assert empty == ""
    assert any("пустой запрос или город уже зафиксирован" in r.message for r in caplog.records)


async def test_city_tool_город_уже_зафиксирован(monkeypatch, data_dir, caplog):
    """Город в контексте — выход сразу, резолвер не зовём."""
    kb = _spb_kb()
    monkeypatch.setattr("graph.tools_registry.vector_kb", kb)
    called = False

    class _Resolver:
        async def resolve(self, text, cities):
            nonlocal called
            called = True
            return CityResolution(slug="spb", name="Санкт-Петербург", is_district=False)

    tool = CityTool(_city_script(data_dir), resolver=_Resolver())
    with caplog.at_level("INFO"):
        already = await tool.run(
            "Санкт-Петербург",
            ConversationContext(city_slug="spb"),
            reply="В городе Санкт-Петербург.",
        )
    assert already == ""
    assert called is False
    assert kb.calls == []
    assert any("пустой запрос или город уже зафиксирован" in r.message for r in caplog.records)


async def test_city_tool_резолвер_вернул_район(monkeypatch, data_dir, caplog):
    """Район — подсказка уточнить город, слаг не пишем."""
    monkeypatch.setattr(
        "graph.tools_registry.vector_kb",
        _FakeKB(cities=[{"slug": "spb", "name": "Санкт-Петербург"}]),
    )

    class _District:
        async def resolve(self, text, cities):
            return CityResolution(slug=None, name=None, is_district=True)

    tool = CityTool(_city_script(data_dir), resolver=_District())
    ctx = ConversationContext()
    with caplog.at_level("INFO"):
        district = await tool.run("", ctx, reply="Просвещения")
    assert "район" in district.lower()
    assert ctx.city_slug is None
    assert any("резолвер вернул район" in r.message for r in caplog.records)


async def test_city_tool_логирует_причины_отказа(monkeypatch, data_dir, caplog):
    """Каждый отказ CityTool.run пишет свою причину в INFO-лог."""
    script = _city_script(data_dir)

    with caplog.at_level("INFO"):
        empty = await CityTool(script).run("", ConversationContext())
    assert empty == ""
    assert any("пустой запрос или город уже зафиксирован" in r.message for r in caplog.records)

    caplog.clear()
    with caplog.at_level("INFO"):
        already = await CityTool(script).run("Пермь", ConversationContext(city_slug="perm"))
    assert already == ""
    assert any("пустой запрос или город уже зафиксирован" in r.message for r in caplog.records)

    caplog.clear()
    monkeypatch.setattr("graph.tools_registry.vector_kb", _FakeKB(cities=[]))
    with caplog.at_level("INFO"):
        no_cities = await CityTool(script).run("Пермь", ConversationContext())
    assert no_cities == ""
    assert any("пустой перечень городов" in r.message for r in caplog.records)

    class _District:
        async def resolve(self, text, cities):
            return CityResolution(slug=None, name=None, is_district=True)

    caplog.clear()
    monkeypatch.setattr(
        "graph.tools_registry.vector_kb",
        _FakeKB(cities=[{"slug": "perm", "name": "Пермь"}]),
    )
    with caplog.at_level("INFO"):
        district = await CityTool(script, resolver=_District()).run(
            "Просвещения", ConversationContext()
        )
    assert "район" in district.lower()
    assert any("резолвер вернул район" in r.message for r in caplog.records)

    class _Empty:
        async def resolve(self, text, cities):
            return CityResolution(slug=None, name=None, is_district=False)

    caplog.clear()
    with caplog.at_level("INFO"):
        no_slug = await CityTool(script, resolver=_Empty()).run("Москва", ConversationContext())
    assert no_slug == ""
    assert any("не дал слаг или название" in r.message for r in caplog.records)

    class _Ok:
        async def resolve(self, text, cities):
            return CityResolution(slug="perm", name="Пермь", is_district=False)

    caplog.clear()
    monkeypatch.setattr(
        "graph.tools_registry.vector_kb",
        _FakeKB(cities=[{"slug": "perm", "name": "Пермь"}], city=None),
    )
    with caplog.at_level("INFO"):
        no_meta = await CityTool(script, resolver=_Ok()).run("Пермь", ConversationContext())
    assert no_meta == ""
    assert any("не отдал мету города" in r.message for r in caplog.records)


async def test_facts_tool_зовёт_collect_facts(monkeypatch, data_dir):
    """Инструмент фактов зовёт collect_facts с переданными потребностями."""
    seen: dict[str, Any] = {}

    async def fake_collect(kb, *, script, needs, city_slug, branch_slug, want_city_choices):
        seen["needs"] = list(needs)
        seen["city_slug"] = city_slug
        seen["want_city_choices"] = want_city_choices
        return {"price_line": "от 10000"}, [{"call": "get_city"}]

    monkeypatch.setattr("graph.facts.collect_facts", fake_collect)
    monkeypatch.setattr("graph.tools_registry.vector_kb", _FakeKB())

    raw = JsonScriptSource(data_dir).fetch("vector_ru", "2")
    script = build_script(raw)
    tool = FactsTool(script, needs=["price", "city_meta"])
    ctx = ConversationContext(city_slug="perm")
    text = await tool.run("", ctx)
    assert seen["needs"] == ["price", "city_meta"]
    assert seen["city_slug"] == "perm"
    assert "price_line" in text
