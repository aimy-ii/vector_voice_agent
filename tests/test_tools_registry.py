"""Тесты реестра инструментов контекстера."""

from __future__ import annotations

from typing import Any

import pytest

from graph.context import ConversationContext
from graph.resolvers import CityResolution
from graph.tools_registry import (
    BranchDetailsTool,
    BranchesTool,
    CityFaqTool,
    CityTool,
    FactsTool,
    NearestBranchesTool,
    build_context_tools,
)
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
        branch: dict[str, Any] | None = None,
        branches_by_slug: dict[str, dict[str, Any]] | None = None,
    ) -> None:
        self._branches = branches or []
        self._cities = cities or []
        self._city = city
        self._branch = branch
        self._branches_by_slug = branches_by_slug or {}
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
        if branch_slug in self._branches_by_slug:
            return self._branches_by_slug[branch_slug]
        return self._branch


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


async def test_branches_tool_пустые_слаги_с_городом_отбирает_сам(branches_kb: _FakeKB):
    """Пустые слаги при городе — отбор по запросу / первые три, не отмена."""
    tool = BranchesTool()
    text = await tool.run("x", ConversationContext(city_slug="perm"), slugs=())
    assert "ул. А, 1" in text
    assert "ул. Б, 2" in text
    assert "ул. В, 3" in text
    assert "ул. Г, 4" not in text
    assert branches_kb.calls == ["list_branches:perm"]


async def test_branches_tool_без_города(branches_kb: _FakeKB):
    """Без города — пустая строка, в справочник не ходим."""
    tool = BranchesTool()
    assert await tool.run("x", ConversationContext(), slugs=["a"]) == ""
    assert branches_kb.calls == []


async def test_branches_tool_чужие_слаги_совпадение_по_запросу(monkeypatch):
    """Слаги не из перечня, запрос совпал с адресом — адреса совпавших."""
    kb = _FakeKB(
        branches=[
            {
                "slug": "spb_prosveshcheniya",
                "address": "Проспект Просвещения, 1",
                "landmark": "метро",
            },
            {"slug": "spb_other", "address": "ул. Другая, 2", "landmark": ""},
            {"slug": "spb_third", "address": "ул. Третья, 3", "landmark": ""},
        ]
    )
    monkeypatch.setattr("graph.tools_registry.vector_kb", kb)
    tool = BranchesTool()
    text = await tool.run(
        "Проспект Просвещения",
        ConversationContext(city_slug="spb"),
        slugs=["prospekt-prosveshcheniya", "krestovskiy-ostrov"],
    )
    assert "Проспект Просвещения, 1" in text
    assert "ул. Другая" not in text
    assert "ул. Третья" not in text


async def test_branches_tool_чужие_слаги_без_совпадения_первые_три(branches_kb: _FakeKB):
    """Слаги не из перечня, запрос ни с чем не совпал — первые три города."""
    tool = BranchesTool()
    text = await tool.run(
        "совсем неизвестное место",
        ConversationContext(city_slug="perm"),
        slugs=["чужой_а", "чужой_б"],
    )
    assert "ул. А, 1" in text
    assert "ул. Б, 2" in text
    assert "ул. В, 3" in text
    assert "ул. Г, 4" not in text


async def test_branches_tool_пустые_слаги_пустой_перечень(monkeypatch):
    """Слаги пустые, перечень города пустой — пустая строка, как раньше."""
    monkeypatch.setattr("graph.tools_registry.vector_kb", _FakeKB(branches=[]))
    tool = BranchesTool()
    assert await tool.run("x", ConversationContext(city_slug="perm"), slugs=()) == ""


async def test_branches_tool_без_поля_адреса(monkeypatch):
    """У филиалов нет поля адреса — пустая строка, как раньше."""
    monkeypatch.setattr(
        "graph.tools_registry.vector_kb",
        _FakeKB(branches=[{"slug": "a", "landmark": "парк"}, {"slug": "b"}]),
    )
    tool = BranchesTool()
    assert await tool.run("", ConversationContext(city_slug="perm"), slugs=["a", "b"]) == ""


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


def _branch_meta(slug: str, *, address: str, landmark: str = "") -> dict[str, Any]:
    """Мета филиала для офлайн-тестов BranchDetailsTool."""
    return {
        "slug": slug,
        "address": address,
        "landmark": landmark,
        "working_hours": "10:00-20:00",
    }


async def test_branch_details_пустой_слаг_берёт_первый_кандидат(monkeypatch, caplog):
    """Без слага в контексте и аргументе — первый из отобранных ранее."""
    meta = _branch_meta("spb_north", address="пр. Просвещения, 1", landmark="метро")
    kb = _FakeKB(
        branches_by_slug={
            "spb_north": meta,
            "spb_other": _branch_meta("spb_other", address="ул. Другая"),
        }
    )
    monkeypatch.setattr("graph.tools_registry.vector_kb", kb)
    ctx = ConversationContext(
        city_slug="spb",
        city_name="Санкт-Петербург",
        static_text="Статика разговора:\nГород: Санкт-Петербург (слаг spb).",
        branch_candidates=["spb_north", "spb_other"],
    )
    with caplog.at_level("INFO"):
        text = await BranchDetailsTool().run("Адрес филиала на севере", ctx, slugs=())
    assert "пр. Просвещения, 1" in text
    assert ctx.branch_slug == "spb_north"
    assert kb.calls == ["get_branch:spb_north"]
    assert any("успех со слагом" in r.message and "spb_north" in r.message for r in caplog.records)


async def test_branch_details_явный_слаг_кандидатов_не_подставляет(monkeypatch, caplog):
    """Явный слаг в аргументе используется; отобранные ранее не берутся."""
    kb = _FakeKB(
        branches_by_slug={
            "spb_explicit": _branch_meta("spb_explicit", address="ул. Явная, 2"),
            "spb_north": _branch_meta("spb_north", address="пр. Просвещения, 1"),
        }
    )
    monkeypatch.setattr("graph.tools_registry.vector_kb", kb)
    ctx = ConversationContext(
        city_slug="spb",
        city_name="Санкт-Петербург",
        static_text="Статика разговора:\nГород: Санкт-Петербург (слаг spb).",
        branch_candidates=["spb_north"],
    )
    with caplog.at_level("INFO"):
        text = await BranchDetailsTool().run("", ctx, slugs=["spb_explicit"])
    assert "ул. Явная, 2" in text
    assert "пр. Просвещения" not in text
    assert ctx.branch_slug == "spb_explicit"
    assert kb.calls == ["get_branch:spb_explicit"]
    assert any(
        "успех со слагом" in r.message and "spb_explicit" in r.message for r in caplog.records
    )


async def test_branch_details_слагов_нет_нигде(monkeypatch, caplog):
    """Слагов нет в контексте, аргументе и кандидатах — пустая строка."""
    monkeypatch.setattr("graph.tools_registry.vector_kb", _FakeKB())
    with caplog.at_level("INFO"):
        empty = await BranchDetailsTool().run("", ConversationContext(), slugs=())
    assert empty == ""
    assert any("слага нет нигде" in r.message for r in caplog.records)


async def test_branch_details_логирует_причины_отказа(monkeypatch, caplog):
    """Каждый выход BranchDetailsTool пишет свою причину в INFO-лог."""
    monkeypatch.setattr("graph.tools_registry.vector_kb", _FakeKB())
    with caplog.at_level("INFO"):
        no_slug = await BranchDetailsTool().run("", ConversationContext(), slugs=())
    assert no_slug == ""
    assert any("слага нет нигде" in r.message for r in caplog.records)

    caplog.clear()
    monkeypatch.setattr("graph.tools_registry.vector_kb", _FakeKB(branch=None))
    with caplog.at_level("INFO"):
        missing = await BranchDetailsTool().run(
            "",
            ConversationContext(branch_slug="ghost"),
            slugs=(),
        )
    assert missing == ""
    assert any("справочник не вернул филиал" in r.message for r in caplog.records)

    caplog.clear()
    meta = _branch_meta("ok", address="ул. Ок, 1")
    monkeypatch.setattr(
        "graph.tools_registry.vector_kb",
        _FakeKB(branches_by_slug={"ok": meta}),
    )
    ctx = ConversationContext(
        city_slug="perm",
        static_text="Статика разговора:\nГород: Пермь (слаг perm).",
        branch_slug="ok",
    )
    with caplog.at_level("INFO"):
        text = await BranchDetailsTool().run("", ctx)
    assert "ул. Ок, 1" in text
    assert any("успех со слагом" in r.message and "ok" in r.message for r in caplog.records)


async def test_branches_tool_сохраняет_кандидатов(branches_kb: _FakeKB):
    """Инструмент филиалов кладёт отобранные слаги в контекст."""
    tool = BranchesTool()
    ctx = ConversationContext(city_slug="perm")
    await tool.run("", ctx, slugs=["b", "a"])
    assert ctx.branch_candidates == ["b", "a"]


class _FakeNearbyKB:
    """Фейковый справочник для инструмента ближайших филиалов."""

    def __init__(
        self,
        *,
        point: tuple[float, float] | None = (58.0, 56.0),
        items: list[dict[str, Any]] | None = None,
    ) -> None:
        self.point = point
        self.items = items if items is not None else []
        self.geocode_calls: list[dict[str, Any]] = []
        self.nearest_calls: list[dict[str, Any]] = []

    async def geocode(
        self, text: str, *, city_slug: str | None = None
    ) -> tuple[float, float] | None:
        self.geocode_calls.append({"text": text, "city_slug": city_slug})
        return self.point

    async def nearest_branches(
        self,
        lat: float,
        lon: float,
        *,
        city_slug: str | None = None,
        limit: int | None = None,
        radius_km: float | None = None,
    ) -> list[dict[str, Any]]:
        self.nearest_calls.append(
            {
                "lat": lat,
                "lon": lon,
                "city_slug": city_slug,
                "limit": limit,
                "radius_km": radius_km,
            }
        )
        return self.items


def test_nearest_branches_есть_в_реестре(data_dir) -> None:
    """Инструмент nearest_branches входит в build_context_tools."""
    script = build_script(JsonScriptSource(data_dir).fetch("vector_ru", "4"))
    names = [t.name for t in build_context_tools(script)]
    assert "nearest_branches" in names


async def test_nearest_branches_новое_место(monkeypatch) -> None:
    """Город известен, место новое: блок, ключ, кандидаты и непустой ответ."""
    items = [
        {"slug": "perm_a", "address": "ул. А, 1", "landmark": "", "distance_km": 0.4},
        {"slug": "perm_b", "address": "ул. Б, 2", "landmark": "", "distance_km": 1.0},
    ]
    kb = _FakeNearbyKB(items=items)
    monkeypatch.setattr("graph.tools_registry.vector_kb", kb)
    tool = NearestBranchesTool()
    ctx = ConversationContext(city_slug="perm")
    text = await tool.run("Солнечный", ctx)
    assert text
    assert "ул. А, 1" in ctx.nearby_text
    assert "ул. Б, 2" in ctx.nearby_text
    assert ctx.nearby_key == "perm:солнечный"
    assert ctx.nearby_found is True
    assert ctx.branch_candidates == ["perm_a", "perm_b"]
    assert len(kb.geocode_calls) == 1


async def test_nearest_branches_повтор_того_же_места(monkeypatch) -> None:
    """Повтор с тем же ключом: в справочник не ходят, контекст не меняется."""
    items = [
        {"slug": "perm_a", "address": "ул. А, 1", "landmark": "", "distance_km": 0.4},
    ]
    kb = _FakeNearbyKB(items=items)
    monkeypatch.setattr("graph.tools_registry.vector_kb", kb)
    tool = NearestBranchesTool()
    ctx = ConversationContext(city_slug="perm")
    first = await tool.run("Солнечный", ctx)
    assert first
    snapshot = (ctx.nearby_text, ctx.nearby_key, ctx.nearby_found, list(ctx.branch_candidates))
    calls_after_first = len(kb.geocode_calls)
    second = await tool.run("Солнечный", ctx)
    assert second
    assert "уже сделан" in second
    assert len(kb.geocode_calls) == calls_after_first
    assert (
        ctx.nearby_text,
        ctx.nearby_key,
        ctx.nearby_found,
        list(ctx.branch_candidates),
    ) == snapshot


async def test_nearest_branches_без_города(monkeypatch) -> None:
    """Без города — пустой ответ, поля пусты, в справочник не ходили."""
    kb = _FakeNearbyKB(items=[{"slug": "x", "address": "ул. X", "distance_km": 1}])
    monkeypatch.setattr("graph.tools_registry.vector_kb", kb)
    tool = NearestBranchesTool()
    ctx = ConversationContext()
    text = await tool.run("Солнечный", ctx)
    assert text == ""
    assert ctx.nearby_text == ""
    assert ctx.nearby_key == ""
    assert ctx.nearby_found is False
    assert kb.geocode_calls == []


async def test_nearest_branches_место_не_опознано(monkeypatch) -> None:
    """Geocode вернул None: пустой ответ, nearby_text с «не удалось», кандидаты не тронуты."""
    kb = _FakeNearbyKB(point=None)
    monkeypatch.setattr("graph.tools_registry.vector_kb", kb)
    tool = NearestBranchesTool()
    ctx = ConversationContext(city_slug="perm", branch_candidates=["keep_me"])
    text = await tool.run("Неизвестно", ctx)
    assert text == ""
    assert "не удалось" in ctx.nearby_text
    assert ctx.nearby_found is False
    assert ctx.branch_candidates == ["keep_me"]
    assert len(kb.geocode_calls) == 1
    assert kb.nearest_calls == []


async def test_nearest_branches_неудача_не_затирает_удачу(monkeypatch) -> None:
    """Неудачный подбор по новому месту не затирает уже найденные адреса."""
    items = [
        {"slug": "perm_a", "address": "ул. А, 1", "landmark": "", "distance_km": 0.4},
    ]
    kb = _FakeNearbyKB(items=items)
    monkeypatch.setattr("graph.tools_registry.vector_kb", kb)
    tool = NearestBranchesTool()
    ctx = ConversationContext(city_slug="perm")
    first = await tool.run("Солнечный", ctx)
    assert first
    kept = ctx.nearby_text
    kb.point = None
    text = await tool.run("Неизвестно", ctx)
    assert text == ""
    assert ctx.nearby_text == kept
    assert ctx.nearby_found is True
    assert ctx.branch_candidates == ["perm_a"]
    assert ctx.nearby_key == "perm:неизвестно"
