"""Тесты резолверов города и филиала."""

from __future__ import annotations

from graph.resolvers import (
    BranchResolution,
    CityResolution,
    exact_city_match,
    resolve_branch,
    resolve_city,
)


class FakeCityResolver:
    """Заглушка резолвера города."""

    def __init__(self, result: CityResolution) -> None:
        self.result = result
        self.calls = 0

    async def resolve(self, text, cities):
        self.calls += 1
        return self.result


class FakeBranchResolver:
    """Заглушка резолвера филиала."""

    def __init__(self, result: BranchResolution) -> None:
        self.result = result
        self.calls = 0

    async def resolve(self, text, branches):
        self.calls += 1
        return self.result


async def test_точное_совпадение_обходится_без_вызова(fake_kb):
    cities = await fake_kb.list_cities()
    resolver = FakeCityResolver(CityResolution(slug="hack", name="Hack"))
    result = await resolve_city("Пермь", cities, resolver=resolver)
    assert result.slug == "perm"
    assert result.name == "Пермь"
    assert resolver.calls == 0


async def test_слаг_вне_перечня_отвергается(fake_kb):
    cities = await fake_kb.list_cities()
    resolver = FakeCityResolver(CityResolution(slug="moscow", name="Москва"))
    result = await resolve_city("что-то", cities, resolver=resolver)
    assert result.slug is None


async def test_район_вместо_города_не_записывается(fake_kb):
    cities = await fake_kb.list_cities()
    resolver = FakeCityResolver(CityResolution(is_district=True))
    result = await resolve_city("Красное Село", cities, resolver=resolver)
    assert result.is_district is True
    assert result.slug is None


async def test_отбор_филиалов_не_больше_трёх_и_из_списка(fake_kb):
    branches = await fake_kb.list_branches("perm")
    resolver = FakeBranchResolver(
        BranchResolution(
            slugs=[
                "perm_chernyshevskogo",
                "perm_ekaterininskaya",
                "perm_lenina",
                "perm_mira",
                "fake",
            ],
            selected=None,
        )
    )
    result = await resolve_branch("центр", branches, resolver=resolver)
    assert len(result.slugs) <= 3
    known = {b["slug"] for b in branches}
    assert all(s in known for s in result.slugs)
    assert "fake" not in result.slugs


def test_exact_city_match():
    cities = [{"slug": "perm", "name": "Пермь"}]
    assert exact_city_match("пермь", cities).slug == "perm"
    assert exact_city_match("нет", cities) is None
