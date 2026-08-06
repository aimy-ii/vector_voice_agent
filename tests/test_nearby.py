"""Офлайн-тесты подбора ближайших филиалов."""

from __future__ import annotations

from typing import Any

import pytest

from graph.nearby import (
    format_found,
    format_missing,
    format_searching,
    is_searching,
    lookup_nearby,
    nearby_key,
    normalize_place,
    should_refresh,
)


class FakeNearbyKB:
    """Фейковый справочник: считает вызовы и запоминает аргументы."""

    def __init__(
        self,
        *,
        point: tuple[float, float] | None = (59.96, 30.33),
        items: list[dict[str, Any]] | None = None,
        geocode_error: Exception | None = None,
        nearest_error: Exception | None = None,
    ) -> None:
        self.point = point
        self.items = items if items is not None else []
        self.geocode_error = geocode_error
        self.nearest_error = nearest_error
        self.geocode_calls: list[dict[str, Any]] = []
        self.nearest_calls: list[dict[str, Any]] = []

    async def geocode(
        self, text: str, *, city_slug: str | None = None
    ) -> tuple[float, float] | None:
        """Запоминает аргументы и отдаёт заготовленную точку."""
        self.geocode_calls.append({"text": text, "city_slug": city_slug})
        if self.geocode_error is not None:
            raise self.geocode_error
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
        """Запоминает аргументы и отдаёт заготовленный список."""
        self.nearest_calls.append(
            {
                "lat": lat,
                "lon": lon,
                "city_slug": city_slug,
                "limit": limit,
                "radius_km": radius_km,
            }
        )
        if self.nearest_error is not None:
            raise self.nearest_error
        return self.items


def test_normalize_place_регистр_пробелы_и_ё() -> None:
    """Регистр, пробелы и «ё» на нормализацию не влияют."""
    assert normalize_place("  Солнечный  ") == "солнечный"
    assert normalize_place("у  Гражданского   проспекта") == "у гражданского проспекта"
    assert normalize_place("Зелёный") == "зеленый"
    assert normalize_place("Ёлки") == "елки"


def test_nearby_key_разные_города_и_пустое_место() -> None:
    """Одинаковое место в разных городах даёт разные ключи; пустое — пустой ключ."""
    assert nearby_key("perm", "Центральный") != nearby_key("krasnoyarsk", "Центральный")
    assert nearby_key("perm", "Центральный") == "perm:центральный"
    assert nearby_key("perm", "  ") == ""
    assert nearby_key(None, "Солнечный") == "-:солнечный"


def test_should_refresh_новое_место() -> None:
    """Пересчёт нужен, когда место новое."""
    assert should_refresh(city_slug="perm", place="Солнечный", current_key="") == "perm:солнечный"


def test_should_refresh_тот_же_ключ() -> None:
    """Повтор того же места пересчитывать не надо."""
    key = nearby_key("perm", "Солнечный")
    assert should_refresh(city_slug="perm", place="солнечный", current_key=key) is None


def test_should_refresh_без_города() -> None:
    """Без города подбор не запускаем."""
    assert should_refresh(city_slug=None, place="Солнечный", current_key="") is None
    assert should_refresh(city_slug="", place="Солнечный", current_key="") is None


def test_should_refresh_пустое_место() -> None:
    """Пустое место пересчёта не требует."""
    assert should_refresh(city_slug="perm", place="  ", current_key="") is None


def test_should_refresh_выключен(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """При выключенном гео-подборе пересчёт не нужен."""
    monkeypatch.setattr("graph.nearby.settings.nearest_branches_enabled", False)
    assert should_refresh(city_slug="perm", place="Солнечный", current_key="") is None


def test_format_found_адреса_расстояния_ориентиры_порядок() -> None:
    """Блок содержит адреса, расстояние с запятой, ориентир и правила."""
    items = [
        {
            "slug": "a",
            "address": "ул. Чернышевского, 4",
            "landmark": "ТРК Нарва",
            "distance_km": 0.4,
        },
        {
            "slug": "b",
            "address": "пр. Ленина, 10",
            "landmark": "",
            "distance_km": 1.2,
        },
    ]
    text = format_found("Солнечный", items)
    assert "ул. Чернышевского, 4" in text
    assert "пр. Ленина, 10" in text
    assert "0,4" in text
    assert "(ТРК Нарва)" in text
    assert text.index("ул. Чернышевского, 4") < text.index("пр. Ленина, 10")
    assert "(ТРК Нарва)" in text.split("\n")[1]
    assert "ТРК Нарва" not in text.split("\n")[2]
    assert "Первый — ближайший" in text
    assert "Называть улицу и дом" in text


def test_format_missing_и_searching_содержат_место() -> None:
    """Формулировки missing и searching называют место."""
    assert "Солнечный" in format_missing("Солнечный")
    assert "Солнечный" in format_searching("Солнечный")


def test_is_searching_только_на_незавершённом_подборе() -> None:
    """Признак висит только на format_searching, не на итогах и не на пустом."""
    assert is_searching(format_searching("Солнечный")) is True
    assert (
        is_searching(format_found("Солнечный", [{"address": "ул. А", "distance_km": 1}])) is False
    )
    assert is_searching(format_missing("Солнечный")) is False
    assert is_searching("") is False


async def test_lookup_nearby_успех() -> None:
    """Успех: два вызова, координаты и город уходят верно, текст как format_found."""
    items = [
        {
            "slug": "perm_a",
            "address": "ул. А, 1",
            "landmark": "",
            "distance_km": 0.4,
        },
        {
            "slug": "perm_b",
            "address": "ул. Б, 2",
            "landmark": "Дом",
            "distance_km": 1.0,
        },
    ]
    kb = FakeNearbyKB(point=(58.01, 56.25), items=items)
    key = "perm:солнечный"
    result = await lookup_nearby(kb, city_slug="perm", place="Солнечный", key=key)

    assert len(kb.geocode_calls) == 1
    assert kb.geocode_calls[0] == {"text": "Солнечный", "city_slug": "perm"}
    assert len(kb.nearest_calls) == 1
    assert kb.nearest_calls[0]["lat"] == 58.01
    assert kb.nearest_calls[0]["lon"] == 56.25
    assert kb.nearest_calls[0]["city_slug"] == "perm"
    assert result.found is True
    assert result.branch_slugs == ["perm_a", "perm_b"]
    assert result.text == format_found("Солнечный", items)
    assert result.key == key


async def test_lookup_nearby_место_не_опознано() -> None:
    """Без координат второго вызова нет, текст — format_missing."""
    kb = FakeNearbyKB(point=None, items=[{"slug": "x"}])
    result = await lookup_nearby(kb, city_slug="perm", place="Неизвестно", key="perm:неизвестно")

    assert len(kb.geocode_calls) == 1
    assert kb.nearest_calls == []
    assert result.found is False
    assert result.text == format_missing("Неизвестно")


async def test_lookup_nearby_филиалов_нет() -> None:
    """Пустой список ближайших: found=False, слаги пусты."""
    kb = FakeNearbyKB(point=(58.0, 56.0), items=[])
    result = await lookup_nearby(kb, city_slug="perm", place="Солнечный", key="perm:солнечный")

    assert len(kb.nearest_calls) == 1
    assert result.found is False
    assert result.branch_slugs == []
    assert result.text == format_missing("Солнечный")


async def test_lookup_nearby_исключение_в_geocode() -> None:
    """Исключение геокодера наружу не летит."""
    kb = FakeNearbyKB(geocode_error=RuntimeError("geocode down"))
    result = await lookup_nearby(kb, city_slug="perm", place="Солнечный", key="perm:солнечный")

    assert result.found is False
    assert kb.nearest_calls == []


async def test_lookup_nearby_исключение_в_nearest() -> None:
    """Исключение nearest наружу не летит."""
    kb = FakeNearbyKB(point=(58.0, 56.0), nearest_error=RuntimeError("nearest down"))
    result = await lookup_nearby(kb, city_slug="perm", place="Солнечный", key="perm:солнечный")

    assert result.found is False
    assert result.text == format_missing("Солнечный")


async def test_lookup_nearby_сохраняет_переданный_ключ() -> None:
    """В результат кладётся тот же key, что передали."""
    kb = FakeNearbyKB(point=None)
    key = "custom:key"
    result = await lookup_nearby(kb, city_slug="perm", place="X", key=key)
    assert result.key == key
