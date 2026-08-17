"""Офлайн-тесты подбора ближайших филиалов."""

from __future__ import annotations

from typing import Any

import pytest

from graph.nearby import (
    NearbyResult,
    apply_result,
    format_found,
    format_missing,
    format_searching,
    has_place_name,
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
    assert normalize_place("у  Гражданского   проспекта") == "гражданского проспекта"
    assert normalize_place("Зелёный") == "зеленый"
    assert normalize_place("Ёлки") == "елки"


def test_normalize_place_ведущие_предлоги_один_ключ() -> None:
    """Ведущие предлоги не меняют ключ: «у метро N» и «метро N» — одно место."""
    assert nearby_key("spb", "у метро Проспект Просвещения") == nearby_key(
        "spb", "метро Проспект Просвещения"
    )
    assert nearby_key("spb", "около Славы") == nearby_key("spb", "Славы")
    assert nearby_key("spb", "рядом с Ушинского") == nearby_key("spb", "Ушинского")
    assert nearby_key("spb", "в районе Солнечного") == nearby_key("spb", "Солнечного")


def test_normalize_place_только_предлог_пустой_ключ() -> None:
    """Место из одного предлога после нормализации пустое — ключа нет."""
    assert normalize_place("у") == ""
    assert nearby_key("spb", "у") == ""
    assert nearby_key("spb", "около") == ""


def test_normalize_place_на_не_съедает_весь_адрес() -> None:
    """«на Чернышевского» и «Чернышевского» — один ключ; адрес не обнуляется."""
    assert nearby_key("spb", "на Чернышевского") == nearby_key("spb", "Чернышевского")
    assert nearby_key("spb", "Чернышевского") == "spb:чернышевского"
    assert normalize_place("Чернышевского") == "чернышевского"


def test_has_place_name_общие_слова_без_названия() -> None:
    """Общие слова и пустая строка — не ориентир: искать нечего."""
    assert has_place_name("Метро") is False
    assert has_place_name("станция метро") is False
    assert has_place_name("север города") is False
    assert has_place_name("в центре города") is False
    assert has_place_name("") is False


def test_has_place_name_есть_название() -> None:
    """Хоть одно слово-название — ориентир пригоден для поиска."""
    assert has_place_name("метро Проспект Просвещения") is True
    assert has_place_name("Проспект Просвещения") is True
    assert has_place_name("Коломяжский проспект 15") is True
    assert has_place_name("север города, проспект Просвещения") is True


async def test_lookup_nearby_без_названия_не_ходит_в_геокодер() -> None:
    """Место без названия: found=False, геокодер не вызывается."""
    kb = FakeNearbyKB(point=(58.0, 56.0), items=[{"slug": "x"}])
    result = await lookup_nearby(kb, city_slug="perm", place="Метро", key="perm:метро")

    assert kb.geocode_calls == []
    assert kb.nearest_calls == []
    assert result.found is False
    assert result.text == format_missing("Метро")
    assert result.key == "perm:метро"


def test_apply_result_удача_вытесняет_прежнее() -> None:
    """Удачный подбор всегда кладёт свой текст и признак."""
    result = NearbyResult(key="k", text="найдено", found=True)
    text, found = apply_result(
        previous_text="старое",
        previous_found=False,
        result=result,
    )
    assert text == "найдено"
    assert found is True


def test_apply_result_неудача_не_затирает_удачу() -> None:
    """Неудачный подбор при уже найденных адресах оставляет прежний блок."""
    result = NearbyResult(key="k", text=format_missing("X"), found=False)
    text, found = apply_result(
        previous_text="три филиала",
        previous_found=True,
        result=result,
    )
    assert text == "три филиала"
    assert found is True


def test_apply_result_неудача_без_прежней_удачи() -> None:
    """Неудачный подбор без прежнего успеха кладёт текст неудачи."""
    missing = format_missing("X")
    result = NearbyResult(key="k", text=missing, found=False)
    text, found = apply_result(
        previous_text="",
        previous_found=False,
        result=result,
    )
    assert text == missing
    assert found is False


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


def test_блок_филиалов_разрешает_назвать_второй() -> None:
    """При сомнении человека — назвать следующий филиал с расстоянием."""
    items = [
        {"slug": "a", "address": "ул. А, 1", "landmark": "", "distance_km": 0.5},
        {"slug": "b", "address": "ул. Б, 2", "landmark": "", "distance_km": 1.0},
        {"slug": "c", "address": "ул. В, 3", "landmark": "", "distance_km": 1.5},
    ]
    text = format_found("метро X", items)
    assert "назвать следующий по списку вместе с его расстоянием" in text


def test_блок_филиалов_не_утверждает_что_ближе_нет() -> None:
    """Блок не утверждает, что в городе ближе ничего нет."""
    items = [
        {"slug": "a", "address": "ул. А, 1", "landmark": "", "distance_km": 0.5},
        {"slug": "b", "address": "ул. Б, 2", "landmark": "", "distance_km": 1.0},
        {"slug": "c", "address": "ул. В, 3", "landmark": "", "distance_km": 1.5},
    ]
    text = format_found("метро X", items).lower()
    assert "в городе нет" not in text
    assert "по всему городу" not in text


def test_блок_филиалов_сохраняет_порядок_и_адреса() -> None:
    """Адреса идут в исходном порядке, нумерация с единицы, расстояния на месте."""
    items = [
        {"slug": "a", "address": "ул. А, 1", "landmark": "Ориентир А", "distance_km": 0.5},
        {"slug": "b", "address": "ул. Б, 2", "landmark": "", "distance_km": 1.0},
        {"slug": "c", "address": "ул. В, 3", "landmark": "Ориентир В", "distance_km": 2.25},
    ]
    text = format_found("метро X", items)
    assert text.index("ул. А, 1") < text.index("ул. Б, 2") < text.index("ул. В, 3")
    assert "1." in text.split("\n")[1]
    assert "2." in text.split("\n")[2]
    assert "3." in text.split("\n")[3]
    assert "0,5" in text
    assert "1,0" in text
    assert "2,2" in text
    assert "(Ориентир А)" in text
    assert "(Ориентир В)" in text


def test_блок_филиалов_с_одним_филиалом() -> None:
    """На списке из одного филиала format_found не падает."""
    text = format_found("место", [{"slug": "a", "address": "ул. А, 1", "distance_km": 0.3}])
    assert "ул. А, 1" in text
    assert "0,3" in text
    assert "назвать следующий по списку вместе с его расстоянием" in text


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


def test_format_found_печатает_часы_и_перерыв() -> None:
    """Блок ближайших печатает часы работы и перерыв, когда справочник их прислал."""
    items = [
        {
            "slug": "a",
            "address": "ул. Чернышевского, 4",
            "landmark": "ТРК Нарва",
            "distance_km": 0.4,
            "working_hours": "09:00–21:00",
            "break_time": "13:00–14:00",
        },
        {
            "slug": "b",
            "address": "пр. Ленина, 10",
            "landmark": "",
            "distance_km": 1.2,
            "working_hours": "10:00–19:00",
            "break_time": "",
        },
    ]
    text = format_found("Солнечный", items)
    lines = text.split("\n")
    assert lines[1] == (
        "1. ул. Чернышевского, 4 — 0,4 км (ТРК Нарва), работает 09:00–21:00, перерыв 13:00–14:00"
    )
    assert lines[2] == "2. пр. Ленина, 10 — 1,2 км, работает 10:00–19:00"


def test_format_found_без_часов_строка_как_раньше() -> None:
    """Без часов в ответе справочника строка филиала совпадает со старым форматом."""
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
    lines = text.split("\n")
    assert lines[1] == "1. ул. Чернышевского, 4 — 0,4 км (ТРК Нарва)"
    assert lines[2] == "2. пр. Ленина, 10 — 1,2 км"


async def test_lookup_nearby_несёт_филиалы_целиком() -> None:
    """Результат несёт филиалы целиком; слаги те же и в том же порядке."""
    items = [
        {
            "slug": "perm_a",
            "address": "ул. А, 1",
            "landmark": "Дом быта",
            "district": "Солнечный",
            "place_type": "филиал",
            "status": "открыт",
            "working_hours": "09:00–21:00",
            "break_time": "13:00–14:00",
            "phone": "+7 342 000-00-01",
            "distance_km": 0.4,
        },
        {
            "slug": "perm_b",
            "address": "ул. Б, 2",
            "landmark": "",
            "distance_km": 1.0,
        },
    ]
    kb = FakeNearbyKB(point=(58.01, 56.25), items=items)
    result = await lookup_nearby(kb, city_slug="perm", place="Солнечный", key="perm:солнечный")

    assert result.found is True
    assert result.branch_slugs == ["perm_a", "perm_b"]
    assert result.branch_cards == items
    assert [card.get("slug") for card in result.branch_cards] == result.branch_slugs
