"""Тесты форматтеров служебных логов — без вывода в лог."""

from __future__ import annotations

from graph.log_fmt import (
    format_check_done,
    format_lookup_done,
    format_plan_done,
    format_spoken_preview,
)


def test_format_check_done_с_основаниями():
    assert format_check_done([]) == "ничего не закрылось"
    assert (
        format_check_done([("theory_format", "диалог"), ("terms", "счётчик")])
        == "закрыт theory_format (диалог), закрыт terms (счётчик)"
    )
    assert format_check_done([("terms", "доставка")]) == "закрыт terms (доставка)"


def test_format_plan_done_со_счётчиками():
    text = format_plan_done(
        step_id="terms",
        route="lookup",
        head=[("terms", 1), ("theory_format", 0)],
        city_slug="spb",
        branch_slug=None,
    )
    assert "шаг terms" in text
    assert "маршрут lookup" in text
    assert "шапка [terms(1), theory_format(0)]" in text
    assert "город=spb" in text
    assert "филиал=—" in text


def test_format_spoken_preview_обрезка():
    short = "Расскажу, как всё устроено."
    assert format_spoken_preview(short) == short
    long = "А" * 80
    preview = format_spoken_preview(long, limit=60)
    assert len(preview) == 60
    assert preview.endswith("…")
    assert preview.startswith("А" * 59)


def test_format_lookup_done_за_ход():
    assert format_lookup_done([]) == "обращений к справочнику нет"
    assert (
        format_lookup_done(
            [
                {"call": "list_cities", "found": 3},
                {"call": "resolve_city", "slug": "perm"},
            ]
        )
        == "2 вызова: list_cities, resolve_city"
    )
