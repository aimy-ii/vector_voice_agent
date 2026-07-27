"""Тесты форматтеров служебных логов — без вывода в лог."""

from __future__ import annotations

from graph.log_fmt import (
    format_check_done,
    format_check_pending,
    format_live_check_state,
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
        call_id="call-42",
    )
    assert "шаг terms" in text
    assert "маршрут lookup" in text
    assert "шапка [terms(1), theory_format(0)]" in text
    assert "звонок call-42" in text
    assert "город=spb" in text
    assert "филиал=—" in text


def test_format_plan_done_без_call_id():
    text = format_plan_done(
        step_id=None,
        route="respond",
        head=[],
        city_slug=None,
        branch_slug=None,
    )
    assert "звонок —" in text
    assert "шапка []" in text


def test_format_live_check_state_пустой_прогресс():
    text = format_live_check_state(attempts={}, status={}, profile={})
    assert text == "счётчики {}, статусы {}, профиль: —"


def test_format_live_check_state_только_ключи_профиля():
    text = format_live_check_state(
        attempts={"city": 1},
        status={"city": "pending"},
        profile={"city": "Санкт-Петербург", "caller_name": "Андрей", "empty": ""},
    )
    assert "счётчики {'city': 1}" in text
    assert "статусы {'city': 'pending'}" in text
    assert "профиль: caller_name, city" in text
    assert "Санкт-Петербург" not in text
    assert "Андрей" not in text


def test_format_check_pending_с_висящими():
    text = format_check_pending(
        pending=[("city", 1)],
        rejected=[("greeting", "исчерпан"), ("branch", "счётчик ноль")],
    )
    assert text == ("на проверку: [city(1)]; отсеяно: greeting — исчерпан, branch — счётчик ноль")


def test_format_check_pending_пусто_с_доступными():
    text = format_check_pending(
        pending=[],
        rejected=[("city", "счётчик ноль")],
        available=[("city", 0), ("branch", 0)],
    )
    assert "на проверку: пусто" in text
    assert "отсеяно: city — счётчик ноль" in text
    assert "доступны: [city(0), branch(0)]" in text


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
