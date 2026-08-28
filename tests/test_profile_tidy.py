"""Значения анкеты приводятся к виду записи, а не остаются репликой.

Заказчик потребовал этого по итогам прогонов: в анкету попадало «В
Санкт-Петербурге» и «у метро Проспект Просвещения». Для чтения терпимо,
для выгрузки в работу — нет: одно и то же место записано по-разному, и
поля не сравнить между звонками.
"""

from __future__ import annotations

import pytest

from graph.profile_tidy import tidy_value


@pytest.mark.parametrize(
    ("key", "spoken", "expected"),
    [
        ("location_hint", "у метро Проспект Просвещения", "Метро Проспект Просвещения"),
        ("location_hint", "рядом с Пионерской", "Пионерской"),
        ("city", "В Санкт-Петербурге", "Санкт-Петербурге"),
        ("city", "это Санкт-Петербург", "Санкт-Петербург"),
        ("transmission", "Ну, механика.", "Механика"),
        ("transmission", "механика", "Механика"),
        ("appointment_time", "  вечером   после семи  ", "Вечером после семи"),
        ("experience", "Прав никогда не было.", "Прав никогда не было"),
    ],
)
def test_значение_приводится(key: str, spoken: str, expected: str) -> None:
    """Пробелы, крайние знаки, вводные слова и предлоги места снимаются."""
    assert tidy_value(key, spoken) == expected


@pytest.mark.parametrize(
    ("key", "spoken"),
    [
        ("theory_format", "В классе"),
        ("payment_pref", "В рассрочку"),
        ("outcome", "В офис приедет в субботу"),
    ],
)
def test_предлог_вне_полей_места_не_трогаем(key: str, spoken: str) -> None:
    """«В классе» — ответ про формат, «Классе» вместо него было бы порчей."""
    assert tidy_value(key, spoken) == spoken


def test_пустое_остаётся_пустым() -> None:
    """Из пробелов и знаков значения не выходит."""
    assert tidy_value("city", "   ") == ""
    assert tidy_value("city", " ... ") == ""
    assert tidy_value("location_hint", "у") == ""


def test_середина_значения_не_меняется() -> None:
    """Приведение снимает края, а слова внутри не переставляет."""
    spoken = "Коломяжский проспект, дом 15, корпус 2"
    assert tidy_value("location_hint", spoken) == spoken
