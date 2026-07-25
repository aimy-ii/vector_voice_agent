"""Тесты трёх веток разговора о цене.

Проверка бесшовности: ветка выбирается по полям ответа справочника, а не по
списку городов и не по константам. Третья ветка реализована сейчас, хотя
сегодня не срабатывает ни разу, — здесь она проверяется подменой `reliable`
на `true`. Появятся настоящие цены — агент заговорит иначе без правок кода.
"""

from __future__ import annotations

import pytest

from script.price import format_amount, pick_branch, price_facts, price_line


@pytest.fixture()
def texts(script):
    """Тексты трёх веток из данных скрипта."""
    return script.params.price


def test_ветка_без_суммы(texts):
    assert pick_branch(None) == "no_amount"
    assert pick_branch({}) == "no_amount"
    assert pick_branch({"amount": None, "reliable": False}) == "no_amount"
    assert pick_branch({"amount": 0, "reliable": True}) == "no_amount"


def test_ветка_неподтверждённой_суммы(texts):
    assert pick_branch({"amount": 43900, "reliable": False}) == "unreliable"


def test_ветка_подтверждённой_суммы(texts):
    assert pick_branch({"amount": 47000, "reliable": True}) == "reliable"


def test_без_суммы_число_не_произносится(texts):
    line = price_line({"amount": None, "reliable": False}, texts)
    assert not any(ch.isdigit() for ch in line)
    assert "уточним" in line or "зафиксируем" in line


def test_неподтверждённая_сумма_звучит_как_примерная(texts):
    line = price_line({"amount": 43900, "reliable": False}, texts)
    assert "43900" in line
    assert "от" in line.lower()
    assert "зафиксируем" in line.lower()


def test_подтверждённая_сумма_звучит_как_точная(texts):
    """Тот самый тест бесшовности: меняется только флаг в ответе API."""
    unreliable = price_line({"amount": 47000, "reliable": False}, texts)
    reliable = price_line({"amount": 47000, "reliable": True}, texts)

    assert "47000" in reliable
    assert reliable != unreliable
    assert "от" not in reliable.lower().split("рублей")[0]
    assert "на тысячу" in reliable.lower()


def test_оговорка_справочника_вслух_не_попадает(texts):
    """`note` написан под передачу менеджеру, а бот и есть менеджер."""
    note = "Это цена «от» с сайта, точную стоимость уточнит менеджер"
    line = price_line({"amount": 43900, "reliable": False, "note": note}, texts)
    assert note not in line
    assert "менеджер" not in line.lower()


def test_ни_одна_ветка_не_отправляет_к_менеджеру(texts):
    for price in (None, {"amount": 43900, "reliable": False}, {"amount": 43900, "reliable": True}):
        assert "менеджер" not in price_line(price, texts).lower()


def test_факты_о_цене_говорят_можно_ли_называть_число(texts):
    без_суммы = price_facts(None, texts)
    с_суммой = price_facts({"amount": 43900, "reliable": False}, texts)

    assert без_суммы["branch"] == "no_amount"
    assert без_суммы["may_name_amount"] is False
    assert с_суммой["may_name_amount"] is True
    assert с_суммой["line"] == price_line({"amount": 43900, "reliable": False}, texts)


def test_сумма_форматируется_слитно():
    assert format_amount(44900) == "44900"
    assert format_amount(44900.0) == "44900"
