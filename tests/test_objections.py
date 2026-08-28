"""Возражения подбираются по перечню заказчика, а не сочиняются моделью.

В скрипте продаж двадцать два шага и ни одного под отработку возражений.
На прогонах бот отбивал «дорого» и «почему предоплата» из общего промпта:
складно, но неуправляемо — от звонка к звонку доводы менялись, и задать
их заказчик не мог.
"""

from __future__ import annotations

import json

import pytest

from graph.tools_registry import ObjectionsTool, build_context_tools
from script.objections import (
    DEFAULT_FILE,
    Objection,
    format_objection,
    load_objections,
    match_objection,
)


@pytest.fixture()
def objections() -> tuple[Objection, ...]:
    """Перечень возражений из файла рядом со скриптом."""
    return load_objections()


def test_файл_возражений_читается(objections: tuple[Objection, ...]) -> None:
    """Перечень непустой, и у каждого возражения есть приметы и доводы."""
    assert objections
    for item in objections:
        assert item.triggers, item.id
        assert item.arguments, item.id


def test_слаги_возражений_уникальны(objections: tuple[Objection, ...]) -> None:
    """Дубль слага означал бы, что одно возражение недостижимо."""
    slugs = [item.id for item in objections]
    assert len(slugs) == len(set(slugs))


@pytest.mark.parametrize(
    ("reply", "expected"),
    [
        ("Дороговато. В соседней автошколе дешевле.", "price_high"),
        ("А почему я должен вносить предоплату?", "prepayment"),
        ("Мне надо подумать и посоветоваться с женой.", "think_it_over"),
        ("Некогда, я работаю пять через два.", "no_time"),
        ("Это далеко, другой конец города.", "far_away"),
        ("Полгода назад начинал в другой школе и бросил.", "quality_doubt"),
    ],
)
def test_возражение_узнаётся(reply: str, expected: str, objections: tuple[Objection, ...]) -> None:
    """Реплики взяты с живых прогонов."""
    found = match_objection(reply, objections)
    assert found is not None, reply
    assert found.id == expected


def test_обычная_реплика_возражением_не_считается(
    objections: tuple[Objection, ...],
) -> None:
    """Ответ по делу не должен подтягивать доводы."""
    assert match_objection("Механика.", objections) is None
    assert match_objection("Санкт-Петербург.", objections) is None
    assert match_objection("", objections) is None


def test_ё_не_мешает_узнаванию(objections: tuple[Objection, ...]) -> None:
    """«Ещё бы подумать» и «еще бы подумать» — одно и то же."""
    assert match_objection("Ещё надо подумать", objections) is not None
    assert match_objection("Еще надо подумать", objections) is not None


def test_блок_содержит_доводы_и_вопрос() -> None:
    """В контекст уходят доводы и обращение, а не готовая реплика."""
    block = format_objection(
        Objection(
            id="x",
            name="Дорого",
            arguments=("Курс полный.", "Есть рассрочка."),
            ask="Как удобнее платить?",
        )
    )
    assert "Дорого" in block
    assert "- Курс полный." in block
    assert "- Есть рассрочка." in block
    assert "Как удобнее платить?" in block
    assert "не придумывать" in block


def test_отсутствие_файла_не_ломает_бота(tmp_path) -> None:
    """Без перечня бот работает как работал — из общего промпта."""
    assert load_objections(tmp_path / "нет-такого.json") == ()


async def test_инструмент_отдаёт_доводы() -> None:
    """Инструмент возвращает блок по реплике клиента."""
    tool = ObjectionsTool()
    out = await tool.run("Дороговато, в соседней автошколе дешевле", context=None)  # type: ignore[arg-type]
    assert "Возражение" in out
    assert "рассрочка" in out.lower()


async def test_инструмент_молчит_на_обычной_реплике() -> None:
    """Не узнал возражение — пустая строка, лишнего в контекст не летит."""
    tool = ObjectionsTool()
    assert await tool.run("Механика", context=None) == ""  # type: ignore[arg-type]


def test_инструмент_в_реестре(script) -> None:
    """Агент контекстера видит инструмент возражений."""
    names = {tool.name for tool in build_context_tools(script)}
    assert "objections" in names


def test_файл_лежит_рядом_со_скриптом() -> None:
    """Перечень — данные скрипта: правится без выката кода."""
    assert DEFAULT_FILE.exists()
    raw = json.loads(DEFAULT_FILE.read_text(encoding="utf-8"))
    assert raw["objections"]
    assert "выдумано" in raw["note"]
