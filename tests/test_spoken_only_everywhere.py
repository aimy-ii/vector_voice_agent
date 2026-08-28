"""Запрет просить писать стоит во всех сборках, а не только в полной.

Заказчик потребовал убедиться, что модели везде сказано не просить
человека что-то написать: разговор телефонный. Проверка нашла дыру —
запрет жил только в промпте полного хода. Живая реакция, ожидание и
добивка собираются своими промптами и шли без него вовсе, а это половина
реплик бота за звонок.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from langchain_core.messages import BaseMessage

from graph.prompts import (
    SPOKEN_INTRO,
    build_filler_messages,
    build_pull_messages,
    build_waiting_messages,
)
from script.build import build_script
from script.models import RawSalesScript

DATA = Path(__file__).resolve().parent.parent / "src" / "script" / "data" / "vector_ru.v4.json"


@pytest.fixture()
def sales_script():
    """Скрипт продаж v4 из данных."""
    raw = RawSalesScript.model_validate(json.loads(DATA.read_text(encoding="utf-8")))
    return build_script(raw)


def _body(messages: list[BaseMessage]) -> str:
    """Системная часть промпта одной строкой."""
    return "\n".join(str(getattr(item, "content", "")) for item in messages)


def _forbids_writing(text: str) -> bool:
    """Есть ли в промпте запрет просить писать."""
    lowered = text.lower()
    return "не просить его написать" in lowered or "не может ничего написать" in lowered


def test_полный_ход_запрещает_просить_писать() -> None:
    """В полной сборке запрет живёт подробно, с примерами."""
    assert _forbids_writing(SPOKEN_INTRO)


def test_живая_реакция_запрещает_просить_писать(sales_script) -> None:
    """Короткая реакция тоже уходит клиенту вслух."""
    assert _forbids_writing(
        _body(build_filler_messages(sales_script, messages=[], history_limit=2))
    )


def test_ожидание_запрещает_просить_писать(sales_script) -> None:
    """Заглушка ожидания — такая же реплика в трубке."""
    messages = build_waiting_messages(
        sales_script,
        messages=[],
        profile={},
        pending_fields=[],
        step=None,
        history_limit=4,
    )
    assert _forbids_writing(_body(messages))


def test_добивка_запрещает_просить_писать(sales_script) -> None:
    """Добивка звучит после молчания и правил речи целиком не несёт."""
    messages = build_pull_messages(
        sales_script,
        messages=[],
        profile={},
        pending_fields=[],
        step=None,
    )
    assert _forbids_writing(_body(messages))


def test_запрет_покрывает_выбор_и_после_звонка(sales_script) -> None:
    """Ни «напишите или позвоните», ни «напишите после» не разрешены."""
    short = _body(build_filler_messages(sales_script, messages=[], history_limit=2))
    assert "ни сейчас, ни после" in short
    assert "напишите или позвоните" in short


def test_скрипт_не_предлагает_анкету_на_заполнение() -> None:
    """Исход «оформить дистанционно» есть, но заполняет не человек.

    Требование шага закрытия называло рабочим исходом «дистанционное
    оформление через анкету». Бот честно следовал ему и предлагал:
    «Я отправлю вам анкету в Telegram, вы её заполните» — то есть нарушал
    запрет просить писать, потому что тот противоречил самому скрипту.
    """
    raw = json.loads(DATA.read_text(encoding="utf-8"))
    closing = next(step for step in raw["steps"] if step["id"] == "closing")

    assert "через анкету" not in closing["requirements"]
    assert "вопросы задаёт агент вслух" in closing["requirements"]
    assert "дистанционное оформление" in closing["requirements"], (
        "исход остаётся рабочим, меняется только способ"
    )
