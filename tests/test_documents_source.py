"""Возражения и варианты полей из админки, с откатом на файл образа."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from script.documents import load_document
from script.field_choices import load_field_choices
from script.objections import load_objections


@dataclass
class ОтветЗаглушка:
    """Минимальный ответ httpx: статус и тело."""

    status_code: int
    payload: Any = None

    def json(self) -> Any:
        """Отдаёт тело ответа."""
        return self.payload


@pytest.fixture
def файл_возражений(data_dir: Path) -> dict[str, Any]:
    """Перечень возражений из образа."""
    return json.loads((data_dir / "objections_ru.json").read_text(encoding="utf-8"))


def _http(monkeypatch, ответ) -> None:
    """Подменяет запрос к справочнику заданным ответом."""

    def подмена(url: str, timeout=None):  # noqa: ANN001, ARG001
        if isinstance(ответ, Exception):
            raise ответ
        return ответ

    monkeypatch.setattr("script.documents.httpx.get", подмена)
    monkeypatch.setattr("script.documents.settings.script_source", "http")


def test_по_умолчанию_читаем_файл(monkeypatch, data_dir):
    """Пока источник ``json``, по сети не ходим вовсе."""

    def запрещено(url: str, timeout=None):  # noqa: ANN001, ARG001
        raise AssertionError("при SCRIPT_SOURCE=json запросов быть не должно")

    monkeypatch.setattr("script.documents.httpx.get", запрещено)
    monkeypatch.setattr("script.documents.settings.script_source", "json")
    документ = load_document("objections", data_dir / "objections_ru.json")
    assert документ is not None
    assert документ["id"] == "vector_ru"


def test_возражения_приезжают_из_админки(monkeypatch, data_dir, файл_возражений):
    """При источнике ``http`` перечень берётся из ответа справочника."""
    правленый = dict(файл_возражений)
    правленый["objections"] = [
        {
            "id": "новое",
            "name": "Возражение из админки",
            "triggers": ["примета"],
            "arguments": ["довод"],
            "ask": "вопрос",
        }
    ]
    _http(monkeypatch, ОтветЗаглушка(200, правленый))

    перечень = load_objections(data_dir / "objections_ru.json")
    assert [item.id for item in перечень] == ["новое"]


def test_админка_молчит_возражения_из_файла(monkeypatch, data_dir):
    """Недоступная админка возвращает перечень к тому, что в образе."""
    import httpx

    _http(monkeypatch, httpx.ConnectError("нет связи"))
    из_файла = json.loads((data_dir / "objections_ru.json").read_text(encoding="utf-8"))
    перечень = load_objections(data_dir / "objections_ru.json")
    assert len(перечень) == len(из_файла["objections"])


def test_админка_ответила_ошибкой_варианты_из_файла(monkeypatch, data_dir):
    """HTTP 503 — не повод остаться без вариантов полей анкеты."""
    _http(monkeypatch, ОтветЗаглушка(503))
    поля = load_field_choices(data_dir / "field_choices_ru.json")
    assert "theory_format" in поля


def test_варианты_приезжают_из_админки(monkeypatch, data_dir):
    """Варианты полей берутся из ответа справочника вместе с порядком."""
    _http(
        monkeypatch,
        ОтветЗаглушка(
            200,
            {
                "id": "vector_ru",
                "version": "2",
                "note": "",
                "fields": {
                    "theory_format": [
                        {"value": "Очно", "triggers": ["очно"]},
                        {"value": "Дистанционно", "triggers": ["онлайн"]},
                    ]
                },
            },
        ),
    )
    поля = load_field_choices(data_dir / "field_choices_ru.json")
    assert [choice.value for choice in поля["theory_format"]] == ["Очно", "Дистанционно"]
