"""Тесты словаря ситуативных заглушек."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from graph.situations import DEFAULT_SLUG, TEMPLATES_KEY, load_situations, pick_filler


def test_словарь_имеет_templates_и_default():
    catalog = load_situations()
    assert DEFAULT_SLUG in catalog
    assert TEMPLATES_KEY in catalog
    assert len(catalog[DEFAULT_SLUG]) >= 5
    assert len(catalog[TEMPLATES_KEY]) >= 5
    assert all("{subject}" in t for t in catalog[TEMPLATES_KEY])


def test_pick_filler_подстановка_предмета():
    phrase = pick_filler("филиалы")
    assert phrase
    assert "филиалы" in phrase


def test_pick_filler_пустой_предмет_берёт_default():
    catalog = load_situations()
    phrase = pick_filler(None)
    assert phrase in catalog[DEFAULT_SLUG]
    assert pick_filler("") in catalog[DEFAULT_SLUG]
    assert pick_filler("   ") in catalog[DEFAULT_SLUG]


def test_pick_filler_мимо_уже_звучавших():
    catalog = load_situations()
    subject = "медкомиссия"
    pool = [t.format(subject=subject) for t in catalog[TEMPLATES_KEY]]
    spoken = pool[:-1]
    phrase = pick_filler(subject, spoken=spoken)
    assert phrase == pool[-1]


def test_pick_filler_никогда_не_пустая():
    for subject in (None, "", "филиалы", "пересдача", "медкомиссия"):
        assert pick_filler(subject).strip()


def test_отсутствие_обязательного_ключа_ошибка_загрузки(tmp_path: Path, monkeypatch):
    bad = tmp_path / "situations.json"
    bad.write_text(json.dumps({"default": ["так… секунду"]}), encoding="utf-8")
    monkeypatch.setattr("graph.situations._DATA_PATH", bad)
    load_situations.cache_clear()
    with pytest.raises(RuntimeError, match="templates"):
        load_situations()
    load_situations.cache_clear()
