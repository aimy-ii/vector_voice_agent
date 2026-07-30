"""Тесты словаря ситуативных заглушек."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from graph.situations import (
    DEFAULT_SLUG,
    TEMPLATES_KEY,
    _ensure_terminal_punct,
    load_situations,
    pick_ack,
    pick_filler,
)


def test_словарь_имеет_templates_и_default():
    catalog = load_situations()
    assert DEFAULT_SLUG in catalog
    assert TEMPLATES_KEY in catalog
    assert len(catalog[DEFAULT_SLUG]) >= 8
    assert len(catalog[TEMPLATES_KEY]) >= 5
    assert all("{subject}" in t for t in catalog[TEMPLATES_KEY])
    assert all("по {subject}" not in t for t in catalog[TEMPLATES_KEY])


def test_pick_filler_подстановка_предмета_именительный():
    phrase = pick_filler("филиалы")
    assert phrase
    assert "филиалы" in phrase
    assert "по филиалы" not in phrase
    assert "филиалы, да… секунду" in phrase or "филиалы" in phrase
    assert phrase[-1] in ".!?…,:;"


def test_pick_filler_шаблон_филиалы_да_секунду(monkeypatch):
    """Конкретный шаблон даёт именительный без «по»."""
    catalog = load_situations()
    target = _ensure_terminal_punct("{subject}, да… секунду".format(subject="филиалы"))
    monkeypatch.setattr(
        "graph.situations.random.choice",
        lambda pool: next(p for p in pool if p.startswith("филиалы, да")),
    )
    phrase = pick_filler("филиалы")
    assert phrase == target
    assert phrase == "филиалы, да… секунду."
    assert "по филиалы" not in phrase
    assert all(t.endswith("{subject}") or "{subject}" in t for t in catalog[TEMPLATES_KEY])


def test_pick_filler_пустой_предмет_берёт_default():
    catalog = load_situations()
    defaults = {_ensure_terminal_punct(p) for p in catalog[DEFAULT_SLUG]}
    phrase = pick_filler(None)
    assert phrase in defaults
    assert pick_filler("") in defaults
    assert pick_filler("   ") in defaults
    assert phrase[-1] in ".!?…,:;"


def test_pick_filler_мимо_уже_звучавших():
    catalog = load_situations()
    subject = "медкомиссия"
    pool = [_ensure_terminal_punct(t.format(subject=subject)) for t in catalog[TEMPLATES_KEY]]
    spoken = pool[:-1]
    phrase = pick_filler(subject, spoken=spoken)
    assert phrase == pool[-1]


def test_pick_filler_никогда_не_пустая():
    for subject in (None, "", "филиалы", "пересдача", "медкомиссия"):
        assert pick_filler(subject).strip()


def test_pick_ack_мимо_уже_звучавших():
    catalog = load_situations()
    pool = [_ensure_terminal_punct(p) for p in catalog[DEFAULT_SLUG]]
    spoken = pool[:-1]
    phrase = pick_ack(spoken=spoken)
    assert phrase == pool[-1]


def test_pick_ack_непустая_при_исчерпанном_наборе():
    catalog = load_situations()
    pool = [_ensure_terminal_punct(p) for p in catalog[DEFAULT_SLUG]]
    phrase = pick_ack(spoken=pool)
    assert phrase.strip()
    assert phrase in pool


def test_pick_ack_без_предмета_в_default():
    catalog = load_situations()
    assert all("{subject}" not in p and "{place}" not in p for p in catalog[DEFAULT_SLUG])
    phrase = pick_ack()
    assert "{subject}" not in phrase
    assert "{place}" not in phrase
    assert phrase[-1] in ".!?…,:;"


def test_отсутствие_обязательного_ключа_ошибка_загрузки(tmp_path: Path, monkeypatch):
    bad = tmp_path / "situations.json"
    bad.write_text(json.dumps({"default": ["так… секунду"]}), encoding="utf-8")
    monkeypatch.setattr("graph.situations._DATA_PATH", bad)
    load_situations.cache_clear()
    with pytest.raises(RuntimeError, match="templates"):
        load_situations()
    load_situations.cache_clear()
