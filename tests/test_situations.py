"""Тесты словаря ситуативных заглушек."""

from __future__ import annotations

from graph.situations import DEFAULT_SLUG, load_situations, pick_filler


def test_словарь_имеет_default_и_доменные_слаги():
    catalog = load_situations()
    assert DEFAULT_SLUG in catalog
    assert len(catalog[DEFAULT_SLUG]) >= 5
    assert "город" in catalog and len(catalog["город"]) == 5
    assert "филиал" in catalog and len(catalog["филиал"]) == 5


def test_pick_filler_пустой_слаг_берёт_default():
    catalog = load_situations()
    phrase = pick_filler(None)
    assert phrase
    assert phrase in catalog[DEFAULT_SLUG]


def test_pick_filler_неизвестный_слаг_берёт_default():
    catalog = load_situations()
    phrase = pick_filler("нет-такого-слага")
    assert phrase
    assert phrase in catalog[DEFAULT_SLUG]


def test_pick_filler_мимо_уже_звучавших():
    catalog = load_situations()
    pool = list(catalog[DEFAULT_SLUG])
    spoken = pool[:-1]
    phrase = pick_filler("default", spoken=spoken)
    assert phrase == pool[-1]


def test_pick_filler_никогда_не_пустая():
    for slug in (None, "", "default", "город", "филиал", "xyz"):
        assert pick_filler(slug).strip()
