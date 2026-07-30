"""Тесты шаблонов заглушек с предметом в именительном."""

from __future__ import annotations

import re

from graph.fillers import (
    _DEFAULT_BRANCH,
    _DEFAULT_CITY,
    _DEFAULT_COST,
    branch_filler,
    city_filler,
    cost_filler,
    pick_filler,
)

_PREPOSITION_BEFORE_PLACE = re.compile(r"\bпо\s*\{place\}")


def test_шаблоны_без_предлога_перед_плейсхолдером():
    for pool in (_DEFAULT_CITY, _DEFAULT_BRANCH, _DEFAULT_COST):
        for tpl in pool:
            assert _PREPOSITION_BEFORE_PLACE.search(tpl) is None, tpl
            assert "{place}" in tpl


def test_подстановка_филиал_именительный(monkeypatch):
    monkeypatch.setattr(
        "graph.fillers.random.choice",
        lambda pool: next(p for p in pool if p.startswith("так, {place}")),
    )
    phrase = branch_filler([])
    assert phrase == "так, филиал… секунду, гляну адреса"
    assert "по филиал" not in phrase


def test_предмет_вне_набора_даёт_none():
    assert pick_filler(["так, {place}"], subject="себя") is None
    assert pick_filler(["так, {place}"], subject=None) is None
    assert city_filler([]) is not None
    assert cost_filler([]) is not None
