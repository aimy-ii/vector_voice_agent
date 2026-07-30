"""Тесты шаблонов заглушек с предметом в именительном."""

from __future__ import annotations

import re

from core.config import Settings
from graph.fillers import (
    _DEFAULT_BRANCH,
    _DEFAULT_CITY,
    _DEFAULT_COST,
    branch_filler,
    bridge_filler,
    city_filler,
    cost_filler,
    pick_filler,
)

_PREPOSITION_BEFORE_PLACE = re.compile(r"\bпо\s*\{place\}")


def test_шаблоны_без_предлога_точки_и_одного_так():
    for pool in (_DEFAULT_CITY, _DEFAULT_BRANCH, _DEFAULT_COST):
        tak_count = sum(1 for tpl in pool if tpl.lstrip().startswith("Так"))
        assert tak_count <= 1, pool
        for tpl in pool:
            assert _PREPOSITION_BEFORE_PLACE.search(tpl) is None, tpl
            assert "…" not in tpl and "..." not in tpl, tpl
            assert not tpl.rstrip().endswith(("…", "...")), tpl
    for tpl in (*_DEFAULT_CITY, *_DEFAULT_BRANCH):
        if "{place}" in tpl or "{city}" in tpl:
            assert not re.search(r"\bпо\s*\{", tpl), tpl


def test_результат_оканчивается_точкой(monkeypatch):
    monkeypatch.setattr("graph.fillers.random.choice", lambda pool: pool[0])
    assert city_filler([]).endswith(".")
    assert branch_filler([]).endswith(".")
    assert cost_filler([]).endswith(".")


def test_city_filler_подставляет_название_или_город(monkeypatch):
    monkeypatch.setattr(
        "graph.fillers.random.choice",
        lambda pool: next(p for p in pool if p.startswith("Секунду, открываю")),
    )
    assert city_filler([], place="Санкт-Петербург") == ("Секунду, открываю Санкт-Петербург.")
    assert city_filler([]) == "Секунду, открываю город."


def test_branch_filler_с_местом_и_без(monkeypatch):
    templates = ["Угу, {place}. Сейчас посмотрю, что рядом."]
    monkeypatch.setattr("graph.fillers.random.choice", lambda pool: pool[0])
    assert branch_filler(templates, place="центр") == ("Угу, центр. Сейчас посмотрю, что рядом.")
    assert branch_filler(templates) == "Угу, филиал. Сейчас посмотрю, что рядом."


def test_ни_одна_фраза_не_оканчивается_многоточием():
    for pool in (_DEFAULT_CITY, _DEFAULT_BRANCH, _DEFAULT_COST):
        for tpl in pool:
            assert not tpl.rstrip().endswith(("…", "...")), tpl


def test_предлога_перед_плейсхолдером_нет():
    for tpl in (*_DEFAULT_CITY, *_DEFAULT_BRANCH):
        assert _PREPOSITION_BEFORE_PLACE.search(tpl) is None, tpl


def test_city_и_branch_из_настроек_оканчиваются_точкой():
    """Шаблоны из настроек: хвост — точка; «по филиал» / «по город» не выходят."""
    city_templates = Settings.model_fields["agent_city_fillers"].default_factory()
    branch_templates = Settings.model_fields["agent_branch_fillers"].default_factory()
    for tpl in city_templates:
        phrase = city_filler([tpl])
        assert phrase is not None
        assert phrase.endswith(".")
        assert "по город" not in phrase
        assert "…" not in phrase and "..." not in phrase
    for tpl in branch_templates:
        phrase = branch_filler([tpl])
        assert phrase is not None
        assert phrase.endswith(".")
        assert "по филиал." not in phrase
        assert "по филиал " not in phrase
        assert "…" not in phrase and "..." not in phrase


def test_хвостовое_многоточие_нормализуется_в_точку(monkeypatch):
    """Шаблон с «…» / «...» на конце после выбора даёт точку."""
    monkeypatch.setattr("graph.fillers.random.choice", lambda pool: pool[0])
    assert city_filler(["Так, {place}. Секунду…"]) == "Так, город. Секунду."
    assert branch_filler(["Так, {place}. Минутку..."]) == "Так, филиал. Минутку."


def test_предмет_вне_набора_даёт_none():
    assert pick_filler(["так, {place}"], subject="себя") is None
    assert pick_filler(["так, {place}"], subject=None) is None
    assert city_filler([]) is not None
    assert cost_filler([]) is not None


def test_bridge_filler_мимо_звучавших_без_плейсхолдеров_и_односложных(monkeypatch):
    templates = [
        "Так, сейчас посмотрю.",
        "Секунду, уточняю.",
        "Ага, вижу.",
    ]
    monkeypatch.setattr("graph.fillers.random.choice", lambda pool: pool[0])
    phrase = bridge_filler(templates, used=["Так, сейчас посмотрю."])
    assert phrase == "Секунду, уточняю."
    assert "{place}" not in phrase
    assert "{city}" not in phrase
    assert phrase.endswith(".")
    assert " " in phrase.rstrip(".")
    assert bridge_filler([], used=["Так, сейчас посмотрю."]) is None
    defaults = Settings.model_fields["agent_bridge_fillers"].default_factory()
    for tpl in defaults:
        assert "{place}" not in tpl and "{city}" not in tpl
        body = tpl.rstrip(".!?,:;")
        assert " " in body, tpl
