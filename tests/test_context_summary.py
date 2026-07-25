"""Тесты имени, заглушек, контекста и саммари."""

from __future__ import annotations

from graph.context import (
    DYN_MISSING,
    DYN_NONE,
    DYN_READY,
    DYN_SEARCHING,
    ConversationContext,
    format_city_static,
    merge_static,
)
from graph.fillers import city_filler
from graph.names import given_name
from graph.summary import build_summary


def test_статусы_динамики_константы_и_дефолт():
    assert DYN_NONE == "не требуется"
    assert DYN_READY == "готово"
    assert DYN_SEARCHING == "в поиске"
    assert DYN_MISSING == "не нашлось"
    ctx = ConversationContext()
    assert ctx.dynamic_status == DYN_NONE
    assert ctx.situation_slug is None
    assert ctx.filler_spoken is False
    assert ctx.render() == ""


def test_имя_три_случая():
    assert given_name("Андрей Андреевич") == "Андрей"
    assert given_name("Андрей Петров") == "Андрей Петров"
    assert given_name("Мария Ивановна") == "Мария"


def test_заглушка_без_вызова_модели():
    phrase = city_filler(["так, {place}… секунду, открываю по {place}"])
    assert phrase is not None
    assert "город" in phrase
    assert "поищу" not in phrase.lower()


def test_в_заглушку_не_попадает_чужой_текст():
    from graph.fillers import FILLER_SUBJECTS, pick_filler

    phrase = pick_filler(
        ["так, {place}… открываю по {place}"],
        subject="город",
    )
    assert phrase is not None
    assert "себя" not in phrase
    assert "Для" not in phrase
    assert pick_filler(["так, {place}"], subject="себя") is None
    assert pick_filler(["так"], subject=None) is None
    assert FILLER_SUBJECTS == frozenset({"город", "филиал", "стоимость"})


def test_контекст_статика_один_раз_цена_фразой(fake_kb):
    city = {
        "slug": "perm",
        "name": "Пермь",
        "categories": [{"code": "B", "duration": "2,5 месяца"}],
        "vehicles": {"manual": ["Solaris"], "automatic": []},
        "theory_formats": ["очно"],
        "documents": ["паспорт"],
        "payment": {"installment": True},
        "messengers": ["Telegram"],
    }
    text = format_city_static(
        city_slug="perm",
        city_name="Пермь",
        city_meta=city,
        price_line="Стоимость обучения — от 43900 рублей.",
    )
    assert "Пермь" in text and "perm" in text
    assert "от 43900" in text
    assert "amount" not in text
    assert "Список филиалов" in text

    ctx = ConversationContext()
    ctx = merge_static(ctx, city_slug="perm", city_name="Пермь", city_meta=city, price_line="фраза")
    again = merge_static(ctx, city_slug="spb", city_name="Питер", city_meta=city)
    assert again.city_slug == "perm"
    assert again.city_name == "Пермь"


def test_мета_филиала_после_выбора():
    ctx = ConversationContext(city_slug="perm", city_name="Пермь", static_text="город")
    assert ctx.branch_slug is None
    ctx = merge_static(
        ctx,
        branch_slug="perm_chernyshevskogo",
        branch_meta={"address": "ул. Чернышевского, 28", "place_type": "учебный офис"},
    )
    assert ctx.branch_slug == "perm_chernyshevskogo"
    assert "Чернышевского" in ctx.static_text
    assert ctx.frozen is True


def test_саммари_город_слагом_и_названием(script):
    summary = build_summary(
        script=script,
        step_status={"name": "closed", "city": "closed"},
        profile={"caller_name": "Андрей", "city": "Пермь"},
        city_slug="perm",
        city_name="Пермь",
        branch_slug=None,
    )
    assert summary["city"] == {"slug": "perm", "name": "Пермь"}
    assert summary["steps"]["name"]["caller_name"] == "Андрей"
    assert "в нашу пользу" not in summary
