"""Тесты ``missing_needs``: когда за данными в справочник идти нельзя."""

from __future__ import annotations

from graph.context import ConversationContext, missing_needs


def test_missing_needs_без_города_не_возвращает_городские():
    """Без города city_meta/price/branches не запрашиваются."""
    ctx = ConversationContext()
    assert missing_needs(ctx, ["city_meta", "price", "branches"]) == []
    assert missing_needs(ctx, ["city_choices"]) == ["city_choices"]
    assert missing_needs(ctx, ["city_meta"], profile={"city": "Пермь"}) == ["city_meta"]


def test_missing_needs_city_meta_при_пустой_статике():
    """Город известен, статики нет — city_meta нужна."""
    ctx = ConversationContext(city_name="Пермь")
    assert missing_needs(ctx, ["city_meta"]) == ["city_meta"]
    assert missing_needs(
        ConversationContext(),
        ["city_meta"],
        profile={"city": "Пермь"},
    ) == ["city_meta"]


def test_missing_needs_city_meta_при_собранной_статике():
    """Статика города уже запечена — city_meta не нужна."""
    ctx = ConversationContext(
        city_slug="perm",
        city_name="Пермь",
        static_text="Статика разговора:\nГород: Пермь (слаг perm).",
    )
    assert missing_needs(ctx, ["city_meta"]) == []


def test_missing_needs_branches_только_с_городом_без_филиала():
    """Потребность branches — только при известном городе и невыбранном филиале."""
    empty = ConversationContext()
    assert missing_needs(empty, ["branches"]) == []

    with_city = ConversationContext(city_slug="perm", city_name="Пермь")
    assert missing_needs(with_city, ["branches"]) == ["branches"]

    with_branch = ConversationContext(
        city_slug="perm",
        city_name="Пермь",
        branch_slug="perm_lenina",
    )
    assert missing_needs(with_branch, ["branches"]) == []

    with_dynamic = ConversationContext(
        city_slug="perm",
        dynamic_text="Филиалы под запрос: ул. Ленина, 1.",
    )
    assert missing_needs(with_dynamic, ["branches"]) == []


def test_missing_needs_price_и_branch_meta():
    """Потребность price — при городе без фразы; branch_meta — при филиале без статики."""
    city = ConversationContext(
        city_slug="perm",
        city_name="Пермь",
        static_text="Статика разговора:\nГород: Пермь (слаг perm).",
    )
    assert missing_needs(city, ["price"]) == ["price"]

    with_price = ConversationContext(
        city_slug="perm",
        static_text="Статика.\nЦена (готовая фраза, произносить только так): от 10000",
    )
    assert missing_needs(with_price, ["price"]) == []

    branch = ConversationContext(branch_slug="perm_lenina")
    assert missing_needs(branch, ["branch_meta"]) == ["branch_meta"]

    branch_ready = ConversationContext(
        branch_slug="perm_lenina",
        static_text="Выбранный филиал (слаг perm_lenina):\nАдрес: ул. Ленина.",
    )
    assert missing_needs(branch_ready, ["branch_meta"]) == []


def test_missing_needs_неизвестная_считается_нужной():
    """Незнакомая потребность всегда остаётся."""
    assert missing_needs(ConversationContext(), ["custom_fact"]) == ["custom_fact"]
