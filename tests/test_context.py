"""Тесты ``missing_needs``: когда за данными в справочник идти нельзя."""

from __future__ import annotations

from graph.context import ContextState, ConversationContext, missing_needs


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


def test_missing_needs_не_возвращает_empty_needs():
    """Потребность из empty_needs не считается недостающей."""
    from graph.context import record_empty_needs

    ctx = ConversationContext(
        city_slug="perm",
        city_name="Пермь",
        empty_needs=["price"],
    )
    assert missing_needs(ctx, ["price", "branches"]) == ["branches"]
    assert missing_needs(ctx, ["price"]) == []

    # После успешного похода потребность уходит — missing снова её видит.
    record_empty_needs(ctx, ["price"], found=True)
    assert ctx.empty_needs == []
    assert missing_needs(ctx, ["price"]) == ["price"]


def test_render_ближайшие_между_статикой_и_динамикой() -> None:
    """Блок ближайших стоит между статикой и динамикой."""
    ctx = ConversationContext(
        static_text="СТАТИКА",
        nearby_text="БЛИЖАЙШИЕ",
        dynamic_text="ДИНАМИКА",
    )
    rendered = ctx.render()
    assert rendered.index("СТАТИКА") < rendered.index("БЛИЖАЙШИЕ")
    assert rendered.index("БЛИЖАЙШИЕ") < rendered.index("ДИНАМИКА")


def test_render_пустой_nearby_не_добавляет_блок() -> None:
    """Пустой nearby_text в документ ничего не добавляет."""
    ctx = ConversationContext(static_text="СТАТИКА", nearby_text="", dynamic_text="ДИНАМИКА")
    rendered = ctx.render()
    assert rendered == "СТАТИКА\n\nДИНАМИКА"


def test_context_state_nearby_поля_переживают_to_context() -> None:
    """ContextState с nearby_* отдаёт те же поля в ConversationContext."""
    state = ContextState(
        nearby_text="Ближайшие филиалы…",
        nearby_key="perm:солнечный",
        nearby_found=True,
    )
    ctx = state.to_context()
    assert ctx.nearby_text == "Ближайшие филиалы…"
    assert ctx.nearby_key == "perm:солнечный"
    assert ctx.nearby_found is True
