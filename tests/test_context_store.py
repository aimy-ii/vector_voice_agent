"""Тесты кеша контекста по идентификатору звонка."""

from __future__ import annotations

from graph.context import DYN_NONE, DYN_READY, ConversationContext
from graph.context_store import (
    CONTEXT_FIELDS_DYNAMIC,
    CONTEXT_FIELDS_STATIC,
    CONTEXT_FIELDS_TURN,
    MemoryContextStore,
    merge_context_fields,
)


async def test_точечная_запись_статика_и_динамика_не_затирают_друг_друга(
    memory_context_store: MemoryContextStore,
):
    """Два канала пишут свои поля — last-write-wins их не смешивает."""
    store = memory_context_store
    base = ConversationContext(
        static_text="Город: Пермь",
        city_slug="perm",
        city_name="Пермь",
        dynamic_status=DYN_NONE,
    )
    await store.save("c1", base)

    static_overlay = ConversationContext(
        static_text="Город: Пермь\nФилиал выбран",
        city_slug="perm",
        city_name="Пермь",
        branch_slug="perm_lenina",
        frozen=True,
    )
    cached = await store.load("c1")
    assert cached is not None
    merged_static = merge_context_fields(cached, static_overlay, CONTEXT_FIELDS_STATIC)
    await store.save("c1", merged_static)

    dynamic_overlay = ConversationContext(
        dynamic_text="Филиалы под запрос: ул. Ленина, 1.",
        dynamic_status=DYN_READY,
        dynamic_reply="какие филиалы у Ленина?",
        situation_slug=None,
        filler_spoken=False,
    )
    cached = await store.load("c1")
    assert cached is not None
    merged_dyn = merge_context_fields(cached, dynamic_overlay, CONTEXT_FIELDS_DYNAMIC)
    await store.save("c1", merged_dyn)

    # Вторая запись статики без динамики не должна затереть динамику.
    again_static = ConversationContext(
        static_text="Город: Пермь\nФилиал выбран",
        city_slug="perm",
        city_name="Пермь",
        branch_slug="perm_lenina",
        frozen=True,
        dynamic_text="",
        dynamic_status=DYN_NONE,
    )
    cached = await store.load("c1")
    assert cached is not None
    final = merge_context_fields(cached, again_static, CONTEXT_FIELDS_STATIC)
    await store.save("c1", final)

    loaded = await store.load("c1")
    assert loaded is not None
    assert loaded.static_text.startswith("Город: Пермь")
    assert loaded.branch_slug == "perm_lenina"
    assert "Филиалы под запрос" in loaded.dynamic_text
    assert loaded.dynamic_status == DYN_READY
    assert loaded.dynamic_reply == "какие филиалы у Ленина?"


async def test_промах_кеша_не_роняет_и_отдаёт_none(memory_context_store: MemoryContextStore):
    memory_context_store.fail = True
    assert await memory_context_store.save("c1", ConversationContext()) is False
    assert await memory_context_store.load("c1") is None


def test_merge_не_затирает_непустую_статику_пустой():
    base = ConversationContext(
        static_text="Город: Пермь",
        city_slug="perm",
        city_name="Пермь",
    )
    overlay = ConversationContext(
        static_text="",
        city_slug="",
        city_name="",
        dynamic_text="новое",
        dynamic_status=DYN_READY,
    )
    merged = merge_context_fields(
        base,
        overlay,
        CONTEXT_FIELDS_STATIC | CONTEXT_FIELDS_DYNAMIC,
    )
    assert merged.static_text == "Город: Пермь"
    assert merged.city_slug == "perm"
    assert merged.city_name == "Пермь"
    assert merged.dynamic_text == "новое"
    assert merged.dynamic_status == DYN_READY


async def test_поля_хода_пишутся_отдельно_не_затирают_статику_и_динамику(
    memory_context_store: MemoryContextStore,
):
    """last_agent_reply пишется набором TURN и не трогает остальные поля."""
    store = memory_context_store
    base = ConversationContext(
        static_text="Город: Пермь",
        city_slug="perm",
        city_name="Пермь",
        dynamic_text="Филиалы под запрос: ул. Ленина, 1.",
        dynamic_status=DYN_READY,
        dynamic_reply="какие у Ленина?",
        last_agent_reply="",
    )
    await store.save("c1", base)

    turn_overlay = ConversationContext(
        last_agent_reply="Рядом с Ленина есть филиал.",
        static_text="",
        dynamic_text="",
        dynamic_status=DYN_NONE,
    )
    cached = await store.load("c1")
    assert cached is not None
    merged = merge_context_fields(cached, turn_overlay, CONTEXT_FIELDS_TURN)
    await store.save("c1", merged)

    loaded = await store.load("c1")
    assert loaded is not None
    assert loaded.last_agent_reply == "Рядом с Ленина есть филиал."
    assert loaded.static_text == "Город: Пермь"
    assert loaded.city_slug == "perm"
    assert "Филиалы под запрос" in loaded.dynamic_text
    assert loaded.dynamic_status == DYN_READY
    assert loaded.dynamic_reply == "какие у Ленина?"


async def test_dynamic_turn_hash_pending_пишутся_с_динамикой(
    memory_context_store: MemoryContextStore,
):
    """dynamic_turn, last_reply_hash и pending_fields — набор динамики."""
    store = memory_context_store
    base = ConversationContext(
        static_text="Город: Пермь",
        city_slug="perm",
        city_name="Пермь",
    )
    await store.save("c1", base)

    dynamic_overlay = ConversationContext(
        dynamic_text="Ищем филиалы.",
        dynamic_status=DYN_READY,
        dynamic_reply="какие филиалы?",
        dynamic_turn=3,
        last_reply_hash="abc123",
        dynamic_reply_hash="dyn456",
        pending_fields=["branch"],
    )
    cached = await store.load("c1")
    assert cached is not None
    merged = merge_context_fields(cached, dynamic_overlay, CONTEXT_FIELDS_DYNAMIC)
    await store.save("c1", merged)

    # Статика поверх не затирает динамические поля.
    static_overlay = ConversationContext(
        static_text="Город: Пермь\nФилиал",
        city_slug="perm",
        city_name="Пермь",
        branch_slug="perm_lenina",
        frozen=True,
        dynamic_turn=0,
        last_reply_hash="",
        dynamic_reply_hash="",
        pending_fields=[],
    )
    cached = await store.load("c1")
    assert cached is not None
    final = merge_context_fields(cached, static_overlay, CONTEXT_FIELDS_STATIC)
    await store.save("c1", final)

    loaded = await store.load("c1")
    assert loaded is not None
    assert loaded.static_text.startswith("Город: Пермь")
    assert loaded.branch_slug == "perm_lenina"
    assert loaded.dynamic_turn == 3
    assert loaded.last_reply_hash == "abc123"
    assert loaded.dynamic_reply_hash == "dyn456"
    assert loaded.pending_fields == ["branch"]
    assert "Ищем филиалы" in loaded.dynamic_text


def test_nearby_поля_в_динамике_не_в_статике() -> None:
    """nearby_text и nearby_key — динамика; в статику не входят."""
    assert "nearby_text" in CONTEXT_FIELDS_DYNAMIC
    assert "nearby_key" in CONTEXT_FIELDS_DYNAMIC
    assert "nearby_text" not in CONTEXT_FIELDS_STATIC
    assert "nearby_key" not in CONTEXT_FIELDS_STATIC


def test_merge_динамики_переносит_nearby() -> None:
    """Слияние по динамике переносит непустые nearby_text и nearby_key."""
    base = ConversationContext(static_text="Город: Пермь")
    overlay = ConversationContext(
        nearby_text="Ближайшие филиалы к месту «Солнечный»",
        nearby_key="perm:солнечный",
    )
    merged = merge_context_fields(base, overlay, CONTEXT_FIELDS_DYNAMIC)
    assert merged.nearby_text == "Ближайшие филиалы к месту «Солнечный»"
    assert merged.nearby_key == "perm:солнечный"
    assert merged.static_text == "Город: Пермь"


def test_merge_динамики_затирает_nearby_пустой_строкой() -> None:
    """Сброс nearby_text пустой строкой проходит (в отличие от статики)."""
    base = ConversationContext(
        static_text="Город: Пермь",
        nearby_text="Ближайшие филиалы…",
        nearby_key="perm:солнечный",
    )
    overlay = ConversationContext(nearby_text="", nearby_key="")
    merged = merge_context_fields(base, overlay, CONTEXT_FIELDS_DYNAMIC)
    assert merged.nearby_text == ""
    assert merged.nearby_key == ""
    assert merged.static_text == "Город: Пермь"
