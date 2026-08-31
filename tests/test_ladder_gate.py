"""Лестница ожидания включается по нуждам шага, а не по общему статусу.

Статус динамики один на весь контекст: пока фон подбирает филиалы, он
стоит «в поиске» независимо от того, о чём сейчас разговор. По одному ему
бот тянул время на шаге «кто учится» — «стоимость сейчас уточняю», — при
том что цена уже лежала в данных. Заказчик закладывал обратное: поиск по
одной теме не должен блокировать ход по другой.
"""

from __future__ import annotations

from graph.context import DYN_SEARCHING, DYN_WORKING, ConversationContext, missing_needs
from graph.facts import needs_of
from graph.nodes import _ladder_prompt_kind


def test_нужд_у_шага_нет_ступень_не_нужна() -> None:
    """Шагу данные не нужны — ход идёт штатно, даже когда фон ищет."""
    kind = _ladder_prompt_kind(
        status=DYN_SEARCHING,
        same_reply=True,
        stubs_spoken=0,
        force_full=True,
    )
    assert kind == "full"


def test_нужды_есть_и_идёт_поиск_ступень_ожидания() -> None:
    """Шагу не хватает данных и фон их ищет — заглушка уместна."""
    kind = _ladder_prompt_kind(
        status=DYN_SEARCHING,
        same_reply=True,
        stubs_spoken=0,
        force_full=False,
    )
    assert kind == "waiting"


def test_нужды_есть_и_идёт_работа_живая_реакция() -> None:
    """Разбор реплики ещё идёт — короткая реакция, а не молчание."""
    kind = _ladder_prompt_kind(
        status=DYN_WORKING,
        same_reply=True,
        stubs_spoken=0,
        force_full=False,
    )
    assert kind == "filler"


def test_диспетчер_отбрасывает_уже_добытое(script_v4) -> None:
    """``missing_needs`` не просит того, что уже лежит в контексте.

    Это и есть признак, по которому ход решает, ждать ему или говорить.
    """
    step = script_v4.step("who_studies")
    context = ConversationContext(dynamic_status=DYN_SEARCHING)

    assert missing_needs(context, needs_of(step), {}) == [], (
        "шагу «кто учится» справочник не нужен — ждать нечего"
    )


def test_вопрос_клиента_держит_ход_на_ожидании() -> None:
    """Ищут то, о чём человек спросил — ждём, даже если шагу данные не нужны.

    Поиск запускают двое: код по нуждам шага и агент по вопросу клиента.
    Второй помечает предмет в ``situation_slug``. Без этой ветки бот
    ответил бы на вопрос, не дождавшись данных, которые уже едут.
    """
    context = ConversationContext(dynamic_status=DYN_SEARCHING, situation_slug="медкомиссия")

    asked_by_client = bool((context.situation_slug or "").strip())
    lead_missing = missing_needs(context, [], {})

    kind = _ladder_prompt_kind(
        status=DYN_SEARCHING,
        same_reply=True,
        stubs_spoken=0,
        force_full=not (bool(lead_missing) or asked_by_client),
    )
    assert kind == "waiting"


def test_поиск_по_чужой_теме_ход_не_держит() -> None:
    """Предмета нет и шагу данные не нужны — это фон по чужой теме."""
    context = ConversationContext(dynamic_status=DYN_SEARCHING)

    asked_by_client = bool((context.situation_slug or "").strip())
    lead_missing = missing_needs(context, [], {})

    kind = _ladder_prompt_kind(
        status=DYN_SEARCHING,
        same_reply=True,
        stubs_spoken=0,
        force_full=not (bool(lead_missing) or asked_by_client),
    )
    assert kind == "full"
