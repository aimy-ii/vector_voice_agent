"""Реплика человека не пропадает, даже если снимок бота пришёл без неё.

Боевой разбор: распознавание услышало «Дороговато. В соседней автошколе
дешевле. Если бы была какая-то скидка, ещё бы можно было подумать»,
лайв-канал её получил, бот на неё ответил — а в истории мозга её нет.
Ход прошёл со снимком без реплики, и мозг записал пустоту молча.
"""

from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage

from graph.context import ConversationContext
from graph.nodes import client_text_for_turn
from graph.transcript import TranscriptEntry

#: Та самая фраза с боевого разбора.
LOST = "Дороговато. В соседней автошколе дешевле. Если бы была какая-то скидка."


def entries(*pairs: tuple[str, str]) -> list[TranscriptEntry]:
    """Собирает историю из пар «роль, текст».

    Args:
        pairs: роль и текст каждой реплики.

    Returns:
        История звонка.
    """
    return [
        TranscriptEntry(entry_id=f"{role}:{index}", role=role, text=text)
        for index, (role, text) in enumerate(pairs)
    ]


def test_обычный_ход_берёт_фразу_из_снимка():
    text = client_text_for_turn(
        snapshot=[AIMessage(content="Механика или автомат?"), HumanMessage(content="Механика.")],
        context=ConversationContext(dynamic_reply="что-то старое"),
        entries=entries(("agent", "Механика или автомат?")),
    )
    assert text == "Механика."


def test_пустой_снимок_восстанавливает_фразу_из_лайв_канала():
    text = client_text_for_turn(
        snapshot=[AIMessage(content="Стоимость обучения — от 39900 рублей.")],
        context=ConversationContext(dynamic_reply=LOST),
        entries=entries(("agent", "Стоимость обучения — от 39900 рублей.")),
    )
    assert text == LOST


def test_восстановленная_фраза_не_дублируется():
    """Та же фраза уже в истории — второй раз не пишем."""
    text = client_text_for_turn(
        snapshot=[AIMessage(content="Поняла.")],
        context=ConversationContext(dynamic_reply=LOST),
        entries=entries(("client", LOST), ("agent", "Поняла.")),
    )
    assert text == ""


def test_нечего_восстанавливать_пустая_строка():
    text = client_text_for_turn(
        snapshot=[AIMessage(content="Поняла.")],
        context=ConversationContext(dynamic_reply=""),
        entries=entries(("agent", "Поняла.")),
    )
    assert text == ""


def test_старая_фраза_из_лайв_канала_не_подставляется_поверх_новой():
    """В истории уже есть более поздняя реплика человека — берём пустоту.

    Иначе разговор поедет назад: мозг допишет позавчерашнее возражение
    после того, как человек уже сказал что-то другое. ``dynamic_reply``
    живёт в кеше и после своего хода, поэтому сверяться с одной только
    последней записью нельзя.
    """
    text = client_text_for_turn(
        snapshot=[AIMessage(content="Поняла.")],
        context=ConversationContext(dynamic_reply=LOST),
        entries=entries(("client", LOST), ("agent", "Скидки есть."), ("client", "А группы когда?")),
    )
    assert text == ""
