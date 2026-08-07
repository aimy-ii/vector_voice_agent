"""Единая история звонка: накопление, слияние со снимком бота, отдача в промпт.

Снимок бота — правда о том, что реально прозвучало, но он отстаёт: пока TTS
не начал говорить, реплики мозга в нём нет. Поэтому историю копим сами, а
снимок используем, чтобы пометить прозвучавшее и подтянуть реплики человека.
Слияние идёт по порядку, а не по множеству: две одинаковые реплики подряд
(«Да.» и «Да.») — это две реплики, схлопывать их нельзя.
"""

from __future__ import annotations

from typing import Sequence

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from pydantic import BaseModel

from graph.history import normalize

#: Роль реплики бота.
ROLE_AGENT = "agent"
#: Роль реплики человека.
ROLE_CLIENT = "client"


class TranscriptEntry(BaseModel):
    """Одна реплика в истории звонка.

    Attributes:
        entry_id: устойчивый идентификатор записи, нужен для трассировки.
        role: ``agent`` или ``client``.
        text: текст реплики; для бота при перебивании заменяется на
            фактически произнесённую часть из снимка.
        spoken: реплика подтверждена снимком бота, то есть прозвучала.
            Реплики человека приходят только из снимка и всегда ``True``.
    """

    entry_id: str
    role: str
    text: str
    spoken: bool = False


def role_of(message: BaseMessage) -> str:
    """Роль сообщения истории.

    Args:
        message: сообщение из снимка бота.

    Returns:
        ``agent`` для реплики бота, иначе ``client``.
    """
    return ROLE_AGENT if isinstance(message, AIMessage) else ROLE_CLIENT


def text_of(message: BaseMessage) -> str:
    """Текст сообщения строкой.

    Args:
        message: сообщение из снимка бота.

    Returns:
        Содержимое без обрамляющих пробелов.
    """
    content = message.content
    if isinstance(content, str):
        return content.strip()
    return " ".join(str(part) for part in content).strip()


def _matches(stored: str, incoming: str) -> bool:
    """Одна ли это реплика: совпадение целиком или усечение при перебивании."""
    left = normalize(stored)
    right = normalize(incoming)
    if not left or not right:
        return False
    return left == right or left.startswith(right)


def append_agent(
    entries: Sequence[TranscriptEntry],
    *,
    turn: int,
    text: str,
) -> list[TranscriptEntry]:
    """Дописывает реплику бота в историю в момент генерации.

    Озвучки ещё не было, поэтому ``spoken`` остаётся ложью — его выставит
    слияние, когда реплика придёт в снимке бота.

    Args:
        entries: накопленная история.
        turn: номер хода, из него собирается идентификатор.
        text: полный текст реплики.

    Returns:
        Новый список с дописанной репликой; пустой текст ничего не меняет.
    """
    body = list(entries)
    clean = (text or "").strip()
    if not clean:
        return body
    body.append(
        TranscriptEntry(
            entry_id=f"agent:{turn}:{len(body)}",
            role=ROLE_AGENT,
            text=clean,
            spoken=False,
        )
    )
    return body


def merge_snapshot(
    entries: Sequence[TranscriptEntry],
    snapshot: Sequence[BaseMessage],
    *,
    turn: int,
) -> list[TranscriptEntry]:
    """Сливает снимок бота с накопленной историей по порядку.

    Для каждой реплики снимка ищем ближайшее несопоставленное совпадение в
    накопленном: совпало — помечаем прозвучавшей и берём текст из снимка
    (при перебивании он короче); не нашли — это новая реплика, дописываем.
    Записи, которых в снимке нет, остаются на своих местах: это реплики
    мозга, которые ещё не успели прозвучать.

    Args:
        entries: накопленная история.
        snapshot: история, присланная ботом на этот ход.
        turn: номер хода, из него собираются идентификаторы новых записей.

    Returns:
        Слитая история.
    """
    stored = list(entries)
    result: list[TranscriptEntry] = []
    cursor = 0
    for message in snapshot:
        role = role_of(message)
        text = text_of(message)
        if not text:
            continue
        found: int | None = None
        probe = cursor
        while probe < len(stored):
            candidate = stored[probe]
            if candidate.role == role and _matches(candidate.text, text):
                found = probe
                break
            probe += 1
        if found is None:
            result.append(
                TranscriptEntry(
                    entry_id=f"{role}:{turn}:{len(result)}",
                    role=role,
                    text=text,
                    spoken=True,
                )
            )
            continue
        result.extend(stored[cursor:found])
        result.append(stored[found].model_copy(update={"text": text, "spoken": True}))
        cursor = found + 1
    result.extend(stored[cursor:])
    return result


def to_messages(entries: Sequence[TranscriptEntry]) -> list[BaseMessage]:
    """Переводит историю в сообщения для промпта и судьи.

    Args:
        entries: накопленная история.

    Returns:
        Список сообщений в порядке разговора.
    """
    out: list[BaseMessage] = []
    for item in entries:
        if item.role == ROLE_AGENT:
            out.append(AIMessage(content=item.text))
        else:
            out.append(HumanMessage(content=item.text))
    return out


def count_spoken_agent(entries: Sequence[TranscriptEntry]) -> int:
    """Сколько реплик бота действительно прозвучало.

    Нужна проверке доставки: запись в историю сама по себе доставкой не
    является, её подтверждает только снимок бота.

    Args:
        entries: накопленная история.

    Returns:
        Число подтверждённых реплик бота.
    """
    return sum(1 for item in entries if item.role == ROLE_AGENT and item.spoken)
