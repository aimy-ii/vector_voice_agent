"""История звонка: мозг ведёт её сам и никогда не теряет.

Снимок, который бот присылает вместе с ходом, историей не является: в нём
только то, что бот успел проговорить, поэтому реплики мозга из него
пропадают. Здесь история копится своим порядком — фраза клиента ложится
в момент прихода хода, реплика бота в момент генерации. Ничего не
сравнивается и не сшивается: каждая фраза записывается ровно один раз тем,
кто её породил.
"""

from __future__ import annotations

from typing import Sequence

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from pydantic import BaseModel

#: Роль реплики бота.
ROLE_AGENT = "agent"
#: Роль реплики человека.
ROLE_CLIENT = "client"


class TranscriptEntry(BaseModel):
    """Одна реплика в истории звонка.

    Attributes:
        entry_id: идентификатор записи для трассировки: роль, ход, позиция.
        role: ``agent`` или ``client``.
        text: текст реплики.
    """

    entry_id: str
    role: str
    text: str


def _append(
    entries: Sequence[TranscriptEntry],
    *,
    role: str,
    turn: int,
    text: str,
) -> list[TranscriptEntry]:
    """Дописывает реплику в конец истории.

    Args:
        entries: накопленная история.
        role: чья реплика.
        turn: номер хода, из него собирается идентификатор.
        text: текст; пустой ничего не меняет.

    Returns:
        Новый список с дописанной репликой.
    """
    body = list(entries)
    clean = (text or "").strip()
    if not clean:
        return body
    body.append(
        TranscriptEntry(
            entry_id=f"{role}:{turn}:{len(body)}",
            role=role,
            text=clean,
        )
    )
    return body


def append_agent(
    entries: Sequence[TranscriptEntry],
    *,
    turn: int,
    text: str,
) -> list[TranscriptEntry]:
    """Дописывает реплику бота в момент генерации.

    Ждать озвучки нельзя: пока TTS молчит, может пройти ещё один ход, и
    реплика потеряется — ровно из-за этого бот повторял одну тему трижды.

    Args:
        entries: накопленная история.
        turn: номер хода.
        text: текст реплики.

    Returns:
        Новый список с дописанной репликой.
    """
    return _append(entries, role=ROLE_AGENT, turn=turn, text=text)


def append_client(
    entries: Sequence[TranscriptEntry],
    *,
    turn: int,
    text: str,
) -> list[TranscriptEntry]:
    """Дописывает фразу клиента в момент прихода хода.

    Распознавание нестриминговое: фраза приходит целиком и ровно один раз,
    поэтому проверять её на повтор не нужно. Две одинаковые фразы подряд —
    это две записи.

    Args:
        entries: накопленная история.
        turn: номер хода.
        text: распознанная фраза.

    Returns:
        Новый список с дописанной фразой.
    """
    return _append(entries, role=ROLE_CLIENT, turn=turn, text=text)


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


def reconcile_last_agent(
    entries: Sequence[TranscriptEntry],
    *,
    spoken: str,
) -> list[TranscriptEntry]:
    """Приводит последнюю реплику бота в истории к тому, что ушло в эфир.

    Реплика ложится в историю в момент генерации: иначе при нескольких ходах
    подряд модель не видит собственных слов и повторяется. Но в эфир она может
    уйти не вся или не уйти вовсе — человек перебил, или следующий ход обогнал
    предыдущий. К началу следующего хода это уже известно: в снимке бота лежит
    то, что действительно прозвучало.

    Правится только последняя запись бота. Списки не сшиваются: всё, что
    старше, уже сверено на предыдущих ходах.

    Args:
        entries: накопленная история.
        spoken: последняя реплика бота из снимка; пустая — бот не говорил.

    Returns:
        История, где последняя запись бота равна прозвучавшему тексту.
        Не прозвучавшая запись удаляется, прозвучавшая частично — урезается.
    """
    body = list(entries)
    if not body or body[-1].role != ROLE_AGENT:
        return body
    said = (spoken or "").strip()
    planned = body[-1].text
    if said == planned:
        return body
    if said and planned.startswith(said):
        body[-1] = body[-1].model_copy(update={"text": said})
        return body
    return body[:-1]
