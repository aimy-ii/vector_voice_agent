"""Речевой фон хода: связки, пока ответ ещё не готов.

Пауза складывается из похода в справочник и работы генератора — разные узлы.
Задача создаётся в ``lookup`` и снимается в ``respond`` на первой дельте.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable, Sequence

from core.config import settings
from graph.fillers import bridge_filler

log = logging.getLogger(__name__)

#: Фоновые задачи по идентификатору звонка.
_tasks: dict[str, asyncio.Task[None]] = {}
#: Прозвучавшие на текущем ходе связки по ``call_id``.
_spoken: dict[str, list[str]] = {}


def background_spoken(call_id: str) -> list[str]:
    """Фразы фона, уже прозвучавшие к этому моменту.

    Args:
        call_id: идентификатор звонка.

    Returns:
        Копия списка прозвучавших связок; пустой, если фона нет.
    """
    return list(_spoken.get(call_id) or [])


async def start_background_speech(
    call_id: str,
    *,
    templates: Sequence[str],
    used: list[str],
    on_speak: Callable[[str], None],
) -> None:
    """Запускает фоновую цепочку связок для этого хода.

    Args:
        call_id: идентификатор звонка.
        templates: шаблоны связок из настроек.
        used: фразы, уже звучавшие в этом звонке (для выбора без повторов).
        on_speak: колбэк отправки готовой фразы в эфир.
    """
    stop_background_speech(call_id)
    heard: list[str] = []
    _spoken[call_id] = heard

    async def _run() -> None:
        """Спит, отдаёт связки до лимита; ошибки только в лог."""
        try:
            local_used = list(used)
            for index in range(settings.bridge_filler_limit):
                delay = settings.bridge_first_delay if index == 0 else settings.bridge_next_delay
                await asyncio.sleep(delay)
                phrase = bridge_filler(templates, used=local_used)
                if not phrase:
                    return
                on_speak(phrase)
                heard.append(phrase)
                local_used.append(phrase)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001
            log.warning("Речевой фон не удался: %s", exc)

    _tasks[call_id] = asyncio.create_task(_run())


def stop_background_speech(call_id: str) -> list[str]:
    """Останавливает фон и возвращает прозвучавшие фразы.

    Повторный вызов на уже снятой задаче безопасен.

    Args:
        call_id: идентификатор звонка.

    Returns:
        Фразы, успевшие прозвучать до остановки.
    """
    task = _tasks.pop(call_id, None)
    heard = list(_spoken.pop(call_id, []))
    if task is not None and not task.done():
        task.cancel()
    return heard
