"""Хранилище контекста разговора по идентификатору звонка.

Контекст живёт в Redis, как прогресс скрипта: основной ход и лайв-канал
пишут в разное время и не делят тред. Статику и динамику пишут разные
каналы точечными полями — иначе last-write-wins затрёт чужую половину.
Промах или недоступность Redis не ломают ход — восстанавливаем из
состояния треда.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Protocol

from core.config import settings
from graph.context import ConversationContext

log = logging.getLogger(__name__)

#: Префикс ключей, чтобы не пересечься с серверными ключами LangGraph.
KEY_PREFIX = "vector:context:"

#: Поля статики: пишет прогрев и разбор города/филиала в лайв-канале.
CONTEXT_FIELDS_STATIC: frozenset[str] = frozenset(
    {
        "static_text",
        "city_slug",
        "city_name",
        "branch_slug",
        "branch_candidates",
        "city_faq",
        "frozen",
    }
)
#: Поля динамики: пишет контекстер и лайв-канал.
CONTEXT_FIELDS_DYNAMIC: frozenset[str] = frozenset(
    {
        "dynamic_text",
        "dynamic_status",
        "situation_slug",
        "filler_spoken",
        "dynamic_reply",
        "dynamic_turn",
        "last_reply_hash",
        "dynamic_reply_hash",
        "pending_fields",
        "empty_needs",
        "nearby_text",
        "nearby_key",
        "conversation_ended",
    }
)
#: Поля хода: пишет только основной ход после генерации.
CONTEXT_FIELDS_TURN: frozenset[str] = frozenset({"last_agent_reply", "transcript"})
#: Все поля контекста (полная запись без слияния).
CONTEXT_FIELDS_ALL: frozenset[str] = (
    CONTEXT_FIELDS_STATIC | CONTEXT_FIELDS_DYNAMIC | CONTEXT_FIELDS_TURN
)


def merge_context_fields(
    base: ConversationContext,
    overlay: ConversationContext,
    fields: frozenset[str],
) -> ConversationContext:
    """Накладывает выбранные поля ``overlay`` на копию ``base``.

    Пустые строки статики не затирают непустые в базе: статика кладётся
    один раз, резать её нельзя.

    Args:
        base: контекст из кеша (актуальный).
        overlay: локальные правки канала.
        fields: какие поля накладывать.

    Returns:
        Новый контекст со слиянием выбранных полей.
    """
    data = base.model_dump()
    overlay_data = overlay.model_dump()
    for name in fields:
        if name not in overlay_data:
            continue
        value = overlay_data[name]
        if name in CONTEXT_FIELDS_STATIC and isinstance(value, str) and not value.strip():
            existing = data.get(name)
            if isinstance(existing, str) and existing.strip():
                continue
        data[name] = value
    return ConversationContext.model_validate(data)


class ContextStore(Protocol):
    """Контракт хранилища контекста разговора."""

    async def load(self, call_id: str) -> ConversationContext | None:
        """Читает контекст по идентификатору звонка или None при промахе."""

    async def save(self, call_id: str, context: ConversationContext) -> bool:
        """Пишет контекст. True — записали, False — Redis недоступен."""


class MemoryContextStore:
    """Заглушка хранилища в памяти процесса — для офлайн-тестов."""

    def __init__(self) -> None:
        """Создаёт пустое хранилище."""
        self._data: dict[str, ConversationContext] = {}
        self.fail: bool = False

    async def load(self, call_id: str) -> ConversationContext | None:
        """Читает контекст или None при промахе/сбое."""
        if self.fail:
            return None
        context = self._data.get(call_id)
        if context is None:
            return None
        return ConversationContext.model_validate(context.model_dump())

    async def save(self, call_id: str, context: ConversationContext) -> bool:
        """Пишет контекст; при ``fail`` притворяется, что Redis недоступен."""
        if self.fail:
            return False
        self._data[call_id] = ConversationContext.model_validate(context.model_dump())
        return True


class RedisContextStore:
    """Хранилище контекста в Redis с собственным префиксом и TTL."""

    def __init__(self, url: str | None = None, *, ttl_seconds: int | None = None) -> None:
        """Создаёт клиент Redis.

        Args:
            url: адрес Redis; пусто — из настроек.
            ttl_seconds: TTL ключа; пусто — из ``settings.script_redis_ttl``.
        """
        self._url = url or settings.redis_url
        self._ttl = ttl_seconds if ttl_seconds is not None else settings.script_redis_ttl
        self._client: Any = None

    def _key(self, call_id: str) -> str:
        """Ключ контекста по идентификатору звонка."""
        return f"{KEY_PREFIX}{call_id}"

    async def _get_client(self) -> Any:
        """Лениво открывает async-клиент Redis."""
        if self._client is None:
            from redis.asyncio import Redis

            self._client = Redis.from_url(self._url, decode_responses=True)
        return self._client

    async def load(self, call_id: str) -> ConversationContext | None:
        """Читает контекст; при любой ошибке — None, ход не роняем."""
        try:
            client = await self._get_client()
            raw = await client.get(self._key(call_id))
        except Exception as exc:  # noqa: BLE001
            log.warning("Redis недоступен при чтении контекста: %s", exc)
            return None
        if not raw:
            return None
        try:
            return ConversationContext.model_validate(json.loads(raw))
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            log.warning("Битый слепок контекста в Redis: %s", exc)
            return None

    async def save(self, call_id: str, context: ConversationContext) -> bool:
        """Пишет контекст с TTL; при ошибке возвращает False."""
        try:
            client = await self._get_client()
            await client.set(
                self._key(call_id),
                json.dumps(context.model_dump(), ensure_ascii=False),
                ex=self._ttl,
            )
            return True
        except Exception as exc:  # noqa: BLE001
            log.warning("Redis недоступен при записи контекста: %s", exc)
            return False

    async def aclose(self) -> None:
        """Закрывает клиент Redis."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None


#: Рабочее хранилище процесса. В тестах подменяется заглушкой.
context_store: ContextStore = RedisContextStore()
