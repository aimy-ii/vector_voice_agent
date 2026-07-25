"""Хранилище прогресса скрипта звонка в Redis.

Рабочий скрипт живёт в Redis: чекер и генератор пишут туда в разное время и
не делят тред. Промах или недоступность Redis не ломают ход — восстанавливаем
из состояния треда и работаем дальше без накопленных пометок чекера. В конце
звонка слепок складывается в тред на постоянку.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from typing import Any, Mapping, Protocol

from core.config import settings

log = logging.getLogger(__name__)

#: Префикс ключей, чтобы не пересечься с серверными ключами LangGraph.
KEY_PREFIX = "vector:script:"


#: Поля прогресса, которые пишет канал генератора.
PROGRESS_FIELDS_GENERATOR: frozenset[str] = frozenset({"attempts", "taken_turn"})
#: Поля прогресса, которые пишет канал чекера.
PROGRESS_FIELDS_CHECKER: frozenset[str] = frozenset({"status", "profile"})
#: Все поля прогресса (полная запись без слияния).
PROGRESS_FIELDS_ALL: frozenset[str] = frozenset({"status", "attempts", "taken_turn", "profile"})


@dataclass
class ScriptProgress:
    """Прогресс скрипта одного звонка.

    Attributes:
        status: статус шага ``pending`` / ``closed``.
        attempts: счётчик попыток задать шаг (раз ведущим в генерации).
        taken_turn: номер хода, когда шаг впервые ушёл в генерацию.
        profile: базовые поля профиля, накопленные чекером в кеше.
    """

    status: dict[str, str] = field(default_factory=dict)
    attempts: dict[str, int] = field(default_factory=dict)
    taken_turn: dict[str, int] = field(default_factory=dict)
    profile: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Сериализует прогресс в словарь."""
        return asdict(self)

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any] | None) -> ScriptProgress:
        """Собирает прогресс из словаря состояния или Redis.

        Args:
            data: сырой словарь или None.

        Returns:
            Экземпляр прогресса; пустой, если данных нет.
        """
        if not data:
            return cls()
        status = dict(data.get("status") or data.get("step_status") or {})
        # Старые слепки v1: done/refused/skipped → closed, open → pending.
        normalized: dict[str, str] = {}
        for key, value in status.items():
            if value in {"done", "refused", "skipped", "closed"}:
                normalized[key] = "closed"
            else:
                normalized[key] = "pending"
        profile_raw = data.get("profile") or {}
        return cls(
            status=normalized,
            attempts={
                str(k): int(v)
                for k, v in dict(data.get("attempts") or data.get("step_attempts") or {}).items()
            },
            taken_turn={str(k): int(v) for k, v in dict(data.get("taken_turn") or {}).items()},
            profile={
                str(k): str(v)
                for k, v in dict(profile_raw).items()
                if v is not None and str(v).strip()
            },
        )


def merge_progress_fields(
    base: ScriptProgress,
    overlay: ScriptProgress,
    fields: frozenset[str],
) -> ScriptProgress:
    """Накладывает выбранные поля ``overlay`` на копию ``base``.

    Args:
        base: прогресс из кеша (актуальный).
        overlay: локальные правки канала.
        fields: какие поля накладывать.

    Returns:
        Новый прогресс со слиянием на уровне ключей словарей.
    """
    merged = ScriptProgress.from_mapping(base.to_dict())
    if "status" in fields:
        merged.status = {**merged.status, **overlay.status}
    if "attempts" in fields:
        merged.attempts = {**merged.attempts, **overlay.attempts}
    if "taken_turn" in fields:
        merged.taken_turn = {**merged.taken_turn, **overlay.taken_turn}
    if "profile" in fields:
        for key, value in overlay.profile.items():
            text = str(value).strip()
            if text:
                merged.profile[key] = text
    return merged


class ScriptStore(Protocol):
    """Контракт хранилища прогресса скрипта."""

    async def load(self, call_id: str) -> ScriptProgress | None:
        """Читает прогресс по идентификатору звонка или None при промахе."""

    async def save(self, call_id: str, progress: ScriptProgress) -> bool:
        """Пишет прогресс. True — записали, False — Redis недоступен."""


class MemoryScriptStore:
    """Заглушка хранилища в памяти процесса — для офлайн-тестов."""

    def __init__(self) -> None:
        """Создаёт пустое хранилище."""
        self._data: dict[str, ScriptProgress] = {}
        self.fail: bool = False

    async def load(self, call_id: str) -> ScriptProgress | None:
        """Читает прогресс или None при промахе/сбое."""
        if self.fail:
            return None
        progress = self._data.get(call_id)
        if progress is None:
            return None
        return ScriptProgress.from_mapping(progress.to_dict())

    async def save(self, call_id: str, progress: ScriptProgress) -> bool:
        """Пишет прогресс; при ``fail`` притворяется, что Redis недоступен."""
        if self.fail:
            return False
        self._data[call_id] = ScriptProgress.from_mapping(progress.to_dict())
        return True


class RedisScriptStore:
    """Хранилище прогресса в Redis с собственным префиксом и TTL."""

    def __init__(self, url: str | None = None, *, ttl_seconds: int | None = None) -> None:
        """Создаёт клиент Redis.

        Args:
            url: адрес Redis; пусто — из настроек.
            ttl_seconds: TTL ключа; пусто — из настроек.
        """
        self._url = url or settings.redis_url
        self._ttl = ttl_seconds if ttl_seconds is not None else settings.script_redis_ttl
        self._client: Any = None

    def _key(self, call_id: str) -> str:
        """Ключ прогресса по идентификатору звонка."""
        return f"{KEY_PREFIX}{call_id}"

    async def _get_client(self) -> Any:
        """Лениво открывает async-клиент Redis."""
        if self._client is None:
            from redis.asyncio import Redis

            self._client = Redis.from_url(self._url, decode_responses=True)
        return self._client

    async def load(self, call_id: str) -> ScriptProgress | None:
        """Читает прогресс; при любой ошибке — None, ход не роняем."""
        try:
            client = await self._get_client()
            raw = await client.get(self._key(call_id))
        except Exception as exc:  # noqa: BLE001
            log.warning("Redis недоступен при чтении скрипта: %s", exc)
            return None
        if not raw:
            return None
        try:
            return ScriptProgress.from_mapping(json.loads(raw))
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            log.warning("Битый слепок скрипта в Redis: %s", exc)
            return None

    async def save(self, call_id: str, progress: ScriptProgress) -> bool:
        """Пишет прогресс с TTL; при ошибке возвращает False."""
        try:
            client = await self._get_client()
            await client.set(
                self._key(call_id),
                json.dumps(progress.to_dict(), ensure_ascii=False),
                ex=self._ttl,
            )
            return True
        except Exception as exc:  # noqa: BLE001
            log.warning("Redis недоступен при записи скрипта: %s", exc)
            return False

    async def aclose(self) -> None:
        """Закрывает клиент Redis."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None


def progress_from_state(state: Mapping[str, Any]) -> ScriptProgress:
    """Восстанавливает прогресс из полей состояния треда.

    Args:
        state: состояние звонка.

    Returns:
        Прогресс без пометок, которых в треде нет.
    """
    snapshot = state.get("script_progress")
    if isinstance(snapshot, Mapping) and snapshot:
        return ScriptProgress.from_mapping(snapshot)
    return ScriptProgress.from_mapping(
        {
            "status": state.get("step_status") or {},
            "attempts": state.get("step_attempts") or {},
            "taken_turn": state.get("step_taken_turn") or {},
            "profile": state.get("profile") or {},
        }
    )


def progress_to_state(progress: ScriptProgress) -> dict[str, Any]:
    """Кладёт прогресс в поля состояния треда (зеркало и слепок).

    Args:
        progress: текущий прогресс.

    Returns:
        Правки для ``CallState``. Профиль сюда не кладём — его сливают
        вызывающие узлы, чтобы не затереть поля, которых нет в кеше.
    """
    payload = progress.to_dict()
    return {
        "step_status": dict(progress.status),
        "step_attempts": dict(progress.attempts),
        "step_taken_turn": dict(progress.taken_turn),
        "script_progress": payload,
    }


#: Рабочее хранилище процесса. В тестах подменяется заглушкой.
script_store: ScriptStore = RedisScriptStore()
