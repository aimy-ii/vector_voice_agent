"""Источник скриптов разговора.

Интерфейс с одним методом: получить сырой скрипт по идентификатору и версии.
Первая реализация читает JSON из каталога, вторая пойдёт по HTTP в базу
заказчика. **Это единственная точка подмены, и она обязана быть бесшовной** —
всё, что выше, работает со скомпилированным объектом и про происхождение
данных не знает.

Кэш живёт на уровне процесса и ключуется парой «идентификатор + версия».
Версия обязательна: каждый ход — новый запуск на сервере, поэтому в состоянии
лежит точная версия и объект берётся из кэша по ней. Выкатили новый скрипт
посреди смены — идущие звонки доигрывают на своей.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Protocol

from core.config import settings
from script.build import CompiledScript, ScriptError, build_script
from script.models import RawScript

log = logging.getLogger(__name__)

#: Имя файла скрипта: `<идентификатор>.v<версия>.json`.
_FILE_RE = re.compile(r"^(?P<id>[a-z0-9_]+)\.v(?P<version>[a-z0-9_.-]+)\.json$")


class ScriptSource(Protocol):
    """Откуда берутся сырые скрипты."""

    def fetch(self, script_id: str, version: str | None) -> RawScript:
        """Отдаёт сырой скрипт.

        Args:
            script_id: идентификатор скрипта.
            version: версия; None — последняя доступная.

        Returns:
            Разобранные данные скрипта.

        Raises:
            ScriptError: скрипта нет или он не разбирается.
        """
        ...


class JsonScriptSource:
    """Читает скрипты из каталога с JSON-файлами."""

    def __init__(self, directory: Path | None = None) -> None:
        """Создаёт источник.

        Args:
            directory: каталог со скриптами; None — каталог из настроек.
        """
        self._dir = directory or settings.script_data_dir

    def _versions(self, script_id: str) -> list[tuple[str, Path]]:
        """Собирает доступные версии скрипта, от старой к новой."""
        found: list[tuple[str, Path]] = []
        if not self._dir.is_dir():
            return found
        for path in sorted(self._dir.iterdir()):
            match = _FILE_RE.match(path.name)
            if match and match.group("id") == script_id:
                found.append((match.group("version"), path))
        return sorted(found, key=lambda item: _version_key(item[0]))

    def fetch(self, script_id: str, version: str | None) -> RawScript:
        """Читает скрипт из файла.

        Args:
            script_id: идентификатор скрипта.
            version: версия; None — последняя в каталоге.

        Returns:
            Разобранные данные скрипта.

        Raises:
            ScriptError: файла нет, он не разбирается или версия не совпадает.
        """
        versions = self._versions(script_id)
        if not versions:
            raise ScriptError(f"Скрипт {script_id!r} не найден в {self._dir}")

        if version is None:
            picked_version, path = versions[-1]
        else:
            match = [item for item in versions if item[0] == version]
            if not match:
                available = ", ".join(v for v, _ in versions)
                raise ScriptError(
                    f"Версия {version!r} скрипта {script_id!r} не найдена. Есть: {available}"
                )
            picked_version, path = match[0]

        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ScriptError(f"Скрипт {path} не читается: {exc}") from exc

        try:
            raw = RawScript.model_validate(payload)
        except Exception as exc:  # noqa: BLE001
            raise ScriptError(f"Скрипт {path} не проходит разбор: {exc}") from exc

        if raw.version != picked_version:
            raise ScriptError(
                f"Версия внутри {path.name} ({raw.version!r}) не совпадает с именем файла "
                f"({picked_version!r})"
            )
        return raw


def _version_key(version: str) -> tuple[int | str, ...]:
    """Ключ сортировки версий: числовые части сравниваются как числа."""
    return tuple(int(part) if part.isdigit() else part for part in re.split(r"[.\-_]", version))


class ScriptRegistry:
    """Кэш скомпилированных скриптов на уровне процесса."""

    def __init__(self, source: ScriptSource | None = None) -> None:
        """Создаёт реестр.

        Args:
            source: источник сырых данных; None — чтение JSON из каталога настроек.
        """
        self._source: ScriptSource = source or JsonScriptSource()
        self._cache: dict[tuple[str, str | None], CompiledScript] = {}

    def get(self, script_id: str, version: str | None = None) -> CompiledScript:
        """Отдаёт скомпилированный скрипт, собирая его при первом обращении.

        Args:
            script_id: идентификатор скрипта.
            version: версия; None — последняя доступная.

        Returns:
            Неизменяемый скомпилированный скрипт.

        Raises:
            ScriptError: скрипт не найден или не проходит проверки.
        """
        key = (script_id, version)
        cached = self._cache.get(key)
        if cached is not None:
            return cached

        compiled = build_script(self._source.fetch(script_id, version))
        self._cache[key] = compiled
        # Тот же объект доступен и по точной версии: состояние звонка хранит
        # её, а не «последнюю», и на следующем ходу попадёт в кэш.
        self._cache[(script_id, compiled.version)] = compiled
        log.info("Скрипт %s версии %s собран", compiled.id, compiled.version)
        return compiled

    def clear(self) -> None:
        """Сбрасывает кэш. Нужен тестам и горячей перезагрузке скриптов."""
        self._cache.clear()


#: Общий реестр процесса.
registry = ScriptRegistry()
