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

Реализаций две. ``JsonScriptSource`` читает файлы из образа, и это
по-прежнему поведение по умолчанию. ``HttpScriptSource`` ходит в админку
справочника и при любой её беде откатывается на те же файлы: админка —
надстройка, из-за неё звонок падать не должен. Какая работает, решает
``SCRIPT_SOURCE`` в ``.env``.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from time import monotonic
from typing import Protocol

import httpx

from core.config import settings
from script.build import CompiledScript, ScriptError, build_script
from script.models import RawSalesScript, RawScript, is_sales_payload

log = logging.getLogger(__name__)

#: Имя файла скрипта: `<идентификатор>.v<версия>.json`.
_FILE_RE = re.compile(r"^(?P<id>[a-z0-9_]+)\.v(?P<version>[a-z0-9_.-]+)\.json$")

#: Сырой скрипт: старый формат или продажи.
RawAnyScript = RawScript | RawSalesScript


class ScriptSource(Protocol):
    """Откуда берутся сырые скрипты."""

    def fetch(self, script_id: str, version: str | None) -> RawAnyScript:
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

    def fetch(self, script_id: str, version: str | None) -> RawAnyScript:
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

        if not isinstance(payload, dict):
            raise ScriptError(f"Скрипт {path} должен быть JSON-объектом")

        try:
            if is_sales_payload(payload):
                raw: RawAnyScript = RawSalesScript.model_validate(payload)
            else:
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


class HttpScriptSource:
    """Читает скрипты из админки справочника по HTTP.

    Сеть в цепочке звонка — это способ его уронить, поэтому здесь нет ни
    одного пути, ведущего к исключению наружу: сервис не ответил, ответил
    не то, ответил битым JSON — берём файл из образа и ведём разговор
    дальше. Админка может лежать сколько угодно, звонки идут.

    Отдельный случай — версия, закреплённая за идущим звонком. Если её нет
    в файлах (её опубликовали из админки уже после сборки образа), берём
    последнюю файловую и громко пишем об этом в лог: разговор на чуть
    другом скрипте лучше оборванного разговора.
    """

    #: Путь ручки документа. Совпадает с контрактом справочника.
    PATH = "/api/scripts/{script_id}"

    def __init__(
        self,
        base_url: str | None = None,
        fallback: ScriptSource | None = None,
        timeout: float | None = None,
    ) -> None:
        """Создаёт источник.

        Args:
            base_url: адрес справочника; None — из настроек.
            fallback: источник на случай недоступности; None — файлы.
            timeout: таймаут запроса; None — из настроек.
        """
        self._base_url = (base_url or settings.vector_kb_url).rstrip("/")
        self._fallback: ScriptSource = fallback or JsonScriptSource()
        self._timeout = timeout if timeout is not None else settings.script_http_timeout

    def fetch(self, script_id: str, version: str | None) -> RawAnyScript:
        """Отдаёт сырой скрипт из админки, при любой беде — из файла.

        Args:
            script_id: идентификатор скрипта.
            version: версия; None — опубликованная.

        Returns:
            Разобранные данные скрипта.

        Raises:
            ScriptError: скрипта нет ни в админке, ни в файлах.
        """
        payload = self._request(script_id, version)
        if payload is not None:
            try:
                return _parse_payload(payload)
            except ScriptError as exc:
                log.warning("Админка отдала скрипт, который не разбирается: %s", exc)
        return self._from_files(script_id, version)

    def _request(self, script_id: str, version: str | None) -> dict[str, object] | None:
        """Ходит в админку за документом.

        Args:
            script_id: идентификатор скрипта.
            version: версия или None.

        Returns:
            Разобранный JSON или None, если админка не ответила.
        """
        url = self._base_url + self.PATH.format(script_id=script_id)
        params = {"version": version} if version else None
        try:
            response = httpx.get(url, params=params, timeout=self._timeout)
        except httpx.HTTPError as exc:
            log.warning("Админка недоступна (%s), скрипт берём из файла", exc)
            return None

        if response.status_code == 404:
            log.warning(
                "В админке нет скрипта %s версии %s, берём из файла",
                script_id,
                version or "опубликованной",
            )
            return None
        if response.status_code != 200:
            log.warning(
                "Админка ответила HTTP %s на скрипт %s, берём из файла",
                response.status_code,
                script_id,
            )
            return None

        try:
            payload = response.json()
        except ValueError as exc:
            log.warning("Админка отдала не JSON (%s), скрипт берём из файла", exc)
            return None
        if not isinstance(payload, dict):
            log.warning("Админка отдала не объект, скрипт берём из файла")
            return None
        return payload

    def _from_files(self, script_id: str, version: str | None) -> RawAnyScript:
        """Читает скрипт из файлов, подставляя последнюю версию вместо пропавшей.

        Args:
            script_id: идентификатор скрипта.
            version: закреплённая версия или None.

        Returns:
            Разобранные данные скрипта.

        Raises:
            ScriptError: скрипта нет и в файлах.
        """
        try:
            return self._fallback.fetch(script_id, version)
        except ScriptError:
            if version is None:
                raise
            log.error(
                "Версии %s скрипта %s нет в файлах образа: звонок доигрывает "
                "на последней файловой версии",
                version,
                script_id,
            )
            return self._fallback.fetch(script_id, None)


def _parse_payload(payload: dict[str, object]) -> RawAnyScript:
    """Разбирает документ админки теми же моделями, что и файл.

    Args:
        payload: тело ответа справочника.

    Returns:
        Разобранные данные скрипта.

    Raises:
        ScriptError: документ не проходит разбор.
    """
    try:
        if is_sales_payload(payload):
            return RawSalesScript.model_validate(payload)
        return RawScript.model_validate(payload)
    except Exception as exc:  # noqa: BLE001
        raise ScriptError(f"Документ админки не проходит разбор: {exc}") from exc


def build_source() -> ScriptSource:
    """Собирает источник скриптов по настройке ``SCRIPT_SOURCE``.

    Returns:
        Источник: файлы образа либо админка с откатом на файлы.
    """
    if settings.script_source.strip().lower() == "http":
        log.info("Источник скриптов: админка справочника %s", settings.vector_kb_url)
        return HttpScriptSource()
    return JsonScriptSource()


class ScriptRegistry:
    """Кэш скомпилированных скриптов на уровне процесса.

    Точная версия кэшируется навсегда: опубликованная версия неизменяема,
    правки идут через новый номер. «Последняя» — та, что запрошена без
    версии, — перепроверяется по таймеру: иначе процесс, поднявшийся до
    публикации, до перезапуска раздавал бы старый скрипт новым звонкам.

    Идущих звонков перепроверка не касается: у них версия закреплена в
    состоянии, и они попадают в кэш по точному ключу.
    """

    def __init__(
        self, source: ScriptSource | None = None, *, latest_ttl: int | None = None
    ) -> None:
        """Создаёт реестр.

        Args:
            source: источник сырых данных; None — по настройке ``SCRIPT_SOURCE``.
            latest_ttl: сколько секунд «последняя» версия считается свежей;
                None — из настроек.
        """
        self._source: ScriptSource = source or build_source()
        self._cache: dict[tuple[str, str | None], CompiledScript] = {}
        #: Когда последний раз спрашивали «последнюю» версию скрипта.
        self._latest_at: dict[str, float] = {}
        self._latest_ttl = latest_ttl if latest_ttl is not None else settings.script_latest_ttl

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
        if version is not None:
            cached = self._cache.get((script_id, version))
            if cached is not None:
                return cached
        elif not self._latest_expired(script_id):
            cached = self._cache.get((script_id, None))
            if cached is not None:
                return cached

        raw = self._source.fetch(script_id, version)
        if version is None:
            self._latest_at[script_id] = monotonic()

        compiled = self._cache.get((script_id, raw.version))
        if compiled is None:
            compiled = build_script(raw)
            self._cache[(script_id, compiled.version)] = compiled
            log.info("Скрипт %s версии %s собран", compiled.id, compiled.version)

        # Тот же объект доступен и по точной версии: состояние звонка хранит
        # её, а не «последнюю», и на следующем ходу попадёт в кэш.
        self._cache[(script_id, version)] = compiled
        return compiled

    def _latest_expired(self, script_id: str) -> bool:
        """Пора ли перепроверить, какая версия скрипта сейчас последняя."""
        if self._latest_ttl <= 0:
            return True
        asked_at = self._latest_at.get(script_id)
        return asked_at is None or monotonic() - asked_at >= self._latest_ttl

    def clear(self) -> None:
        """Сбрасывает кэш. Нужен тестам и горячей перезагрузке скриптов."""
        self._cache.clear()
        self._latest_at.clear()


#: Общий реестр процесса.
registry = ScriptRegistry()
