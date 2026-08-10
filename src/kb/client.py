"""Клиент справочника Vector KB для голосового агента.

Тонкая обёртка над HTTP-API: агент спрашивает город и филиал, получает готовые
факты для реплики. Сервис стоит на той же машине, поэтому сетевых задержек нет,
но запрос в цикле хода всё равно нежелателен — поэтому справочник кэшируется
в памяти процесса, а по сети ходим только при промахе кэша.

Устройство разговора:

1. `list_cities` отдаёт города парами «слаг — название». Список уходит модели,
   она выбирает слаг сама. Расшифровки обязательны: по одним слагам город не
   опознать, `kyrgan` это Курган, `tagil` — Нижний Тагил, `novosib` — Новосибирск.
2. Код кладёт выбранный слаг в состояние звонка и забирает по нему мету города.
3. Когда клиент назовёт улицу или район, `list_branches` отдаёт филиалы этого
   города — слаг, адрес и ориентир. Модель выбирает слаг филиала, код забирает
   по нему мету.

Ошибки наружу не выбрасываются: при недоступности сервиса методы возвращают
None или пустой список, а агент продолжает разговор без справочных данных.
Падать посреди звонка из-за HTTP хуже, чем ответить общими словами.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

import httpx

from core.config import settings

# --- Настройки интеграции -----------------------------------------------------

#: Адрес сервиса. В vector-infra — имя сервиса в общей сети; локальный дефолт
#: настроек — ``host.docker.internal`` (справочник на порту хоста).
VECTOR_KB_BASE_URL = settings.vector_kb_url

#: Префикс всех ручек API.
API_PREFIX = "/api"

#: Таймаут одного запроса, секунды. Держим коротким: сервис локальный,
#: а долгое ожидание в голосовом диалоге ощущается как зависание.
REQUEST_TIMEOUT = settings.vector_kb_timeout

#: Сколько раз повторить запрос при сетевой ошибке.
REQUEST_RETRIES = 2

#: Пауза между повторами, секунды.
RETRY_DELAY = 0.2

#: Сколько живёт кэш справочника, секунды. Данные меняются только после
#: парсинга, поэтому час — с запасом.
CACHE_TTL = settings.vector_kb_cache_ttl

#: Максимум записей в кэше. Кэш общий на все ручки: три ключа на город плюс
#: меты филиалов плюс подбор по месту — сотни не хватало, старые записи
#: вытеснялись раньше, чем протухали.
CACHE_MAX_CITIES = 512

#: Префикс логов интеграции.
LOG_PREFIX = "[VECTOR_KB]"

# --- Пути ручек ---------------------------------------------------------------

PATH_HEALTH = "/health"
PATH_CITIES = "/cities"
PATH_CITIES_ENUM = "/cities/enum"
PATH_CITIES_RESOLVE = "/cities/resolve"
PATH_CITY = "/cities/{city_slug}"
PATH_CITY_BRANCHES = "/cities/{city_slug}/branches"
PATH_CITY_BRANCHES_ENUM = "/cities/{city_slug}/branches/enum"
PATH_BRANCH = "/branches/{branch_slug}"
PATH_BRANCHES_NEAREST = "/branches/nearest"
PATH_GEOCODE = "/geocode"
PATH_PARSE = "/parse"
PATH_PARSE_JOB = "/parse/{job_id}"
PATH_RELOAD = "/reload"

# --- Что бот обязан произнести ------------------------------------------------

#: Ключ оговорки о цене в мете города. Если в ответе есть сумма, оговорку
#: из этого поля нужно произнести вместе с ней: число на сайте — маркетинговое
#: «от» и заметно ниже реального чека.
PRICE_NOTE_KEY = "note"

#: Ключ признака подтверждённости цены. Пока False — цена не проверена.
PRICE_RELIABLE_KEY = "reliable"

logger = logging.getLogger(__name__)


def _place_cache_key(text: str) -> str:
    """Нормализует произнесённое место для ключа кэша.

    Регистр, лишние пробелы и «ё» на ключ влиять не должны: «Солнечный» и
    « солнечный » — одна и та же точка и один поход к геокодеру.

    В справочник уходит исходный текст: геокодеру полезна живая формулировка,
    нормализация нужна только ключу.

    Args:
        text: место так, как его назвал человек.

    Returns:
        Нормализованная строка; пустая, если текста нет.
    """
    return " ".join((text or "").replace("ё", "е").replace("Ё", "Е").lower().split())


class VectorKBClient:
    """Клиент справочника автошколы для голосового агента.

    Держит одно HTTP-соединение и кэш справочника в памяти. Инициализация
    ленивая: соединение создаётся при первом обращении, закрывается через
    `close` при остановке воркера.

    HTTP-клиент создаётся с `trust_env=False`: справочник — локальный сервис
    внутри сети compose, прокси из окружения предназначен только для внешних
    сервисов (OpenAI, ElevenLabs) и сюда применяться не должен.

    Пример:

        kb = VectorKBClient()

        cities = await kb.list_cities()          # список для модели
        city = await kb.get_city("perm")         # модель вернула слаг
        branches = await kb.list_branches("perm")
        branch = await kb.get_branch("perm_chernyshevskogo")
    """

    def __init__(
        self,
        base_url: str = VECTOR_KB_BASE_URL,
        timeout: float = REQUEST_TIMEOUT,
        cache_ttl: int = CACHE_TTL,
    ) -> None:
        """Создаёт клиент без сетевых обращений.

        Args:
            base_url: адрес сервиса справочника.
            timeout: таймаут одного запроса в секундах.
            cache_ttl: время жизни кэша в секундах.
        """
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._cache_ttl = cache_ttl
        self._http: httpx.AsyncClient | None = None
        self._cache: dict[str, tuple[float, Any]] = {}
        self._lock = asyncio.Lock()

    # --- соединение -----------------------------------------------------------

    async def _init_client(self) -> httpx.AsyncClient:
        """Создаёт HTTP-клиент при первом обращении.

        `trust_env=False` отключает чтение прокси из переменных окружения:
        справочник доступен напрямую внутри сети compose.
        """
        if self._http is None:
            async with self._lock:
                if self._http is None:
                    self._http = httpx.AsyncClient(
                        base_url=f"{self._base_url}{API_PREFIX}",
                        timeout=self._timeout,
                        trust_env=False,
                    )
        return self._http

    async def close(self) -> None:
        """Закрывает соединение и чистит кэш. Вызывать при остановке воркера."""
        if self._http is not None:
            await self._http.aclose()
            self._http = None
        self._cache.clear()

    async def __aenter__(self) -> VectorKBClient:
        """Вход в асинхронный контекстный менеджер."""
        await self._init_client()
        return self

    async def __aexit__(self, *args: object) -> None:
        """Выход из контекстного менеджера: закрывает соединение."""
        await self.close()

    # --- кэш ------------------------------------------------------------------

    def _cached(self, key: str) -> Any | None:
        """Достаёт живое значение из кэша или None, если протухло."""
        entry = self._cache.get(key)
        if entry is None:
            return None
        stored_at, value = entry
        if time.monotonic() - stored_at > self._cache_ttl:
            self._cache.pop(key, None)
            return None
        return value

    def _store(self, key: str, value: Any) -> None:
        """Кладёт значение в кэш, вытесняя самое старое при переполнении."""
        if len(self._cache) >= CACHE_MAX_CITIES:
            oldest = min(self._cache, key=lambda k: self._cache[k][0])
            self._cache.pop(oldest, None)
        self._cache[key] = (time.monotonic(), value)

    def invalidate_cache(self) -> None:
        """Сбрасывает кэш. Вызывать после обновления данных парсингом."""
        self._cache.clear()
        logger.info("%s Кэш справочника сброшен", LOG_PREFIX)

    # --- транспорт ------------------------------------------------------------

    async def _request(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        json_body: dict[str, Any] | None = None,
        timeout: float | None = None,
        retries: int | None = None,
    ) -> Any | None:
        """Выполняет запрос с повторами и мягкой обработкой ошибок.

        Args:
            method: HTTP-метод.
            path: путь относительно префикса API.
            params: параметры строки запроса.
            json_body: тело запроса.
            timeout: таймаут этого запроса, секунды. None — общий таймаут
                клиента. Нужен геокодеру: он единственный ходит наружу.
            retries: сколько попыток делать. None — общее значение
                ``REQUEST_RETRIES``.

        Returns:
            Разобранный JSON или None при ошибке и при ответе 404.
        """
        client = await self._init_client()
        last: Exception | None = None
        attempts = REQUEST_RETRIES if retries is None else max(1, retries)
        # httpx понимает timeout=None как «ждать вечно», поэтому параметр
        # передаём только когда он задан — иначе работает таймаут клиента.
        extra: dict[str, Any] = {}
        if timeout is not None:
            extra["timeout"] = timeout

        for attempt in range(1, attempts + 1):
            try:
                response = await client.request(
                    method, path, params=params, json=json_body, **extra
                )
                body_size = len(response.content)
                logger.info(
                    "%s Запрос %s %s → %s, тело %s байт",
                    LOG_PREFIX,
                    method,
                    path,
                    response.status_code,
                    body_size,
                )
                if response.status_code == 404:
                    logger.info("%s Не найдено: %s %s", LOG_PREFIX, method, path)
                    return None
                response.raise_for_status()
                return response.json()
            except httpx.HTTPStatusError as exc:
                logger.warning(
                    "%s Ответ %s на %s %s",
                    LOG_PREFIX,
                    exc.response.status_code,
                    method,
                    path,
                )
                logger.info(
                    "%s Ошибка запроса %s %s: HTTP %s",
                    LOG_PREFIX,
                    method,
                    path,
                    exc.response.status_code,
                )
                return None
            except httpx.HTTPError as exc:
                last = exc
                logger.info(
                    "%s Ошибка запроса %s %s: %s",
                    LOG_PREFIX,
                    method,
                    path,
                    exc,
                )
                if attempt < attempts:
                    await asyncio.sleep(RETRY_DELAY * attempt)

        logger.error("%s Сервис недоступен: %s %s (%s)", LOG_PREFIX, method, path, last)
        return None

    # --- живость --------------------------------------------------------------

    async def health(self) -> dict[str, Any] | None:
        """Проверяет живость сервиса и объём загруженных данных."""
        return await self._request("GET", PATH_HEALTH)

    async def is_ready(self) -> bool:
        """Готов ли справочник отвечать: сервис жив и города загружены."""
        data = await self.health()
        return bool(data and data.get("cities_count"))

    # --- города ---------------------------------------------------------------

    async def list_cities(self) -> list[dict[str, Any]]:
        """Отдаёт города с названиями и числом филиалов.

        Returns:
            Список записей со слагом, названием и числом филиалов.
        """
        cached = self._cached("cities")
        if cached is not None:
            return cached
        data = await self._request("GET", PATH_CITIES)
        result = data if isinstance(data, list) else []
        if result:
            self._store("cities", result)
        return result

    async def cities_enum(self) -> list[str]:
        """Отдаёт плоский список слагов городов.

        Это опора выбора города: список уходит модели перечислением, поэтому
        вернувшийся слаг существует по построению и промахнуться нельзя.

        Returns:
            Список слагов; пустой, если сервис недоступен.
        """
        cached = self._cached("cities_enum")
        if cached is not None:
            return cached
        data = await self._request("GET", PATH_CITIES_ENUM)
        result = data if isinstance(data, list) else []
        if result:
            self._store("cities_enum", result)
        return result

    async def resolve_city(self, text: str) -> str | None:
        """Пробует опознать город по разговорному названию.

        Быстрый путь, не опора: сервис сравнивает точное название и небольшую
        таблицу разговорных вариантов, падежей не понимает. «Пермь» разберётся,
        «из Перми» — нет. При промахе город выбирает модель из `cities_enum`.

        Args:
            text: реплика или её кусок с названием города.

        Returns:
            Слаг города или None, если совпадения нет.
        """
        if not text:
            return None
        data = await self._request("GET", PATH_CITIES_RESOLVE, params={"text": text})
        if isinstance(data, dict):
            slug = data.get("slug")
            return slug if isinstance(slug, str) else None
        return None

    async def get_city(self, city_slug: str) -> dict[str, Any] | None:
        """Отдаёт всё, что известно о городе, для пересказа клиенту.

        В ответе: число филиалов и автодромов, категории со сроками, автопарк,
        форматы теории, документы, условия оплаты, частые вопросы, контакты
        и цена с обязательной оговоркой.

        Args:
            city_slug: слаг города, который вернула модель.

        Returns:
            Мета города или None, если города нет или сервис недоступен.
        """
        if not city_slug:
            return None
        key = f"city:{city_slug}"
        cached = self._cached(key)
        if cached is not None:
            return cached
        data = await self._request("GET", PATH_CITY.format(city_slug=city_slug))
        if isinstance(data, dict):
            self._store(key, data)
            return data
        return None

    # --- филиалы --------------------------------------------------------------

    async def list_branches(self, city_slug: str) -> list[dict[str, Any]]:
        """Отдаёт филиалы города: слаг, адрес и ориентир.

        Ориентир — название здания или комплекса вроде «ТРК Нарва». Клиенты
        ориентируются по нему охотнее, чем по номеру дома, поэтому в реплике
        его стоит называть, когда он есть.

        Args:
            city_slug: слаг города.

        Returns:
            Список филиалов; пустой, если города нет или сервис недоступен.
        """
        if not city_slug:
            return []
        key = f"branches:{city_slug}"
        cached = self._cached(key)
        if cached is not None:
            return cached
        data = await self._request("GET", PATH_CITY_BRANCHES.format(city_slug=city_slug))
        result = data if isinstance(data, list) else []
        if result:
            self._store(key, result)
        return result

    async def branches_enum(self, city_slug: str) -> list[str]:
        """Отдаёт плоский список слагов филиалов города.

        Нужен там же, где `cities_enum`: модель выбирает филиал из готового
        перечисления, и промахнуться мимо существующего слага нельзя.

        Args:
            city_slug: слаг города.

        Returns:
            Список слагов филиалов; пустой, если города нет или сервис недоступен.
        """
        if not city_slug:
            return []
        key = f"branches_enum:{city_slug}"
        cached = self._cached(key)
        if cached is not None:
            return cached
        data = await self._request("GET", PATH_CITY_BRANCHES_ENUM.format(city_slug=city_slug))
        result = data if isinstance(data, list) else []
        if result:
            self._store(key, result)
        return result

    # --- подбор по месту ------------------------------------------------------

    async def geocode(
        self,
        text: str,
        *,
        city_slug: str | None = None,
    ) -> tuple[float, float] | None:
        """Переводит произнесённое место в координаты.

        Единственный вызов справочника, который ходит наружу — к геокодеру.
        Поэтому таймаут короткий и попытка одна: ход столько ждать не должен,
        а фон переспросит на следующей реплике.

        Отрицательный ответ тоже кэшируется: если место не опознано, второй
        поход за той же фразой ничего не изменит.

        Args:
            text: место словами («Солнечный», «у Гражданского проспекта»).
            city_slug: слаг города; сужает поиск и разводит одноимённые
                районы в разных городах.

        Returns:
            Пара «широта, долгота» или None, если место не опознано либо
            справочник недоступен.
        """
        place = _place_cache_key(text)
        if len(place) < 2:
            return None
        key = f"nearby:{city_slug or '-'}:{place}"
        cached = self._cached(key)
        if cached is not None:
            return (cached[0], cached[1]) if cached else None

        params: dict[str, Any] = {"text": (text or "").strip()}
        if city_slug:
            params["city_slug"] = city_slug
        data = await self._request(
            "GET",
            PATH_GEOCODE,
            params=params,
            timeout=settings.geocode_timeout_seconds,
            retries=1,
        )
        point: list[float] = []
        if isinstance(data, dict) and data.get("found"):
            lat = data.get("lat")
            lon = data.get("lon")
            if isinstance(lat, (int, float)) and isinstance(lon, (int, float)):
                point = [float(lat), float(lon)]
        if data is not None:
            self._store(key, point)
        return (point[0], point[1]) if point else None

    async def nearest_branches(
        self,
        lat: float,
        lon: float,
        *,
        city_slug: str | None = None,
        limit: int | None = None,
        radius_km: float | None = None,
    ) -> list[dict[str, Any]]:
        """Отдаёт ближайшие к точке филиалы, отсортированные по расстоянию.

        Сети на стороне справочника здесь нет — это арифметика по уже
        проставленным координатам.

        Args:
            lat: широта точки.
            lon: долгота точки.
            city_slug: слаг города; без него подбор идёт по всей сети и может
                вернуть филиал соседнего города.
            limit: сколько филиалов вернуть; None — значение из настроек.
            radius_km: радиус подбора; None — значение из настроек.

        Returns:
            Список филиалов со слагом, адресом, ориентиром и расстоянием;
            пустой, если в радиусе никого нет или справочник недоступен.
        """
        take = settings.nearest_branches_limit if limit is None else limit
        radius = settings.nearest_branches_radius_km if radius_km is None else radius_km
        key = f"nearest:{city_slug or '-'}:{lat:.4f}:{lon:.4f}:{take}:{radius}"
        cached = self._cached(key)
        if cached is not None:
            return cached

        params: dict[str, Any] = {
            "lat": lat,
            "lon": lon,
            "limit": take,
            "radius_km": radius,
        }
        if city_slug:
            params["city_slug"] = city_slug
        data = await self._request("GET", PATH_BRANCHES_NEAREST, params=params)
        result = data if isinstance(data, list) else []
        if result:
            self._store(key, result)
        return result

    async def get_branch(self, branch_slug: str) -> dict[str, Any] | None:
        """Отдаёт всё, что известно о филиале.

        Поле `status` бывает «работает» и «скоро открытие». У неоткрытых часы
        работы пустые: записывать туда нельзя, но сказать, что филиал скоро
        откроется, можно и нужно.

        Поле `place_type` бывает «учебный офис» и «автодром». На автодром
        приезжать за договором не надо — это стоит проговаривать.

        Args:
            branch_slug: слаг филиала, уникальный на всю сеть.

        Returns:
            Мета филиала или None.
        """
        if not branch_slug:
            return None
        key = f"branch:{branch_slug}"
        cached = self._cached(key)
        if cached is not None:
            return cached
        data = await self._request("GET", PATH_BRANCH.format(branch_slug=branch_slug))
        if isinstance(data, dict):
            self._store(key, data)
            return data
        return None

    # --- обновление данных ----------------------------------------------------

    async def start_parse(
        self,
        *,
        only: list[str] | None = None,
        force: bool = False,
        include_external: bool = False,
        include_done: bool = False,
    ) -> dict[str, Any] | None:
        """Запускает обновление справочника в фоне.

        Полный обход занимает около десяти минут. Параллельно выполняется не
        больше одной задачи: при попытке запустить вторую сервис ответит 409,
        и метод вернёт None.

        Флаг `include_done` по умолчанию выключен: Санкт-Петербург,
        Екатеринбург и Пермь собраны вручную, их пересборка ухудшит данные.

        Args:
            only: слаги городов для частичного обновления.
            force: перекачивать страницы, игнорируя кэш.
            include_external: включать города на отдельных доменах.
            include_done: пересобирать города, собранные вручную.

        Returns:
            Словарь с `job_id` и статусом или None при отказе.
        """
        body = {
            "only": only,
            "force": force,
            "include_external": include_external,
            "include_done": include_done,
        }
        return await self._request("POST", PATH_PARSE, json_body=body)

    async def parse_status(self, job_id: str) -> dict[str, Any] | None:
        """Отдаёт состояние фоновой задачи парсинга.

        Args:
            job_id: идентификатор из `start_parse`.

        Returns:
            Состояние задачи или None, если задача не найдена.
        """
        if not job_id:
            return None
        return await self._request("GET", PATH_PARSE_JOB.format(job_id=job_id))

    async def reload(self) -> dict[str, Any] | None:
        """Просит сервис перечитать файлы справочника в память.

        После успешного вызова локальный кэш клиента сбрасывается, иначе агент
        продолжил бы отвечать по старым данным.

        Returns:
            Ответ сервиса с числом загруженных городов или None.
        """
        data = await self._request("POST", PATH_RELOAD)
        if data is not None:
            self.invalidate_cache()
        return data


#: Общий экземпляр клиента для агента. Создаётся без сетевых обращений,
#: соединение поднимается при первом запросе.
vector_kb = VectorKBClient()
