"""Подбор ближайших филиалов по месту, которое назвал человек.

Координаты и расстояния считает справочник, здесь — нормализация места, ключ
пересчёта и формулировки для контекста. Филиал выбирает код: порядок приходит
из справочника и не меняется, модель озвучивает готовое.

Модуль ничего не пишет в контекст и никого не зовёт сам: он только считает и
форматирует. Кто и когда его вызывает — забота живого канала.
"""

from __future__ import annotations

import logging
from typing import Any, Mapping, Protocol, Sequence

from pydantic import BaseModel, Field

from core.config import settings

log = logging.getLogger(__name__)


class NearbyKB(Protocol):
    """Что модулю нужно от справочника."""

    async def geocode(
        self, text: str, *, city_slug: str | None = None
    ) -> tuple[float, float] | None:
        """Переводит место словами в координаты."""

    async def nearest_branches(
        self,
        lat: float,
        lon: float,
        *,
        city_slug: str | None = None,
        limit: int | None = None,
        radius_km: float | None = None,
    ) -> list[dict[str, Any]]:
        """Отдаёт ближайшие филиалы к точке."""


class NearbyResult(BaseModel):
    """Итог подбора: готовый блок для контекста и слаги филиалов.

    Attributes:
        key: ключ пересчёта — по нему видно, что подбор уже сделан.
        text: блок для контекста; либо перечень с правилами, либо строка
            о том, что место не опознано.
        branch_slugs: слаги подобранных филиалов в порядке близости.
        found: удалось ли подобрать хоть один филиал.
    """

    key: str
    text: str
    branch_slugs: list[str] = Field(default_factory=list)
    found: bool = False


def normalize_place(text: str) -> str:
    """Нормализует произнесённое место.

    Регистр, лишние пробелы и «ё» на смысл не влияют: «Солнечный» и
    « солнечный » — одно и то же место и один подбор.

    Args:
        text: место так, как его назвал человек.

    Returns:
        Нормализованная строка; пустая, если текста нет.
    """
    return " ".join((text or "").replace("ё", "е").replace("Ё", "Е").lower().split())


def nearby_key(city_slug: str | None, place: str) -> str:
    """Ключ пересчёта: город плюс нормализованное место.

    Город в ключе обязателен: одноимённые районы есть в разных городах, и без
    города подбор для «Центрального» из Перми совпал бы с красноярским.

    Args:
        city_slug: слаг города или None.
        place: место словами.

    Returns:
        Строка ключа; пустая, если места нет.
    """
    normalized = normalize_place(place)
    if not normalized:
        return ""
    return f"{city_slug or '-'}:{normalized}"


def should_refresh(*, city_slug: str | None, place: str, current_key: str) -> str | None:
    """Нужно ли пересчитывать подбор.

    Пересчитываем, только когда есть город, есть место и ключ отличается от
    того, по которому подбор уже сделан. Повтор того же места ходов не стоит.

    Без города не ходим совсем: подбор по всей сети вернул бы филиал соседнего
    города, а выдуманный адрес хуже отсутствия адреса.

    Args:
        city_slug: слаг города или None.
        place: место словами из формы разговора.
        current_key: ключ, по которому подбор уже сделан.

    Returns:
        Новый ключ, если пересчитывать надо; иначе None.
    """
    if not settings.nearest_branches_enabled:
        return None
    if not city_slug:
        return None
    key = nearby_key(city_slug, place)
    if not key or key == current_key:
        return None
    return key


def _km(value: Any) -> str:
    """Расстояние строкой с запятой в качестве разделителя."""
    try:
        return f"{float(value):.1f}".replace(".", ",")
    except (TypeError, ValueError):
        return "—"


def format_searching(place: str) -> str:
    """Блок на время подбора.

    Не готовая фраза, а правила: что можно, чего нельзя и что делать дальше.

    Args:
        place: место словами, как его назвал человек.

    Returns:
        Текст блока.
    """
    return (
        f"Ближайшие филиалы к месту «{place.strip()}» сейчас подбираются.\n"
        "Адреса не называть, пока их нет, и район заново не переспрашивать.\n"
        "Сказать про подбор одной фразой, вести разговор дальше и вернуться к "
        "филиалам, как появятся."
    )


def format_found(place: str, items: Sequence[Mapping[str, Any]]) -> str:
    """Блок с подобранными филиалами.

    Порядок пришёл из справочника и меняться не должен: он посчитан по
    координатам, а не выбран моделью.

    Args:
        place: место словами.
        items: филиалы от справочника в порядке близости.

    Returns:
        Текст блока.
    """
    lines: list[str] = [
        f"Ближайшие филиалы к месту «{place.strip()}» — посчитано по координатам, "
        "порядок не менять:"
    ]
    for number, item in enumerate(items, start=1):
        address = str(item.get("address") or "").strip()
        landmark = str(item.get("landmark") or "").strip()
        tail = f" ({landmark})" if landmark else ""
        lines.append(f"{number}. {address} — {_km(item.get('distance_km'))} км{tail}")
    lines.append("Первый — ближайший, о нём и речь. Остальные — только если первый не подошёл.")
    lines.append("Называть улицу и дом; ориентир добавляется к адресу, а не вместо него.")
    return "\n".join(lines)


def format_missing(place: str) -> str:
    """Блок, когда место не опознано или филиалов рядом нет.

    О пробеле вслух не сообщаем: просим ориентир иначе и идём дальше.

    Args:
        place: место словами.

    Returns:
        Текст блока.
    """
    return (
        f"Место «{place.strip()}» по координатам определить не удалось, филиалы по "
        "нему не подобраны.\n"
        "Адрес по этому месту не называть.\n"
        "Спросить ориентир иначе — улицей, станцией метро или известным зданием — и "
        "вести разговор дальше."
    )


async def lookup_nearby(
    kb: NearbyKB,
    *,
    city_slug: str,
    place: str,
    key: str,
) -> NearbyResult:
    """Считает подбор: координаты места, затем ближайшие филиалы.

    Второго похода не делаем, если место не опознано — считать нечего.
    Ошибки справочника наружу не летят: подбор не состоялся, разговор идёт
    дальше.

    Args:
        kb: клиент справочника.
        city_slug: слаг города.
        place: место словами из формы разговора.
        key: ключ пересчёта от ``should_refresh``.

    Returns:
        Итог подбора: блок для контекста, слаги филиалов и признак успеха.
    """
    try:
        point = await kb.geocode(place, city_slug=city_slug)
    except Exception as exc:  # noqa: BLE001
        log.warning("Геокодер не ответил по месту %r: %s", place, exc)
        point = None
    if point is None:
        log.info("Подбор филиалов: место %r не опознано, город %r", place, city_slug)
        return NearbyResult(key=key, text=format_missing(place), found=False)

    try:
        items = await kb.nearest_branches(point[0], point[1], city_slug=city_slug)
    except Exception as exc:  # noqa: BLE001
        log.warning("Подбор ближайших не ответил по месту %r: %s", place, exc)
        items = []
    if not items:
        log.info("Подбор филиалов: рядом с %r ничего нет, город %r", place, city_slug)
        return NearbyResult(key=key, text=format_missing(place), found=False)

    slugs = [str(item.get("slug") or "").strip() for item in items]
    log.info(
        "Подбор филиалов: место %r, город %r, найдено %s",
        place,
        city_slug,
        len(items),
    )
    return NearbyResult(
        key=key,
        text=format_found(place, items),
        branch_slugs=[slug for slug in slugs if slug],
        found=True,
    )
