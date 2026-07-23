"""Общие фикстуры офлайн-тестов.

Ни сети, ни ключей, ни весов: справочник и модель подменяются заглушками,
скрипт читается из данных проекта. Логика диалога проверяется на чистых
функциях, голосовой слой не участвует вовсе.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from script.build import CompiledScript, build_script  # noqa: E402
from script.models import RawScript  # noqa: E402
from script.source import JsonScriptSource, ScriptRegistry  # noqa: E402


@pytest.fixture(scope="session")
def data_dir() -> Path:
    """Каталог с JSON-скриптами проекта."""
    return ROOT / "src" / "script" / "data"


@pytest.fixture(scope="session")
def raw_script(data_dir: Path) -> RawScript:
    """Сырой базовый скрипт из данных проекта."""
    return JsonScriptSource(data_dir).fetch("vector_ru", None)


@pytest.fixture()
def script(raw_script: RawScript) -> CompiledScript:
    """Скомпилированный базовый скрипт."""
    return build_script(raw_script)


@pytest.fixture()
def registry(data_dir: Path) -> ScriptRegistry:
    """Чистый реестр скриптов с источником из каталога проекта."""
    return ScriptRegistry(JsonScriptSource(data_dir))


class FakeKB:
    """Заглушка справочника: отвечает из памяти, в сеть не ходит.

    Повторяет контракт клиента ровно в том объёме, который использует граф,
    включая мягкое поведение при отсутствии данных: None и пустые списки
    вместо исключений.
    """

    def __init__(
        self,
        *,
        cities: list[dict[str, Any]] | None = None,
        city: dict[str, Any] | None = None,
        branches: list[dict[str, Any]] | None = None,
        branch: dict[str, Any] | None = None,
    ) -> None:
        """Создаёт заглушку с заранее заданными ответами."""
        self._cities = cities if cities is not None else [{"slug": "perm", "name": "Пермь"}]
        self._city = city
        self._branches = branches if branches is not None else []
        self._branch = branch
        self.calls: list[str] = []

    async def list_cities(self) -> list[dict[str, Any]]:
        """Список городов со слагами и названиями."""
        self.calls.append("list_cities")
        return self._cities

    async def cities_enum(self) -> list[str]:
        """Плоское перечисление слагов городов."""
        self.calls.append("cities_enum")
        return [c["slug"] for c in self._cities]

    async def resolve_city(self, text: str) -> str | None:
        """Точное совпадение названия города, без падежей."""
        self.calls.append("resolve_city")
        needle = text.strip().lower().replace("ё", "е")
        for city in self._cities:
            if city["name"].strip().lower().replace("ё", "е") == needle:
                return city["slug"]
        return None

    async def get_city(self, city_slug: str) -> dict[str, Any] | None:
        """Мета города."""
        self.calls.append("get_city")
        return self._city

    async def list_branches(self, city_slug: str) -> list[dict[str, Any]]:
        """Филиалы города."""
        self.calls.append("list_branches")
        return self._branches

    async def branches_enum(self, city_slug: str) -> list[str]:
        """Плоское перечисление слагов филиалов."""
        self.calls.append("branches_enum")
        return [b["slug"] for b in self._branches]

    async def get_branch(self, branch_slug: str) -> dict[str, Any] | None:
        """Мета филиала."""
        self.calls.append("get_branch")
        return self._branch


@pytest.fixture()
def fake_kb() -> FakeKB:
    """Заглушка справочника с одним городом и двумя филиалами."""
    return FakeKB(
        cities=[{"slug": "perm", "name": "Пермь"}, {"slug": "krasnoyarsk", "name": "Красноярск"}],
        city={
            "slug": "perm",
            "name": "Пермь",
            "branches_count": 4,
            "categories": [{"code": "B", "duration": "2,5 месяца", "start_frequency": None}],
            "vehicles": {"manual": ["Hyundai Solaris"], "automatic": ["Kia Rio"]},
            "theory_formats": ["очно", "дистанционно"],
            "price": {"amount": 43900, "is_from": True, "reliable": False, "note": "оговорка"},
        },
        branches=[
            {"slug": "perm_chernyshevskogo", "address": "ул. Чернышевского, 28", "landmark": None},
            {
                "slug": "perm_ekaterininskaya",
                "address": "ул. Екатерининская, 109а",
                "landmark": "Моби Дик",
            },
        ],
        branch={
            "slug": "perm_chernyshevskogo",
            "address": "ул. Чернышевского, 28",
            "place_type": "учебный офис",
            "status": "работает",
            "working_hours": "ПН-ПТ 10:00-19:00",
        },
    )
