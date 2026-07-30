"""Общие фикстуры офлайн-тестов.

Ни сети, ни ключей, ни весов: справочник, Redis, чекер и резолверы
подменяются заглушками. Скрипт по умолчанию — v2.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from graph.context_store import MemoryContextStore  # noqa: E402
from script.build import CompiledScript, build_script  # noqa: E402
from script.models import RawSalesScript, RawScript  # noqa: E402
from script.source import JsonScriptSource, ScriptRegistry  # noqa: E402
from script.store import MemoryScriptStore  # noqa: E402


@pytest.fixture(scope="session")
def data_dir() -> Path:
    """Каталог с JSON-скриптами проекта."""
    return ROOT / "src" / "script" / "data"


@pytest.fixture(scope="session")
def raw_script(data_dir: Path) -> RawScript:
    """Сырой рабочий скрипт v2."""
    return JsonScriptSource(data_dir).fetch("vector_ru", "2")


@pytest.fixture(scope="session")
def raw_script_v1(data_dir: Path) -> RawScript:
    """Сырой скрипт v1 — для проверки, что обе версии собираются."""
    return JsonScriptSource(data_dir).fetch("vector_ru", "1")


@pytest.fixture(scope="session")
def raw_script_v3(data_dir: Path) -> RawScript:
    """Сырой скрипт v3."""
    return JsonScriptSource(data_dir).fetch("vector_ru", "3")


@pytest.fixture(scope="session")
def raw_script_v4(data_dir: Path) -> RawSalesScript:
    """Сырой скрипт продаж v4."""
    raw = JsonScriptSource(data_dir).fetch("vector_ru", "4")
    assert isinstance(raw, RawSalesScript)
    return raw


@pytest.fixture()
def script(raw_script: RawScript) -> CompiledScript:
    """Скомпилированный рабочий скрипт v2."""
    return build_script(raw_script)


@pytest.fixture()
def script_v1(raw_script_v1: RawScript) -> CompiledScript:
    """Скомпилированный скрипт v1."""
    return build_script(raw_script_v1)


@pytest.fixture()
def script_v3(raw_script_v3: RawScript) -> CompiledScript:
    """Скомпилированный скрипт v3."""
    return build_script(raw_script_v3)


@pytest.fixture()
def script_v4(raw_script_v4) -> CompiledScript:
    """Скомпилированный скрипт продаж v4."""
    return build_script(raw_script_v4)


@pytest.fixture()
def registry(data_dir: Path) -> ScriptRegistry:
    """Чистый реестр скриптов с источником из каталога проекта."""
    return ScriptRegistry(JsonScriptSource(data_dir))


@pytest.fixture()
def memory_store() -> MemoryScriptStore:
    """Заглушка Redis прогресса скрипта в памяти."""
    return MemoryScriptStore()


@pytest.fixture()
def memory_context_store() -> MemoryContextStore:
    """Заглушка Redis контекста разговора в памяти."""
    return MemoryContextStore()


class FakeKB:
    """Заглушка справочника: отвечает из памяти, в сеть не ходит."""

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
        cities=[
            {"slug": "perm", "name": "Пермь"},
            {"slug": "krasnoyarsk", "name": "Красноярск"},
            {"slug": "spb", "name": "Санкт-Петербург"},
        ],
        city={
            "slug": "perm",
            "name": "Пермь",
            "branches_count": 4,
            "categories": [{"code": "B", "duration": "2,5 месяца", "start_frequency": None}],
            "vehicles": {"manual": ["Hyundai Solaris"], "automatic": ["Kia Rio"]},
            "theory_formats": ["очно", "дистанционно"],
            "documents": ["паспорт", "СНИЛС"],
            "payment": {"installment": True},
            "messengers": ["Max", "Telegram"],
            "call_hours": "09:00-21:00",
            "price": {"amount": 43900, "is_from": True, "reliable": False, "note": "оговорка"},
        },
        branches=[
            {"slug": "perm_chernyshevskogo", "address": "ул. Чернышевского, 28", "landmark": None},
            {
                "slug": "perm_ekaterininskaya",
                "address": "ул. Екатерининская, 109а",
                "landmark": "Моби Дик",
            },
            {"slug": "perm_lenina", "address": "ул. Ленина, 1", "landmark": None},
            {"slug": "perm_mira", "address": "ул. Мира, 2", "landmark": "ТЦ"},
        ],
        branch={
            "slug": "perm_chernyshevskogo",
            "address": "ул. Чернышевского, 28",
            "place_type": "учебный офис",
            "status": "работает",
            "working_hours": "ПН-ПТ 10:00-19:00",
            "landmark": None,
        },
    )
