"""Скрипт из админки: подмена источника должна быть незаметной.

Главная проверка здесь одна: скомпилированный скрипт, собранный из ответа
админки, обязан совпасть со скриптом из файла. Всё остальное — про то, что
беда с админкой не роняет звонок: не ответила, ответила не тем, ответила
битым — берём файл и работаем.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from script.build import ScriptError, build_script
from script.source import HttpScriptSource, JsonScriptSource, ScriptRegistry


@dataclass
class ОтветЗаглушка:
    """Минимальный ответ httpx: статус и тело."""

    status_code: int
    payload: Any = None
    битый: bool = False

    def json(self) -> Any:
        """Отдаёт тело или роняет разбор, как настоящий ответ с мусором."""
        if self.битый:
            raise ValueError("не JSON")
        return self.payload


@pytest.fixture
def файл_скрипта(data_dir: Path) -> dict[str, Any]:
    """Содержимое опубликованного скрипта — то же, что отдаст админка."""
    return json.loads((data_dir / "vector_ru.v4.json").read_text(encoding="utf-8"))


def _источник(monkeypatch, ответ, data_dir: Path) -> HttpScriptSource:
    """Собирает источник, у которого запрос отвечает заданным образом."""

    def подмена(url: str, params=None, timeout=None):  # noqa: ANN001, ARG001
        if isinstance(ответ, Exception):
            raise ответ
        return ответ

    monkeypatch.setattr("script.source.httpx.get", подмена)
    return HttpScriptSource(
        base_url="http://vector-kb:8317",
        fallback=JsonScriptSource(data_dir),
    )


def test_скрипт_из_админки_совпадает_со_скриптом_из_файла(monkeypatch, data_dir, файл_скрипта):
    """Собранный из ответа админки скрипт неотличим от собранного из файла.

    Это и есть бесшовность переезда: всё, что выше источника, работает со
    скомпилированным объектом и про происхождение данных не знает.
    """
    источник = _источник(monkeypatch, ОтветЗаглушка(200, файл_скрипта), data_dir)

    из_админки = build_script(источник.fetch("vector_ru", None))
    из_файла = build_script(JsonScriptSource(data_dir).fetch("vector_ru", "4"))

    assert из_админки.id == из_файла.id
    assert из_админки.version == из_файла.version
    assert list(из_админки.steps) == list(из_файла.steps)
    for ключ, шаг in из_файла.steps.items():
        assert из_админки.steps[ключ] == шаг


def test_админка_не_ответила_берём_файл(monkeypatch, data_dir):
    """Сетевая ошибка не роняет ход: скрипт приезжает из образа."""
    import httpx

    источник = _источник(monkeypatch, httpx.ConnectError("нет связи"), data_dir)
    raw = источник.fetch("vector_ru", "4")
    assert raw.version == "4"


def test_админка_ответила_404_берём_файл(monkeypatch, data_dir):
    """Документа в базе ещё нет — работаем по файлу, как до переезда."""
    источник = _источник(monkeypatch, ОтветЗаглушка(404), data_dir)
    assert источник.fetch("vector_ru", "4").version == "4"


def test_админка_ответила_пятисоткой_берём_файл(monkeypatch, data_dir):
    """База админки лежит — звонку это безразлично."""
    источник = _источник(monkeypatch, ОтветЗаглушка(503), data_dir)
    assert источник.fetch("vector_ru", "4").version == "4"


def test_битый_ответ_админки_берём_файл(monkeypatch, data_dir):
    """Мусор вместо JSON — тоже повод взять файл, а не упасть."""
    источник = _источник(monkeypatch, ОтветЗаглушка(200, битый=True), data_dir)
    assert источник.fetch("vector_ru", "4").version == "4"


def test_ответ_не_проходит_разбор_берём_файл(monkeypatch, data_dir):
    """Документ без шагов моделью не собирается — значит, идём в файл."""
    источник = _источник(monkeypatch, ОтветЗаглушка(200, {"id": "vector_ru"}), data_dir)
    assert источник.fetch("vector_ru", "4").version == "4"


def test_закреплённой_версии_нет_в_файлах_берём_последнюю(monkeypatch, data_dir):
    """Версия, опубликованная после сборки образа, не должна рвать звонок.

    Звонок закрепил версию, которой в образе нет, а админка в этот момент
    отвалилась. Разговор на последней файловой версии лучше оборванного.
    """
    источник = _источник(monkeypatch, ОтветЗаглушка(503), data_dir)
    raw = источник.fetch("vector_ru", "99")
    assert raw.version == "4"


def test_скрипта_нет_нигде_это_ошибка(monkeypatch, data_dir):
    """Если скрипта нет ни в админке, ни в файлах, молчать нельзя."""
    источник = _источник(monkeypatch, ОтветЗаглушка(404), data_dir)
    with pytest.raises(ScriptError):
        источник.fetch("нет_такого", None)


def test_точная_версия_запрашивается_у_админки_один_раз(monkeypatch, data_dir, файл_скрипта):
    """Ход не должен ходить по сети каждый раз: версия уже закреплена."""
    запросы: list[Any] = []

    def подмена(url: str, params=None, timeout=None):  # noqa: ANN001, ARG001
        запросы.append(params)
        return ОтветЗаглушка(200, файл_скрипта)

    monkeypatch.setattr("script.source.httpx.get", подмена)
    источник = HttpScriptSource(base_url="http://kb", fallback=JsonScriptSource(data_dir))
    реестр = ScriptRegistry(источник, latest_ttl=60)

    первый = реестр.get("vector_ru", "4")
    второй = реестр.get("vector_ru", "4")

    assert первый is второй
    assert len(запросы) == 1


def test_последняя_версия_перечитывается_по_таймеру(monkeypatch, data_dir, файл_скрипта):
    """Опубликованную версию процесс подхватывает без перезапуска.

    Запрос без версии — это начало нового звонка. Если бы «последняя»
    кэшировалась навсегда, процесс, поднявшийся до публикации, раздавал бы
    старый скрипт до самого перезапуска.
    """
    запросы: list[Any] = []

    def подмена(url: str, params=None, timeout=None):  # noqa: ANN001, ARG001
        запросы.append(params)
        return ОтветЗаглушка(200, файл_скрипта)

    monkeypatch.setattr("script.source.httpx.get", подмена)
    источник = HttpScriptSource(base_url="http://kb", fallback=JsonScriptSource(data_dir))
    реестр = ScriptRegistry(источник, latest_ttl=0)

    реестр.get("vector_ru", None)
    реестр.get("vector_ru", None)

    assert len(запросы) == 2


def test_свежая_последняя_версия_из_кэша(monkeypatch, data_dir, файл_скрипта):
    """Пока срок свежести не вышел, по сети не ходим."""
    запросы: list[Any] = []

    def подмена(url: str, params=None, timeout=None):  # noqa: ANN001, ARG001
        запросы.append(params)
        return ОтветЗаглушка(200, файл_скрипта)

    monkeypatch.setattr("script.source.httpx.get", подмена)
    источник = HttpScriptSource(base_url="http://kb", fallback=JsonScriptSource(data_dir))
    реестр = ScriptRegistry(источник, latest_ttl=3600)

    реестр.get("vector_ru", None)
    реестр.get("vector_ru", None)

    assert len(запросы) == 1


def test_версия_из_ответа_закрепляется_в_кэше(monkeypatch, data_dir, файл_скрипта):
    """После запроса без версии тот же объект достаётся по точному номеру.

    Ход кладёт версию в состояние звонка и на следующем ходу просит её
    явно — запрос по сети повториться не должен.
    """
    запросы: list[Any] = []

    def подмена(url: str, params=None, timeout=None):  # noqa: ANN001, ARG001
        запросы.append(params)
        return ОтветЗаглушка(200, файл_скрипта)

    monkeypatch.setattr("script.source.httpx.get", подмена)
    источник = HttpScriptSource(base_url="http://kb", fallback=JsonScriptSource(data_dir))
    реестр = ScriptRegistry(источник, latest_ttl=3600)

    последний = реестр.get("vector_ru", None)
    точный = реестр.get("vector_ru", последний.version)

    assert точный is последний
    assert len(запросы) == 1
