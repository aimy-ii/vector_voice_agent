"""Тесты источника скриптов: v1 и v2, подмена источника бесшовна."""

from __future__ import annotations

import json

import pytest

from script.build import ScriptError, build_script
from script.models import RawScript
from script.source import JsonScriptSource, ScriptRegistry


def test_последняя_версия_берётся_без_указания(data_dir):
    raw = JsonScriptSource(data_dir).fetch("vector_ru", None)
    assert raw.version == "2"


def test_точная_версия_v1_поднимается(data_dir):
    raw = JsonScriptSource(data_dir).fetch("vector_ru", "1")
    assert raw.id == "vector_ru"
    assert raw.version == "1"


def test_точная_версия_v2_поднимается(data_dir):
    raw = JsonScriptSource(data_dir).fetch("vector_ru", "2")
    assert raw.version == "2"


def test_v1_и_v2_обе_собираются(raw_script, raw_script_v1):
    v2 = build_script(raw_script)
    v1 = build_script(raw_script_v1)
    assert v2.version == "2"
    assert v1.version == "1"
    assert "practice" in v2.steps
    assert "presentation" in v1.steps


def test_несуществующая_версия_падает_с_перечнем(data_dir):
    with pytest.raises(ScriptError, match="Есть:"):
        JsonScriptSource(data_dir).fetch("vector_ru", "99")


def test_несуществующий_скрипт_падает(data_dir):
    with pytest.raises(ScriptError, match="не найден"):
        JsonScriptSource(data_dir).fetch("нет_такого", None)


def test_версия_в_файле_обязана_совпадать_с_именем(tmp_path, raw_script):
    payload = raw_script.model_dump()
    payload["version"] = "7"
    (tmp_path / "vector_ru.v2.json").write_text(
        json.dumps(payload, ensure_ascii=False), encoding="utf-8"
    )
    with pytest.raises(ScriptError, match="не совпадает"):
        JsonScriptSource(tmp_path).fetch("vector_ru", None)


def test_битый_json_падает_на_загрузке(tmp_path):
    (tmp_path / "vector_ru.v1.json").write_text("{не json", encoding="utf-8")
    with pytest.raises(ScriptError, match="не читается"):
        JsonScriptSource(tmp_path).fetch("vector_ru", None)


def test_реестр_отдаёт_тот_же_объект(registry):
    первый = registry.get("vector_ru", "2")
    второй = registry.get("vector_ru", "2")
    assert первый is второй


def test_версии_сортируются_как_числа(tmp_path, raw_script):
    for version in ("2", "10"):
        payload = raw_script.model_dump()
        payload["version"] = version
        (tmp_path / f"vector_ru.v{version}.json").write_text(
            json.dumps(payload, ensure_ascii=False), encoding="utf-8"
        )
    assert JsonScriptSource(tmp_path).fetch("vector_ru", None).version == "10"


def test_подмена_источника_бесшовна(raw_script):
    """Вторая реализация пойдёт по HTTP; выше по стеку ничего не меняется."""

    class InMemorySource:
        """Источник из памяти вместо каталога."""

        def __init__(self) -> None:
            self.asked: list[tuple[str, str | None]] = []

        def fetch(self, script_id: str, version: str | None) -> RawScript:
            """Отдаёт заранее подготовленный скрипт."""
            self.asked.append((script_id, version))
            return raw_script

    source = InMemorySource()
    compiled = ScriptRegistry(source).get("vector_ru", "2")

    assert compiled.id == "vector_ru"
    assert compiled.steps["city"].kind == "question"
    assert source.asked == [("vector_ru", "2")]
