"""Тесты сборки скрипта: скрипт обязан падать на загрузке, а не в звонке."""

from __future__ import annotations

import copy

import pytest

from script.build import ScriptError, build_script
from script.models import RawScript


def _mutate(raw: RawScript, **changes: object) -> RawScript:
    """Копирует сырой скрипт с точечными правками."""
    payload = copy.deepcopy(raw.model_dump())
    payload.update(changes)
    return RawScript.model_validate(payload)


def test_базовый_скрипт_собирается(script):
    assert script.id == "vector_ru"
    assert script.version == "2"
    assert "city" in script.steps
    assert "practice" in script.steps
    assert script.key == ("vector_ru", "2")


def test_собранный_скрипт_знает_кто_заполняет_поле(script):
    assert script.filled_by["city"] == "city"
    assert script.filled_by["theory_format"] == "theory_format"


def test_собранный_скрипт_неизменяем(script):
    with pytest.raises(Exception):
        script.id = "другой"


def test_справка_и_возражение_находятся_по_идентификатору(script):
    assert script.aside("medcheck") is not None
    assert script.aside("think") is not None
    assert script.aside("нет-такого") is None


def test_повтор_идентификатора_шага_ловится(raw_script):
    payload = copy.deepcopy(raw_script.model_dump())
    payload["steps"].append(copy.deepcopy(payload["steps"][0]))
    with pytest.raises(ScriptError, match="Повтор идентификатора"):
        build_script(RawScript.model_validate(payload))


def test_ссылка_на_необъявленное_поле_ловится(raw_script):
    payload = copy.deepcopy(raw_script.model_dump())
    payload["steps"][0]["fills"] = ["такого_поля_нет"]
    with pytest.raises(ScriptError, match="необъявленные поля"):
        build_script(RawScript.model_validate(payload))


def test_ожидание_несуществующего_шага_ловится(raw_script):
    payload = copy.deepcopy(raw_script.model_dump())
    payload["steps"][0]["after"] = ["призрак"]
    with pytest.raises(ScriptError, match="несуществующие шаги"):
        build_script(RawScript.model_validate(payload))


def test_дословный_блок_без_текста_ловится(raw_script):
    payload = copy.deepcopy(raw_script.model_dump())
    for step in payload["steps"]:
        if step["id"] == "practice":
            step["text"] = None
            step["branches"] = None
    with pytest.raises(ScriptError, match="Дословный шаг"):
        build_script(RawScript.model_validate(payload))


def test_недостижимый_шаг_ловится(raw_script):
    payload = copy.deepcopy(raw_script.model_dump())
    payload["profile_fields"].append({"key": "сирота", "title": "Никем не заполняется"})
    payload["steps"][0]["requires"] = ["сирота"]
    with pytest.raises(ScriptError, match="недостижим"):
        build_script(RawScript.model_validate(payload))


def test_цикл_ожидания_ловится(raw_script):
    payload = copy.deepcopy(raw_script.model_dump())
    by_id = {s["id"]: s for s in payload["steps"]}
    by_id["city"]["after"] = ["name"]
    by_id["name"]["after"] = ["city"]
    with pytest.raises(ScriptError, match="цикл ожидания"):
        build_script(RawScript.model_validate(payload))


def test_скрипт_без_аварийной_реплики_ловится(raw_script):
    payload = copy.deepcopy(raw_script.model_dump())
    payload["params"]["fallback"] = ""
    with pytest.raises(ScriptError, match="аварийной реплики"):
        build_script(RawScript.model_validate(payload))


def test_проверочный_вопрос_обязателен_для_информирования_с_проверкой(raw_script):
    payload = copy.deepcopy(raw_script.model_dump())
    for step in payload["steps"]:
        if step["id"] == "practice":
            step["check_question"] = None
    with pytest.raises(ScriptError, match="проверочного вопроса"):
        build_script(RawScript.model_validate(payload))


def test_step_разбирает_новые_поля_и_дефолты():
    """Step принимает why/examples/avoid и даёт пустые значения без них."""
    from script.models import Step

    bare = Step.model_validate({"id": "x", "kind": "question", "goal": "узнать город"})
    assert bare.why == ""
    assert bare.examples == []
    assert bare.avoid == ""

    full = Step.model_validate(
        {
            "id": "y",
            "kind": "inform",
            "goal": "рассказать",
            "why": "зачем",
            "examples": ["фраза один", "фраза два"],
            "avoid": "нельзя так",
            "text": "готово",
        }
    )
    assert full.why == "зачем"
    assert full.examples == ["фраза один", "фраза два"]
    assert full.avoid == "нельзя так"


def test_inform_только_с_examples_собирается(raw_script):
    """Шаг inform проходит проверку, если есть examples без text и branches."""
    payload = copy.deepcopy(raw_script.model_dump())
    for step in payload["steps"]:
        if step["id"] == "terms":
            step["text"] = None
            step["branches"] = None
            step["examples"] = ["Срок обучения — около двух месяцев."]
            step["verbatim"] = False
    compiled = build_script(RawScript.model_validate(payload))
    assert compiled.steps["terms"].examples == ["Срок обучения — около двух месяцев."]
    assert compiled.steps["terms"].text is None


def test_inform_без_text_branches_examples_падает(raw_script):
    """Шаг inform без text, branches и examples — ошибка сборки."""
    payload = copy.deepcopy(raw_script.model_dump())
    for step in payload["steps"]:
        if step["id"] == "terms":
            step["text"] = None
            step["branches"] = None
            step["examples"] = []
            step["verbatim"] = False
    with pytest.raises(ScriptError, match="информирования"):
        build_script(RawScript.model_validate(payload))


def test_objection_разбирает_why():
    """Objection принимает why и даёт пустую строку по умолчанию."""
    from script.models import Objection

    bare = Objection.model_validate({"id": "think", "triggers": ["подумаю"], "text": "Хорошо."})
    assert bare.why == ""
    full = Objection.model_validate(
        {"id": "think", "triggers": ["подумаю"], "text": "Хорошо.", "why": "не давить"}
    )
    assert full.why == "не давить"
