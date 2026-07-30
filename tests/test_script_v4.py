"""Тесты скрипта продаж v4: сборка, шапка, промпт, прогрев, настройки."""

from __future__ import annotations

import copy

import pytest

from core.config import settings
from graph.facts import needs_of
from graph.prompts import _EXAMPLES_PREFIX, _describe_step, build_turn_messages, unknown_block
from graph.tools_registry import build_context_tools, needs_from_knowledge
from script.build import ScriptError, build_script, params_from_settings
from script.models import RawSalesScript, SalesStep
from script.planner import script_head
from script.source import JsonScriptSource


def test_v4_собирается_26_шагов_по_шесть_полей(script_v4):
    """Скрипт v4 собирается; у каждого шага ровно шесть полей."""
    assert script_v4.is_sales
    assert script_v4.version == "4"
    assert len(script_v4.steps) == 26
    assert len(script_v4.step_order) == 26
    empty_knowledge = 0
    for step in script_v4.steps.values():
        dumped = step.model_dump()
        assert set(dumped) == {
            "id",
            "name",
            "order",
            "requirements",
            "examples",
            "knowledge",
        }
        assert isinstance(dumped["knowledge"], list)
        if not dumped["knowledge"]:
            empty_knowledge += 1
    assert empty_knowledge == 8
    experience = script_v4.step("experience")
    assert "подбадривать" in experience.requirements


def test_v4_порядок_по_order(script_v4):
    """Шаги в step_order идут по возрастанию order."""
    orders = [script_v4.step(sid).order for sid in script_v4.step_order]
    assert orders == sorted(orders)
    assert script_v4.step_order[0] == "greeting"
    assert script_v4.step_order[1] == "city"


@pytest.mark.parametrize(
    "field,value",
    [
        ("name", ""),
        ("name", "   "),
        ("requirements", ""),
        ("requirements", "  "),
        ("examples", []),
    ],
)
def test_v4_пустые_поля_не_проходят_сборку(raw_script_v4, field, value):
    """Пустые name / requirements / examples — ошибка сборки."""
    payload = copy.deepcopy(raw_script_v4.model_dump())
    payload["steps"][0][field] = value
    with pytest.raises(ScriptError):
        build_script(RawSalesScript.model_validate(payload))


def test_v4_пустой_пример_в_списке_не_проходит(raw_script_v4):
    payload = copy.deepcopy(raw_script_v4.model_dump())
    payload["steps"][0]["examples"] = ["нормальный", "  "]
    with pytest.raises(ScriptError, match="пустой образец"):
        build_script(RawSalesScript.model_validate(payload))


def test_v4_повтор_id_не_проходит(raw_script_v4):
    payload = copy.deepcopy(raw_script_v4.model_dump())
    payload["steps"].append(copy.deepcopy(payload["steps"][0]))
    payload["steps"][-1]["order"] = 99999
    with pytest.raises(ScriptError, match="Повтор идентификатора"):
        build_script(RawSalesScript.model_validate(payload))


def test_v4_повтор_order_не_проходит(raw_script_v4):
    payload = copy.deepcopy(raw_script_v4.model_dump())
    payload["steps"][1]["order"] = payload["steps"][0]["order"]
    with pytest.raises(ScriptError, match="Повтор порядка"):
        build_script(RawSalesScript.model_validate(payload))


def test_knowledge_пустой_и_заполненный():
    """Шаг разбирается с пустым knowledge и с заполненным списком."""
    empty = SalesStep.model_validate(
        {
            "id": "x",
            "name": "Шаг",
            "order": 1,
            "requirements": "сделать",
            "examples": ["фраза"],
            "knowledge": [],
        }
    )
    assert empty.knowledge == []

    filled = SalesStep.model_validate(
        {
            "id": "y",
            "name": "Шаг",
            "order": 2,
            "requirements": "сделать",
            "examples": ["фраза"],
            "knowledge": ["филиалы города", "свободные места"],
        }
    )
    assert filled.knowledge == ["филиалы города", "свободные места"]

    bare = SalesStep.model_validate(
        {
            "id": "z",
            "name": "Шаг",
            "order": 3,
            "requirements": "сделать",
            "examples": ["фраза"],
        }
    )
    assert bare.knowledge == []


def test_шапка_v4_по_order_и_потолок(script_v4):
    """Шапка идёт по order; висящие + один новый; потолок уважается."""
    head = script_head(script_v4, status={}, attempts={}, profile={}, pending_soft_cap=4)
    assert [s.id for s in head] == ["greeting"]

    attempts = {"greeting": 1, "city": 1, "who_studies": 1}
    head = script_head(script_v4, status={}, attempts=attempts, profile={}, pending_soft_cap=4)
    assert [s.id for s in head] == ["greeting", "city", "who_studies", "experience"]

    attempts = {"greeting": 1, "city": 1, "who_studies": 1, "experience": 1}
    head = script_head(script_v4, status={}, attempts=attempts, profile={}, pending_soft_cap=4)
    assert [s.id for s in head] == ["greeting", "city", "who_studies", "experience"]
    assert "transmission" not in {s.id for s in head}

    status = {"greeting": "closed", "city": "closed"}
    head = script_head(script_v4, status=status, attempts={}, profile={}, pending_soft_cap=4)
    assert [s.id for s in head] == ["who_studies"]


def test_v1_v2_v3_собираются_как_раньше(script, script_v1, script_v3, data_dir):
    """v1, v2 и v3 собираются и не помечаются как продажи."""
    assert script.version == "2"
    assert not script.is_sales
    assert script_v1.version == "1"
    assert not script_v1.is_sales
    assert script_v3.version == "3"
    assert not script_v3.is_sales
    assert "city" in script.steps
    assert JsonScriptSource(data_dir).fetch("vector_ru", "3").version == "3"


def test_промпт_v4_название_требования_образцы(script_v4):
    """Вывод шага: название, требования, образцы с пометкой про дословность."""
    step = script_v4.step("city")
    lines = _describe_step(step, {}, {}, heading="Шаг")
    text = "\n".join(lines)
    assert "Название: Выявление города" in text
    assert "Требования:" in text
    assert "Записать город в форму как city" in text
    assert _EXAMPLES_PREFIX in text
    assert "Подскажите, в каком городе планируете обучение?" in text

    messages = build_turn_messages(
        script=script_v4,
        steps=[step],
        profile={},
        facts={},
        history=[],
        asides_done=[],
    )
    content = messages[0].content
    assert "Выявление города" in content
    assert _EXAMPLES_PREFIX in content


def test_промпт_v4_нехватка_данных_из_knowledge(script_v4):
    """При непустом knowledge и отсутствии фактов — строка о нехватке."""
    step = script_v4.step("terms")
    assert step.knowledge
    lines = _describe_step(step, {}, {}, heading="Шаг", context_text="")
    text = "\n".join(lines)
    assert "В контексте нет данных:" in text
    assert "время до первого занятия" in text
    assert "не выдумывать" in text

    # Если факты уже в контексте — строки нет.
    ctx = "срок обучения по городу: 2 месяца; время до первого занятия по вождению: 3 дня"
    lines_ok = _describe_step(step, {}, {}, heading="Шаг", context_text=ctx)
    assert "В контексте нет данных:" not in "\n".join(lines_ok)


def test_промпт_v4_пустой_knowledge_без_строки_нехватки(script_v4):
    """При пустом knowledge строки о нехватке данных нет."""
    step = script_v4.step("greeting")
    assert step.knowledge == []
    lines = _describe_step(step, {}, {}, heading="Шаг", context_text="")
    assert "В контексте нет данных:" not in "\n".join(lines)


def test_прогрев_по_всему_списку_knowledge(script_v4):
    """needs_of / прогрев идут по всему списку knowledge."""
    city = script_v4.step("city")
    assert needs_of(city) == ["city_choices"]
    assert needs_of(city) == needs_from_knowledge(city.knowledge)

    terms = script_v4.step("terms")
    needs = needs_of(terms)
    assert "city_meta" in needs
    assert needs == needs_from_knowledge(terms.knowledge)
    # Факт без маппинга в справочник не даёт потребности — просто не найдётся.
    assert "время до первого занятия по вождению" in terms.knowledge
    assert needs_from_knowledge(["время до первого занятия по вождению"]) == []

    price = script_v4.step("price")
    assert needs_of(price) == ["price"]

    branch = script_v4.step("branch")
    assert "branches" in needs_of(branch)


def test_заглушки_и_фолбэк_v4_из_настроек(script_v4, data_dir, monkeypatch):
    """Для нового формата fallback/unknown берутся из настроек агента."""
    monkeypatch.setattr(settings, "agent_fallback", "Тестовый фолбэк из настроек.")
    monkeypatch.setattr(settings, "agent_unknown", "Тестовый unknown из настроек.")
    params = params_from_settings()
    assert params.fallback == "Тестовый фолбэк из настроек."
    assert params.unknown == "Тестовый unknown из настроек."

    raw = JsonScriptSource(data_dir).fetch("vector_ru", "4")
    compiled = build_script(raw)
    assert compiled.params.fallback == "Тестовый фолбэк из настроек."
    assert compiled.params.unknown == "Тестовый unknown из настроек."
    block = unknown_block(compiled)
    assert "Тестовый unknown из настроек." in block


def test_реестр_инструментов_всегда_branches_faq_details(script_v4, script):
    """Реестр одинаков для продаж и legacy: город, филиалы, FAQ, детали, факты."""
    for compiled in (script_v4, script):
        tools = build_context_tools(compiled)
        assert [t.name for t in tools] == [
            "city",
            "branches",
            "city_faq",
            "branch_details",
            "facts",
        ]
