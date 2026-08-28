"""Тесты скрипта продаж v4: сборка, шапка, промпт, прогрев, настройки."""

from __future__ import annotations

import copy

import pytest

from core.config import settings
from graph.facts import knowledge_of, needs_of
from graph.prompts import _describe_step, build_turn_messages, unknown_block
from graph.tools_registry import build_context_tools, needs_from_knowledge
from script.build import ScriptError, build_script, params_from_settings
from script.models import RawSalesScript, SalesStep
from script.planner import script_head
from script.source import JsonScriptSource


def test_v4_собирается_22_шага_по_полям(script_v4):
    """Скрипт v4 собирается; у каждого шага семь полей, включая form."""
    assert script_v4.is_sales
    assert script_v4.version == "4"
    assert len(script_v4.steps) == 22
    assert len(script_v4.step_order) == 22
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
            "form",
        }
        assert isinstance(dumped["knowledge"], list)
        assert isinstance(dumped["form"], str)
        if not dumped["knowledge"]:
            empty_knowledge += 1
    assert empty_knowledge == 7
    experience = script_v4.step("experience")
    assert "подбадривать" in experience.requirements


def test_v4_location_hint_между_who_studies_и_experience(script_v4):
    """Шаг location_hint: order 45, позиция в step_order, пустой knowledge."""
    step = script_v4.step("location_hint")
    assert step.order == 45
    assert step.knowledge == []
    order = script_v4.step_order
    assert order.index("location_hint") == order.index("who_studies") + 1
    assert order.index("experience") == order.index("location_hint") + 1


def test_v4_branch_не_спрашивает_район_заново(script_v4):
    """В требованиях шага branch есть запрет спрашивать район заново."""
    req = script_v4.step("branch").requirements
    assert "заново не спрашивать" in req
    assert "нет в переданных ближайших" in req


def test_v4_branch_examples_не_спрашивают_район(script_v4):
    """Образцы branch не пересекаются с location_hint и не содержат «район»."""
    branch = set(script_v4.step("branch").examples)
    location = set(script_v4.step("location_hint").examples)
    assert branch.isdisjoint(location)
    assert all("район" not in example.lower() for example in branch)


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
    assert empty.form == ""

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
    assert bare.form == ""


def test_шапка_v4_по_order_и_потолок(script_v4):
    """Шапка идёт по order; висящие + один новый; потолок уважается."""
    head = script_head(script_v4, status={}, attempts={}, profile={}, pending_soft_cap=4)
    assert [s.id for s in head] == ["greeting"]

    attempts = {"greeting": 1, "city": 1, "who_studies": 1}
    head = script_head(script_v4, status={}, attempts=attempts, profile={}, pending_soft_cap=4)
    assert [s.id for s in head] == ["greeting", "city", "who_studies", "location_hint"]

    attempts = {"greeting": 1, "city": 1, "who_studies": 1, "location_hint": 1}
    head = script_head(script_v4, status={}, attempts=attempts, profile={}, pending_soft_cap=4)
    assert [s.id for s in head] == ["greeting", "city", "who_studies", "location_hint"]
    assert "experience" not in {s.id for s in head}

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
    """Вывод шага: название, требования, примеры; служебное form не попадает."""
    step = script_v4.step("city")
    lines = _describe_step(step, {}, {}, heading="Шаг")
    text = "\n".join(lines)
    assert "## Выявление города" in text
    assert "**Требования**" in text
    assert "Спросить город обучения и записать его" in text
    assert "Записать город в форму как city" not in text
    assert "**Примеры**" in text
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
    assert "**Примеры**" in content
    assert "ВНИМАНИЕ: примеры" in content


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
    assert city.knowledge == []
    assert needs_of(city) == []
    assert needs_of(city) == needs_from_knowledge(city.knowledge)
    assert knowledge_of(city) == []

    branch = script_v4.step("branch")
    assert branch.knowledge == []
    assert needs_of(branch) == []
    assert knowledge_of(branch) == []

    terms = script_v4.step("terms")
    needs = needs_of(terms)
    assert "city_meta" in needs
    assert needs == needs_from_knowledge(terms.knowledge)
    # Факт без маппинга в справочник не даёт потребности — просто не найдётся.
    assert "время до первого занятия по вождению" in terms.knowledge
    assert needs_from_knowledge(["время до первого занятия по вождению"]) == []
    assert knowledge_of(terms) == list(terms.knowledge)

    included = script_v4.step("included")
    assert "city_meta" in needs_of(included)

    price = script_v4.step("price")
    assert needs_of(price) == ["price"]
    assert price.knowledge
    assert knowledge_of(price) == ["стоимость обучения в городе"]


def test_knowledge_of_без_шага_пустой():
    """Без шага knowledge_of пуст."""
    assert knowledge_of(None) == []


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
    """Реестр одинаков для продаж и legacy.

    Город, филиалы, FAQ, детали филиала, ближайшие, факты шага и доводы
    под возражение. Порядок закреплён: агент контекстера выбирает
    инструмент по описанию, и перестановка меняет то, что он видит первым.
    """
    for compiled in (script_v4, script):
        tools = build_context_tools(compiled)
        assert [t.name for t in tools] == [
            "city",
            "branches",
            "city_faq",
            "branch_details",
            "nearest_branches",
            "facts",
            "objections",
        ]


def test_v4_сводка_непустая_и_в_скомпилированном(raw_script_v4, script_v4):
    """Скрипт vector_ru v4 читается; поле сводки непустое и доходит до сборки."""
    assert raw_script_v4.summary.strip()
    assert "федеральную сеть автошкол" in raw_script_v4.summary
    assert script_v4.summary == raw_script_v4.summary
    assert script_v4.summary.strip()


def test_messenger_подтверждает_номер_а_не_спрашивает_вслепую(script_v4):
    """Шаг messenger: сначала этот номер или другой; номер — только после «на другой»."""
    step = script_v4.step("messenger")
    req = step.requirements.lower()
    assert "первым делом" in req
    assert "с которого" in req and "звон" in req
    assert "на него или на другой" in req
    assert "только после" in req
    assert "на другой" in req
    assert "ошибка" in req
    for blind in (
        "назовите номер",
        "продиктуйте номер",
        "какой у вас номер",
        "подскажите номер",
        "скажите номер",
    ):
        assert blind not in req
    examples = " ".join(step.examples).lower()
    for blind in (
        "назовите номер",
        "продиктуйте номер",
        "какой у вас номер",
        "подскажите номер",
        "скажите свой номер",
    ):
        assert blind not in examples
    assert any("с которого" in ex.lower() and "звон" in ex.lower() for ex in step.examples)
    assert len(script_v4.steps) == 22


def test_скрипт_без_сводки_читается_с_пустым_полем():
    """Скрипт продаж без поля summary читается без ошибок; сводка пустая."""
    payload = {
        "id": "no_summary",
        "version": "1",
        "steps": [
            {
                "id": "only",
                "name": "Шаг",
                "order": 1,
                "requirements": "Спросить имя.",
                "examples": ["Как вас зовут?"],
                "knowledge": [],
            }
        ],
    }
    raw = RawSalesScript.model_validate(payload)
    assert raw.summary == ""
    compiled = build_script(raw)
    assert compiled.summary == ""


@pytest.mark.parametrize("step_id", ["branch", "discount_check", "tariff", "price_lock"])
def test_v4_шаги_с_реакцией_имеют_строку_закрытия(script_v4, step_id):
    """Строка «Шаг закрыт, когда…» стоит до строки «Зачем:»."""
    lines = script_v4.step(step_id).requirements.split("\n")
    closed_at = next(i for i, line in enumerate(lines) if line.startswith("Шаг закрыт, когда"))
    why_at = next(i for i, line in enumerate(lines) if line.startswith("Зачем:"))
    assert closed_at < why_at


def test_v4_закрытие_branch_требует_согласия(script_v4):
    """Закрытие branch: согласие на офис; названный без подтверждения не закрывает."""
    req = script_v4.step("branch").requirements
    assert "согласился на конкретный офис" in req
    assert "Названный, но не подтверждённый филиал шаг не закрывает" in req


def test_v4_location_hint_просит_то_что_находится_по_координатам(script_v4):
    """Шаг location_hint просит район/улицу/метро, не абстрактный ориентир."""
    req = script_v4.step("location_hint").requirements
    assert "район города, улица или станция метро" in req
    assert "или ориентир" not in req


def test_v4_location_hint_запрещает_просить_объект(script_v4):
    """Шаг location_hint не просит здание или торговый центр вместо адреса."""
    req = script_v4.step("location_hint").requirements
    assert "просить назвать здание, торговый центр" in req


def test_v4_branch_ветка_когда_филиалов_не_передали(script_v4):
    """Шаг branch просит другой ориентир, если филиалов не передали."""
    req = script_v4.step("branch").requirements
    assert "Филиалов не передали" in req
    assert "попросить другой ориентир" in req


def test_v4_branch_возвращается_к_филиалу_после_ориентира(script_v4):
    """После нового ориентира branch снова называет ближайший филиал."""
    req = script_v4.step("branch").requirements
    assert "вернуться к филиалу и назвать ближайший" in req


def test_v4_branch_запрещает_выдумывать_филиал_и_зацикливаться(script_v4):
    """Шаг branch не выдумывает филиал и не крутит «подбор идёт» без нового ориентира."""
    req = script_v4.step("branch").requirements
    assert "выдумывать филиал по названному месту" in req
    assert "больше одного раза подряд" in req


def test_v4_branch_не_запрещает_обещание_в_мессенджер(script_v4):
    """Обещание прислать детали в мессенджер в branch не запрещено."""
    req = script_v4.step("branch").requirements
    assert "мессенджер" not in req
    assert "Telegram" not in req


def test_v4_location_hint_и_branch_сохраняют_прежние_условия(script_v4):
    """Регресс: ключевые условия location_hint и branch на месте."""
    branch = script_v4.step("branch").requirements
    location = script_v4.step("location_hint").requirements
    assert "Шаг закрыт, когда человек согласился на конкретный офис" in branch
    assert "перечислять все филиалы города" in branch
    assert "Сказать, что подберёшь ближайший филиал" in location


def test_v4_location_hint_и_branch_число_строк_requirements(script_v4):
    """У location_hint три строки требований, у branch — пять."""
    assert len(script_v4.step("location_hint").requirements.split("\n")) == 3
    assert len(script_v4.step("branch").requirements.split("\n")) == 5


def test_v4_закрытие_discount_check_фиксирует_ответ_по_категории(script_v4):
    """Закрытие discount_check: категории как подсказка, ответ закрывает шаг.

    Шаг называет две-три категории как подсказку, подбирает их под собеседника
    и не разбирает отсев вслух. Шаг закрывает любой ответ человека — и «да»,
    и «нет». Заодно сторожим запрет считать цену со скидкой: сумму со скидкой
    называет менеджер при оформлении, бот пересчётом не занимается.
    """
    req = script_v4.step("discount_check").requirements
    assert "ответил, попадает он под категорию" in req
    assert "Назвать две-три льготные категории" in req
    assert "мужчине скидку молодым мамам не называть" in req
    assert "не назвав ни одной категории" in req
    assert "вслух отбрасывать неподходящие категории" in req
    assert "вычитать скидку из стоимости" in req


def test_v4_закрытие_tariff_покрывает_единственный_тариф(script_v4):
    """Закрытие tariff: выбор тарифа или сообщение, что вариант один."""
    req = script_v4.step("tariff").requirements
    assert "выбрал тариф" in req
    assert "если тариф один" in req


def test_v4_закрытие_price_lock_требует_согласия(script_v4):
    """Закрытие price_lock: согласие закрепить условия."""
    req = script_v4.step("price_lock").requirements
    assert "согласился закрепить условия" in req


@pytest.mark.parametrize("step_id", ["branch", "price_lock"])
def test_v4_короткое_согласие_засчитывается(script_v4, step_id):
    """Короткое согласие по смыслу засчитывается; молчание — нет."""
    req = script_v4.step(step_id).requirements
    assert "важен смысл ответа, а не его длина" in req
    assert "молчание и уход от ответа согласием не считаются" in req


def test_требования_других_шагов_не_изменились(script_v4):
    """Кроме вынесенной строки про форму и правки messenger требования не менялись."""
    import json
    from pathlib import Path

    allowed = {"branch", "discount_check", "tariff", "price_lock", "greeting", "messenger"}
    for step_id, step in script_v4.steps.items():
        if step_id in allowed:
            continue
        lines = step.requirements.split("\n")
        assert not any(line.startswith("Шаг закрыт, когда") for line in lines), step_id

    expected = json.loads(
        (Path(__file__).resolve().parent / "data" / "v4_requirements_expected.json").read_text(
            encoding="utf-8"
        )
    )
    for step_id, req in expected.items():
        if step_id == "messenger":
            continue
        assert script_v4.step(step_id).requirements == req, step_id


def test_ни_один_шаг_не_говорит_про_форму_в_требованиях(script_v4):
    """Ни у одного шага requirements не содержит «в форму»."""
    for step_id, step in script_v4.steps.items():
        assert "в форму" not in step.requirements, step_id


def test_служебные_указания_переехали_в_поле_form(script_v4):
    """Ровно четырнадцать шагов с непустым form; ключевые поля на месте."""
    with_form = [s for s in script_v4.steps.values() if s.form.strip()]
    assert len(with_form) == 14
    assert "caller_name" in script_v4.step("greeting").form
    messenger_form = script_v4.step("messenger").form
    assert "messenger" in messenger_form
    assert "caller_phone" in messenger_form


def test_поле_form_не_попадает_в_промпт_генератора(script_v4):
    """Служебное form не уходит в системное сообщение генератора."""
    step = script_v4.step("theory_format")
    assert step.form
    assert "Записать в форму" in step.form
    messages = build_turn_messages(
        script=script_v4,
        steps=[step],
        profile={},
        facts={},
        history=[],
        asides_done=[],
    )
    content = messages[0].content
    assert "Записать в форму" not in content
    assert "Предложить выбор формата теории" in content


def test_поле_form_не_попадает_в_промпт_судьи(script_v4):
    """То, что судья получает по шагу, не содержит «Записать в форму»."""
    from graph.checker import closure_criterion
    from script.models import SalesStep

    step = script_v4.step("theory_format")
    assert isinstance(step, SalesStep)
    assert "Записать в форму" in step.form
    # Судья для SalesStep получает step.requirements, не step.form.
    judge_payload = "\n".join(
        [
            f"Название: {step.name}",
            f"Критерий закрытия: {closure_criterion(step)}",
            f"Требования:\n{step.requirements}",
        ]
    )
    assert "Записать в форму" not in judge_payload
    assert "Предложить выбор формата теории" in judge_payload


def test_messenger_требует_зачитать_номер(script_v4):
    """В требованиях messenger — зачитать номер обратно целиком, один раз."""
    req = script_v4.step("messenger").requirements
    assert "зачитать обратно вслух целиком" in req
    assert "один раз, в той же реплике" in req
    assert "третий раз не переспрашивать" in req


def test_messenger_не_зачитывает_известный_номер(script_v4):
    """Номер, с которого звонят, зачитывать не нужно."""
    assert "зачитывать нечего, он уже известен" in script_v4.step("messenger").requirements


def test_messenger_закрытие_не_изменилось(script_v4):
    """Строка закрытия messenger без изменений."""
    assert (
        "Шаг закрыт, когда понятно, на какой номер и в какой мессенджер писать."
        in script_v4.step("messenger").requirements.split("\n")
    )
