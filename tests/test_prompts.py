"""Тесты сборки промпта и подготовки фактов справочника."""

from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage

from graph.facts import (
    branch_choices,
    branch_summary,
    city_choices,
    city_summary,
    collect_facts,
    confirm_branch,
    confirm_city,
    needs_of,
)
from graph.prompts import (
    _NO_MECHANICS,
    aside_block,
    build_turn_messages,
    context_block,
    facts_block,
    fill_facts,
    naturalness_block,
    persona_block,
    profile_block,
    step_block,
    steps_block,
)


def test_подстановка_фактов_в_текст():
    assert (
        fill_facts("Стоимость: {price_line}", {"price_line": "от 43900"}) == "Стоимость: от 43900"
    )


def test_неизвестный_плейсхолдер_не_роняет_ход():
    assert fill_facts("Адрес: {branch_address}", {}) == "Адрес:"


def test_персона_и_правила_попадают_в_промпт(script):
    block = persona_block(script)
    assert "Дарья" in block
    assert "Вектор" in block
    assert "ты и есть менеджер" in block


def test_профиль_разделяет_роли(script):
    block = profile_block(script, {"caller_name": "Ольга"})
    assert "Ольга" in block
    assert "звонящий" in block
    assert "будущий курсант" in block
    assert "student_name" in block


def test_шаг_с_образцом_попадает_в_промпт(script):
    """Текст verbatim-шага — образец с пометкой, не готовая реплика мимо модели."""
    from graph.prompts import _SAMPLE_PREFIX

    step = script.step("practice")
    messages = build_turn_messages(
        script=script,
        steps=[step],
        profile={},
        facts={},
        history=[],
        asides_done=[],
    )
    content = messages[0].content
    assert _SAMPLE_PREFIX in content
    filled = step.text or ""
    assert filled in content
    assert "Опорная формулировка (можно сказать своими словами)" not in content
    assert "подводи разговор к завершению" not in content.lower()


def test_цена_образца_и_факт_в_промпте_генератора(script):
    """Факт price_line и образец шага price уходят модели вместе."""
    from graph.prompts import _SAMPLE_PREFIX

    step = script.step("price")
    facts = {"price_line": "Стоимость — от 43900 рублей.", "city": {"name": "Пермь"}}
    messages = build_turn_messages(
        script=script,
        steps=[step],
        profile={},
        facts=facts,
        history=[],
        asides_done=[],
    )
    content = messages[0].content
    assert "price_line" in content or "43900" in content
    assert _SAMPLE_PREFIX in content
    assert "от 43900" in content
    assert step.goal in content


def test_промпт_запрещает_восторги_и_разрешает_короткое_подтверждение(script):
    block = naturalness_block(ask_for_move=True)
    lowered = block.lower()
    assert "не оценивай" in lowered and "выбор клиента" in lowered
    assert "хороший выбор" in lowered
    assert "возраст подходит" in lowered
    assert "сложности" in lowered or "лёгкости" in lowered
    assert "молча учесть" in lowered


def test_промпт_требует_живой_проверочный_вопрос(script):
    """check_question — образец смысла; правило требует живую формулировку."""
    step = script.step("practice")
    assert step.check_question
    block = steps_block([step], {}, {}, attempts={})
    assert "Образец смысла проверки" in block
    assert step.check_question in block
    assert f"«{step.check_question}»" in block
    assert "не зачитывай дословно" in block.lower()
    natural = naturalness_block(ask_for_move=True).lower()
    assert "своими словами" in natural
    assert "каждый раз по-разному" in natural
    assert "канцелярская пластинка" in natural
    assert "задай его целиком" not in natural
    assert "не сокращай" not in natural
    messages = build_turn_messages(
        script=script,
        steps=[step],
        profile={},
        facts={},
        history=[],
        asides_done=[],
    )
    content = messages[0].content
    assert step.check_question in content
    assert "не зачитывай" in content.lower()
    assert "не сокращай" not in content.lower()
    assert "задай целиком" not in content.lower()


def test_обычный_шаг_разрешает_свои_слова(script):
    block = step_block(script.step("city"), {}, {})
    assert "своими словами" in block


def test_промпт_видит_шапку_и_запрет_переспрашивать(script):
    block = steps_block(
        [script.step("city"), script.step("who_studies")],
        {},
        {},
        attempts={"city": 1},
    )
    assert "city" in block
    assert "who_studies" in block
    natural = naturalness_block(ask_for_move=True)
    assert "не переспрашивай" in natural.lower()
    assert "верно?" in natural.lower()


def test_промпт_требует_заканчивать_ходом_к_собеседнику(script):
    block = naturalness_block(ask_for_move=True)
    assert "вопросом" in block.lower() and "предложением" in block.lower()
    messages = build_turn_messages(
        script=script,
        steps=[script.step("city")],
        profile={},
        facts={},
        history=[],
        asides_done=[],
    )
    assert "ходом к собеседнику" in messages[0].content.lower()


def test_шапка_с_образцом_тоже_требует_хода_к_собеседнику(script):
    """Незакрытый вопрос в шапке — ask_for_move и для шага с образцом."""
    messages = build_turn_messages(
        script=script,
        steps=[script.step("terms")],
        profile={},
        facts={},
        history=[],
        asides_done=[],
    )
    content = messages[0].content.lower()
    assert "ходом к собеседнику" in content
    assert "сформулируй в этом ключе" in content


def test_шапка_с_висящими_и_образцом_одним_промптом(script):
    """Несколько шагов в шапке + образец — всё в одном системном сообщении."""
    from graph.prompts import _SAMPLE_PREFIX

    messages = build_turn_messages(
        script=script,
        steps=[script.step("city"), script.step("terms")],
        profile={},
        facts={"price_line": "срок два месяца"},
        history=[],
        asides_done=[],
        attempts={"city": 1},
    )
    content = messages[0].content
    assert "city" in content
    assert "terms" in content
    assert _SAMPLE_PREFIX in content
    assert "Шапка скрипта" in content
    assert content.count("Шаг:") >= 2


def test_шапка_пуста_текст_завершения(script):
    messages = build_turn_messages(
        script=script, step=None, profile={}, facts={}, history=[], asides_done=[]
    )
    assert "подводи разговор к завершению" in messages[0].content.lower()


def test_запрет_механики_ровно_один_раз(script):
    messages = build_turn_messages(
        script=script,
        steps=[script.step("city")],
        profile={"caller_name": "Мария"},
        facts={"city": {"name": "Пермь"}},
        history=[],
        asides_done=[],
        spoken_filler="Секунду, гляну.",
        context_text="Город: Пермь",
    )
    content = messages[0].content
    assert content.count(_NO_MECHANICS) == 1


def test_порядок_блоков_персона_профиль_шапка(script):
    messages = build_turn_messages(
        script=script,
        steps=[script.step("city")],
        profile={},
        facts={},
        history=[],
        asides_done=[],
    )
    content = messages[0].content
    persona_i = content.index("Дарья")
    profile_i = content.index("Что уже известно")
    steps_i = content.index("Шапка скрипта")
    assert persona_i < profile_i < steps_i


def test_промпт_запрещает_озвучивать_механику(script):
    block = persona_block(script)
    assert "не проговаривай" in block.lower()
    assert _NO_MECHANICS in block


def test_перечень_городов_не_в_промпте(script):
    messages = build_turn_messages(
        script=script,
        steps=[script.step("city")],
        profile={},
        facts={"city_choices": [{"slug": "perm", "name": "Пермь"}]},
        history=[],
        asides_done=[],
    )
    assert "city_choices" not in messages[0].content
    assert "perm" not in messages[0].content or "Город" in messages[0].content


def test_персона_впереди_контекста(script):
    messages = build_turn_messages(
        script=script,
        steps=[script.step("name")],
        profile={},
        facts={},
        history=[],
        asides_done=[],
        context_text="Статика: город Пермь.",
    )
    content = messages[0].content
    assert content.index("Дарья") < content.index("Статика: город Пермь")


def test_ни_один_goal_не_объясняет_мотивацию(raw_script):
    forbidden = ("от этого зависит", "от него зависит", "от неё зависит", "это нам")
    for step in raw_script.steps:
        lowered = step.goal.lower()
        for phrase in forbidden:
            assert phrase not in lowered, f"{step.id}: {step.goal}"


def test_пустые_факты_не_засоряют_промпт():
    assert facts_block({}) == ""
    assert facts_block({"city": None, "branches": []}) == ""
    assert "Пермь" in facts_block({"city": {"name": "Пермь"}})


def test_перечень_возражений_без_справок(script):
    block = aside_block(script, done=["think"])
    assert "think" in block
    assert "уже отвечали" in block
    assert "medcheck" not in block
    assert "справка" not in block.lower()
    for help_id in script.helps:
        assert help_id not in block


def test_справка_доходит_через_динамику_контекста(script):
    med = script.helps["medcheck"].text
    messages = build_turn_messages(
        script=script,
        step=script.step("name"),
        profile={},
        facts={},
        history=[],
        asides_done=[],
        context_text=f"Город: Пермь\n\n{med}",
        dynamic_status="готово",
    )
    content = messages[0].content
    assert med in content
    assert "medcheck" not in aside_block(script, done=[])


def test_запрос_к_модели_содержит_системный_блок_и_хвост(script):
    history = [HumanMessage(content=f"реплика {i}") for i in range(12)]
    messages = build_turn_messages(
        script=script,
        step=script.step("city"),
        profile={},
        facts={},
        history=history,
        asides_done=[],
    )
    assert isinstance(messages[0], SystemMessage)
    assert len(messages) == 9
    assert messages[-1].content == "реплика 11"


def test_пустая_история_не_ломает_запрос(script):
    messages = build_turn_messages(
        script=script, step=None, profile={}, facts={}, history=[], asides_done=[]
    )
    assert len(messages) == 2
    assert "Все шаги скрипта закрыты" in messages[0].content


def test_города_отдаются_со_названиями():
    choices = city_choices([{"slug": "kyrgan", "name": "Курган"}, {"slug": "нет"}])
    assert choices == [{"slug": "kyrgan", "name": "Курган"}]


def test_филиалы_перечисляются_с_ориентиром():
    choices = branch_choices(
        [
            {"slug": "a", "address": "ул. Ленина, 1", "landmark": "ТЦ Колизей"},
            {"slug": "b", "address": "ул. Мира, 2", "landmark": None},
        ]
    )
    assert choices[0]["landmark"] == "ТЦ Колизей"
    assert "landmark" not in choices[1]


def test_выжимка_города_не_тащит_цену():
    summary = city_summary({"name": "Пермь", "price": {"amount": 1}, "vehicles": {"manual": ["X"]}})
    assert summary["city"] == "Пермь"
    assert summary["vehicles_manual"] == ["X"]
    assert "price" not in summary


def test_на_автодром_за_договором_не_отправляем():
    автодром = branch_summary({"place_type": "автодром", "status": "работает"})
    неоткрытый = branch_summary({"place_type": "учебный офис", "status": "скоро открытие"})
    офис = branch_summary({"place_type": "учебный офис", "status": "работает"})

    assert автодром["can_sign_here"] is False
    assert неоткрытый["can_sign_here"] is False
    assert офис["can_sign_here"] is True


def test_потребности_шага(script):
    assert needs_of(script.step("branch")) == ["branches"]
    assert needs_of(None) == []


async def test_факты_собираются_по_потребностям_шага(script, fake_kb):
    facts, journal = await collect_facts(
        fake_kb,
        script=script,
        needs=["city_meta", "price", "branches"],
        city_slug="perm",
        branch_slug=None,
        want_city_choices=False,
    )
    assert facts["city"]["city"] == "Пермь"
    assert facts["price"]["branch"] == "unreliable"
    assert "43900" in facts["price_line"]
    assert len(facts["branches"]) == 4
    assert [c["call"] for c in journal] == ["get_city", "list_branches"]


async def test_недоступный_справочник_не_роняет_ход(script):
    from tests.conftest import FakeKB

    kb = FakeKB(cities=[], city=None, branches=[])
    facts, journal = await collect_facts(
        kb,
        script=script,
        needs=["city_meta", "price"],
        city_slug="perm",
        branch_slug=None,
        want_city_choices=True,
    )
    assert facts["price"]["branch"] == "no_amount"
    assert "city" not in facts
    assert journal


async def test_слаг_города_из_перечисления_принимается(fake_kb):
    assert await confirm_city(fake_kb, "perm") == "perm"


async def test_название_города_разбирается_быстрым_путём(fake_kb):
    assert await confirm_city(fake_kb, "Пермь") == "perm"


async def test_город_вне_сети_не_подтверждается(fake_kb):
    assert await confirm_city(fake_kb, "Москва") is None
    assert await confirm_city(fake_kb, "") is None


async def test_филиал_проверяется_по_перечислению_города(fake_kb):
    assert await confirm_branch(fake_kb, "perm", "perm_chernyshevskogo") == "perm_chernyshevskogo"
    assert await confirm_branch(fake_kb, "perm", "выдуманный") is None
    assert await confirm_branch(fake_kb, None, "perm_chernyshevskogo") is None


def test_context_block_пустой():
    assert context_block("") == ""
    assert "Пермь" in context_block("город Пермь")
