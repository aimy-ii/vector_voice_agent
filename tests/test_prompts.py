"""Тесты сборки промпта и подготовки фактов справочника."""

from __future__ import annotations

import re

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from core.config import settings
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
    SPEECH_RULES,
    _describe_step,
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

#: Число правил речи — замена формулировок не увеличивает набор.
_SPEECH_RULES_COUNT = 19

#: Обращения к модели на «ты» (после удаления цитат-примеров в «ёлочках»).
_TY_ADDRESS = re.compile(
    r"(?i)(?<![\w])(ты|тебе|тебя|тобой|твой|твоя|твоё|твое|твои|твою|"
    r"твоей|твоего|твоим|твоих|твоём|твоем)(?![\w])"
)


def _strip_guillemets(text: str) -> str:
    """Убирает фрагменты в «ёлочках» — там примеры запрещённых форм, не обращение."""
    return re.sub(r"«[^»]*»", "", text)


def test_подстановка_фактов_в_текст():
    assert (
        fill_facts("Стоимость: {price_line}", {"price_line": "от 43900"}) == "Стоимость: от 43900"
    )


def test_неизвестный_плейсхолдер_не_роняет_ход():
    assert fill_facts("Адрес: {branch_address}", {}) == "Адрес:"


def test_персона_из_настроек(monkeypatch):
    """persona_block берёт имя, компанию, роль и тон из настроек."""
    monkeypatch.setattr(settings, "agent_name", "Анна")
    monkeypatch.setattr(settings, "agent_company", "ТестКомпания")
    monkeypatch.setattr(settings, "agent_role", "менеджер тестовой школы")
    monkeypatch.setattr(settings, "agent_tone", "спокойный и чёткий")
    block = persona_block()
    assert "Анна" in block
    assert "ТестКомпания" in block
    assert "менеджер тестовой школы" in block
    assert "спокойный и чёткий" in block
    assert "Роль:" in block
    assert "Ты —" not in block
    assert "только на «Вы»" in block


def test_правило_рода_из_agent_gender(monkeypatch):
    """При female — женский род, при male — мужской."""
    monkeypatch.setattr(settings, "agent_gender", "female")
    female = persona_block()
    assert "женском роде" in female
    assert "поняла" in female
    assert "мужском роде" not in female

    monkeypatch.setattr(settings, "agent_gender", "male")
    male = persona_block()
    assert "мужском роде" in male
    assert "понял" in male
    assert "женском роде" not in male


def test_системное_сообщение_на_вы_без_ты_к_модели(script):
    """В системном сообщении есть правило на «Вы» и нет обращений к модели на «ты»."""
    messages = build_turn_messages(
        script=script,
        steps=[script.step("city")],
        profile={},
        facts={},
        history=[],
        asides_done=[],
    )
    content = messages[0].content
    assert "только на «Вы»" in content
    cleaned = _strip_guillemets(content)
    match = _TY_ADDRESS.search(cleaned)
    assert match is None, f"обращение на «ты»: {match.group(0) if match else ''}"


def test_системное_сообщение_содержит_ключевые_правила_речи(script):
    """В SPEECH_RULES — шаг задачи, образцы, открытые вопросы, прощание, тон."""
    messages = build_turn_messages(
        script=script,
        steps=[script.step("city")],
        profile={},
        facts={},
        history=[],
        asides_done=[],
    )
    content = messages[0].content
    assert "только на «Вы»" in content
    assert "шаг из текущей задачи" in content
    assert "форма фразы, а не текст реплики" in content
    assert "Открытых вопросов не задавать" in content
    assert "когда человек сам прощается словами" in content
    assert "Тон разговорный, не рекламный" in content
    assert "Если по ответу человека есть что сказать по делу" in content
    assert "Если реплика клиента бессвязна, оборвана или не отвечает" in content


def test_системное_сообщение_содержит_правила_реакции_и_хвостов(script):
    """Реакция по делу, конец реплики и запрет пустых проверок — замены, не дубли."""
    messages = build_turn_messages(
        script=script,
        steps=[script.step("city")],
        profile={},
        facts={},
        history=[],
        asides_done=[],
    )
    content = messages[0].content
    assert "Если по ответу человека есть что сказать по делу" in content
    assert "без предисловия" in content
    assert "Вежливые пустышки" in content
    assert "Реплика заканчивается движением разговора вперёд" in content
    assert "Выдумывать вопрос, чтобы чем-то закончить, нельзя" in content
    assert "Проверок после рассказа не бывает" in content
    assert "пустой вопрос, на который человек отвечает «да»" in content
    assert "рассказать и задать этот вопрос в одной реплике" in content
    assert "Вопрос звучит так, как спросил бы человек в разговоре" in content
    assert "в живой речи не встречаются" in content
    assert "Каждая реплика заканчивается вопросом или конкретным предложением" not in content
    assert "Проверочный вопрос в конце рассказа спрашивает о том" not in content
    assert "Спрашивать надо предметно" not in content
    assert "Реплика состоит из двух частей" not in content
    assert "Голый вопрос без реакции — ошибка" not in content
    assert "дежурной формулой" not in content
    assert "Вопрос задаётся по-человечески" not in content
    assert "по коробке определились" not in content
    assert "Связка с предыдущей репликой делается по существу" not in content
    assert len(SPEECH_RULES) == _SPEECH_RULES_COUNT


def test_скрипты_v1_v4_собирают_ход_с_правилами_речи(script_v1, script, script_v3, script_v4):
    """Скрипты v1–v4 по-прежнему собирают системное сообщение с правилами."""
    for compiled in (script_v1, script, script_v3, script_v4):
        first = next(iter(compiled.steps))
        messages = build_turn_messages(
            script=compiled,
            steps=[compiled.step(first)],
            profile={},
            facts={},
            history=[],
            asides_done=[],
        )
        assert isinstance(messages[0], SystemMessage)
        content = messages[0].content
        assert "Если по ответу человека есть что сказать по делу" in content
        assert "Реплика заканчивается движением разговора вперёд" in content
        assert "Проверок после рассказа не бывает" in content
        assert "рассказать и задать этот вопрос в одной реплике" in content
        assert "Вопрос звучит так, как спросил бы человек в разговоре" in content
        assert "Каждая реплика заканчивается вопросом или конкретным предложением" not in content
        assert "Проверочный вопрос в конце рассказа спрашивает о том" not in content
        assert "Реплика состоит из двух частей" not in content


def test_профиль_разделяет_роли(script):
    block = profile_block(script, {"caller_name": "Ольга"})
    assert "Ольга" in block
    assert "звонящий" in block
    assert "будущий курсант" in block
    assert "student_name" in block


def test_describe_step_выводит_why_examples_avoid():
    """Непустые why/examples/avoid попадают в описание; пустые метки не печатаются."""
    from script.models import Step

    filled = Step.model_validate(
        {
            "id": "city",
            "kind": "question",
            "goal": "узнать город",
            "why": "без города факты наугад",
            "examples": ["В каком городе удобнее?", "Где планируете учиться?"],
            "avoid": "не зачитывать список городов",
        }
    )
    lines = _describe_step(filled, {}, {}, heading="Шаг")
    text = "\n".join(lines)
    assert "Зачем: без города факты наугад" in text
    assert "Образцы формулировок (не зачитывать дословно, это форма фразы, а не текст):" in text
    assert "В каком городе удобнее?" in text
    assert "Где планируете учиться?" in text
    assert "На этом шаге нельзя: не зачитывать список городов" in text

    bare = Step.model_validate({"id": "name", "kind": "question", "goal": "имя"})
    bare_text = "\n".join(_describe_step(bare, {}, {}, heading="Шаг"))
    assert "Зачем:" not in bare_text
    assert "Образцы формулировок" not in bare_text
    assert "На этом шаге нельзя:" not in bare_text


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
    assert "не оценивать" in lowered and "выбор клиента" in lowered
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
    assert "не зачитывать дословно" in block.lower()
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
    assert "не зачитывать" in content.lower()
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
    assert "незакрытые" in block.lower() or "незакрытая" in block.lower()
    assert "уже спрашивали, ответа нет" in block
    natural = naturalness_block(ask_for_move=True)
    assert "не переспрашивать" in natural.lower()
    assert "верно?" in natural.lower()


def test_промпт_держит_границы_шага_и_незакрытый_вопрос(script):
    """Модель не забегает вперёд и возвращается к незакрытому после побочного."""
    natural = naturalness_block(ask_for_move=True).lower()
    assert "только те вопросы" in natural
    assert "не придумывать" in natural
    assert "не забегать" in natural
    assert "заодно" in natural
    assert "до конца" in natural
    assert "вернуться" in natural
    assert "незакрытое" in natural or "незакрытый вопрос" in natural
    assert "повторите" in natural
    assert "побочные вопросы — норма" not in natural
    messages = build_turn_messages(
        script=script,
        steps=[script.step("name")],
        profile={},
        facts={},
        history=[
            AIMessage(content="Как вас зовут?"),
            HumanMessage(content="повторите, я не услышал"),
        ],
        asides_done=[],
        attempts={"name": 1},
    )
    content = messages[0].content
    assert "только те вопросы" in content.lower()
    assert "незакрытый вопрос не исчезает" in content.lower()
    assert "name" in content
    assert "уже спрашивали, ответа нет" in content
    assert messages[-1].content == "повторите, я не услышал"
    assert messages[-2].content == "Как вас зовут?"


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
    assert "сформулировать в этом ключе" in content


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
    assert "подводить разговор к завершению" in messages[0].content.lower()


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


def test_промпт_запрещает_озвучивать_механику():
    block = persona_block()
    assert "не проговаривается" in block.lower()
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
