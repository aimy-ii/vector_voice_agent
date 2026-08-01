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
    speech_rules_block,
    step_block,
    steps_block,
)

#: Число правил речи, включая пункт 0 про примеры и пункт про опору на диалог.
_SPEECH_RULES_COUNT = 27

#: Начала девяти правил про ведение разговора — подряд после «одна тема».
_LEAD_SPEECH_RULE_STARTS: tuple[str, ...] = (
    "Разговор ведёт агент:",
    "Реплика не заканчивается в никуда",
    "Что пообещал рассказать — обязан рассказать:",
    "Если по текущей теме нового не осталось",
    "Вопрос по делу двигает разговор дальше:",
    "Вопросы вида «что бы Вы хотели узнать»",
    "Разрешения продолжать не спрашивают:",
    "Пустые проверки «что скажете?»",
    "Если по текущему пункту сказать нечего",
)

#: Обращения к модели на «ты» (после удаления цитат-примеров в «ёлочках»).
_TY_ADDRESS = re.compile(
    r"(?i)(?<![\w])(ты|тебе|тебя|тобой|твой|твоя|твоё|твое|твои|твою|"
    r"твоей|твоего|твоим|твоих|твоём|твоем)(?![\w])"
)


def _strip_guillemets(text: str) -> str:
    """Убирает фрагменты в «ёлочках» — там примеры запрещённых форм, не обращение."""
    return re.sub(r"«[^»]*»", "", text)


def _lead_speech_rules() -> tuple[str, ...]:
    """Возвращает подряд идущие правила речи про ведение разговора агентом."""
    start = next(
        index
        for index, rule in enumerate(SPEECH_RULES)
        if rule.startswith(_LEAD_SPEECH_RULE_STARTS[0])
    )
    return SPEECH_RULES[start : start + len(_LEAD_SPEECH_RULE_STARTS)]


def _lead_speech_joined() -> str:
    """Склеивает правила ведения разговора для поиска по объединённому тексту."""
    return "\n".join(_lead_speech_rules())


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
    assert "только на «Вы»" in speech_rules_block()


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
    """В SPEECH_RULES — «Вы», тон, прощание, бессвязная реплика, запрет повторов."""
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
    assert "Тон разговорный, не рекламный" in content
    assert "ни свои фразы, ни примеры" in content.lower()
    assert "не прощаться, пока человек сам не прощается" in content.lower()
    assert "Если реплика клиента бессвязна, оборвана или не отвечает" in content
    assert "Не повторять дословно то, что уже говорил" in content


def test_системное_сообщение_содержит_правила_реакции_и_хвостов(script):
    """Запрет пустых проверок, ведение разговора агентом, живой вопрос."""
    messages = build_turn_messages(
        script=script,
        steps=[script.step("city")],
        profile={},
        facts={},
        history=[],
        asides_done=[],
    )
    content = messages[0].content
    assert "Пустые проверки" in content or "пустые проверки" in content.lower()
    assert "согласие с содержанием" in content.lower()
    assert "разговор ведёт агент" in content.lower()
    assert "не спрашивает у собеседника, что тому рассказать" in content.lower()
    assert "Вопрос звучит так, как спросил бы человек в разговоре" in content
    assert "реплика ожидания" in content.lower()
    assert "точное число" in content.lower() or "несколько»" in content
    assert "Реплика всегда заканчивается передачей хода собеседнику" not in content
    assert "Реплика всегда заканчивается обращением к человеку" not in content
    assert "Заканчивать ничем нельзя" not in content
    assert "Короткий возврат хода после рассказа" not in content
    assert "запретом не считается" not in content
    assert "Молчать после рассказа нельзя" not in content
    assert "Каждая реплика заканчивается вопросом или конкретным предложением" not in content
    assert "Проверочный вопрос в конце рассказа спрашивает о том" not in content
    assert "Спрашивать надо предметно" not in content
    assert "Реплика состоит из двух частей" not in content
    assert "Голый вопрос без реакции — ошибка" not in content
    assert "дежурной формулой" not in content
    assert "Вопрос задаётся по-человечески" not in content
    assert "по коробке определились" not in content
    assert "Связка с предыдущей репликой делается по существу" not in content
    assert "закончить самим фактом" not in content
    assert "закончить фактом, без пустого хвоста" not in content
    assert "Проверок после рассказа не бывает" not in content
    assert "Побуждать человека к ответу" not in content
    assert "реплика без вопроса допустима, но не должна быть правилом" not in content
    assert len(SPEECH_RULES) == _SPEECH_RULES_COUNT


def test_правила_ведения_разговора_идут_подряд():
    """Девять правил ведения разговора идут подряд; соседние не сдвинуты."""
    lead = _lead_speech_rules()
    assert len(lead) == len(_LEAD_SPEECH_RULE_STARTS)
    for rule, start in zip(lead, _LEAD_SPEECH_RULE_STARTS, strict=True):
        assert rule.startswith(start)
    topic_index = next(
        index for index, rule in enumerate(SPEECH_RULES) if "одна тема за реплику" in rule.lower()
    )
    lead_index = SPEECH_RULES.index(lead[0])
    assert lead_index == topic_index + 1
    after = SPEECH_RULES[lead_index + len(lead)]
    assert after.startswith("Вопрос звучит так, как спросил бы человек")
    joined = _lead_speech_joined().lower()
    assert "вопрос по делу" in joined
    assert "реплика ожидания" in joined
    assert "согласие с содержанием" in joined
    assert "запретом не считается" not in joined
    assert "Заканчивать ничем нельзя" not in joined
    assert "Побуждать человека к ответу" not in joined


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
        assert "Пустые проверки" in content or "пустые проверки" in content.lower()
        assert "согласие с содержанием" in content.lower()
        assert "Вопрос звучит так, как спросил бы человек в разговоре" in content
        assert "разговор ведёт агент" in content.lower()
        assert "Реплика всегда заканчивается обращением к человеку" not in content
        assert "Заканчивать ничем нельзя" not in content
        assert "Реплика всегда заканчивается передачей хода собеседнику" not in content
        assert "запретом не считается" not in content
        assert "Каждая реплика заканчивается вопросом или конкретным предложением" not in content
        assert "Проверочный вопрос в конце рассказа спрашивает о том" not in content
        assert "Реплика состоит из двух частей" not in content
        assert "закончить фактом, без пустого хвоста" not in content
        assert "Побуждать человека к ответу" not in content


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
    assert "**Примеры**" in text
    assert "В каком городе удобнее?" in text
    assert "Где планируете учиться?" in text
    assert "На этом шаге нельзя: не зачитывать список городов" in text

    bare = Step.model_validate({"id": "name", "kind": "question", "goal": "имя"})
    bare_text = "\n".join(_describe_step(bare, {}, {}, heading="Шаг"))
    assert "Зачем:" not in bare_text
    assert "**Примеры**" not in bare_text
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
    block = steps_block([step], {}, {})
    assert "Образец смысла проверки" in block
    assert step.check_question in block
    assert f"«{step.check_question}»" in block
    assert "не зачитывать дословно" in block.lower()
    natural = naturalness_block(ask_for_move=True).lower()
    assert "своими словами" in natural
    assert "одинаковый конец не повторяется" in natural or "каждый раз своими словами" in natural
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


def test_промпт_видит_шапку_без_пометок_попыток(script):
    """Шапка перечисляет шаги без пометок «новый» / «уже спрашивали»."""
    block = steps_block(
        [script.step("city"), script.step("who_studies")],
        {},
        {},
    )
    assert script.step("city").goal in block
    assert script.step("who_studies").goal in block
    assert "**Требования**" in block
    assert "новый вопрос" not in block
    assert "уже спрашивали, ответа нет" not in block
    assert "впереди — не забегать" not in block
    natural = naturalness_block(ask_for_move=True)
    assert "не переспрашивать" in natural.lower()
    assert "верно?" in natural.lower()


def test_промпт_держит_границы_шага_и_незакрытый_вопрос(script):
    """После побочного обмена — вернуться к делу; историю не терять."""
    natural = naturalness_block(ask_for_move=True).lower()
    assert "вернуться к делу" in natural or "мягко вернуться" in natural
    assert "до конца" in natural
    assert "повторите" in natural
    assert "своих вопросов не придумывать" not in natural
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
    )
    content = messages[0].content
    assert "вернуться к делу" in natural or "мягко вернуться" in natural
    assert script.step("name").goal in content or "name" in content.lower()
    assert "уже спрашивали, ответа нет" not in content
    assert "новый вопрос" not in content
    assert messages[-1].content == "повторите, я не услышал"
    assert messages[-2].content == "Как вас зовут?"


def test_промпт_требует_заканчивать_ходом_к_собеседнику(script):
    """При незакрытом вопросе naturalness требует хода; в правилах — ведение."""
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
    content = messages[0].content.lower()
    assert "разговор ведёт агент" in content
    assert "не спрашивает у собеседника, что тому рассказать" in content


def test_шапка_с_образцом_тоже_требует_хода_к_собеседнику(script):
    """Незакрытый вопрос в перечне — ask_for_move и для шага с образцом."""
    messages = build_turn_messages(
        script=script,
        steps=[script.step("terms")],
        profile={},
        facts={},
        history=[],
        asides_done=[],
    )
    content = messages[0].content.lower()
    assert "разговор ведёт агент" in content
    assert "сформулировать в этом ключе" in content


def test_шапка_с_висящими_и_образцом_одним_промптом(script):
    """Несколько шагов в шапке + образец у текущего — всё в одном системном."""
    from graph.prompts import _SAMPLE_PREFIX

    messages = build_turn_messages(
        script=script,
        steps=[script.step("terms"), script.step("city")],
        profile={},
        facts={"price_line": "срок два месяца"},
        history=[],
        asides_done=[],
    )
    content = messages[0].content
    assert script.step("city").goal in content
    assert script.step("terms").goal in content
    assert _SAMPLE_PREFIX in content
    assert "# СЕЙЧАС ГОВОРИМ ОБ ЭТОМ" in content
    assert "**Требования**" in content
    assert "Ведущий шаг" not in content
    assert "Висящий шаг" not in content
    assert "новый вопрос" not in content
    assert "уже спрашивали, ответа нет" not in content


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
    how_i = content.index("# КАК РАБОТАТЬ")
    rules_i = content.index("# ПРАВИЛА РЕЧИ")
    ctx_i = content.index("# КОНТЕКСТ")
    steps_i = content.index("# СЕЙЧАС ГОВОРИМ ОБ ЭТОМ")
    form_i = content.index("# ФОРМА ОТВЕТА")
    assert how_i < rules_i < ctx_i < steps_i < form_i
    assert content.index("Дарья") > how_i
    assert content.index("Что уже известно") > ctx_i


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
    """Полная сборка: системное сообщение + вся переданная история."""
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
    assert len(messages) == 13
    assert messages[1].content == "реплика 0"
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


def test_шапка_перечень_без_запрета_нового_вопроса(script):
    """Шапка — перечень уместных тем; нет запрета «новых вопросов нет»."""
    messages = build_turn_messages(
        script=script,
        steps=[script.step("name"), script.step("city")],
        profile={},
        facts={},
        history=[],
        asides_done=[],
    )
    content = messages[0].content
    lowered = content.lower()
    assert "# сейчас говорим об этом" in lowered
    assert "**требования**" in lowered
    assert "ни одного нового вопроса не задавать" not in lowered
    assert "новый вопрос — только один" not in lowered
    assert "из переданных данных как есть" in lowered


def test_шапка_правило_гарантированного_хода_и_жёсткий_запрет(script):
    """В системном сообщении — шаги в работе и запрет выдуманных фактов в правилах."""
    messages = build_turn_messages(
        script=script,
        steps=[script.step("name"), script.step("city")],
        profile={},
        facts={},
        history=[],
        asides_done=[],
    )
    content = messages[0].content.lower()
    assert "# сейчас говорим об этом" in content
    assert "из переданных данных как есть" in content
    assert "по последней реплике собеседника" not in content
    assert "новый вопрос" not in content
    assert "уже спрашивали, ответа нет" not in content
    assert "впереди — не забегать" not in content


def test_naturalness_pending_only_отличается_от_обычного():
    """pending_only меняет пункт про ход к собеседнику, остальное общее."""
    ordinary = naturalness_block(ask_for_move=True, pending_only=False)
    pending = naturalness_block(ask_for_move=True, pending_only=True)
    assert ordinary != pending
    assert "вопросом или конкретным" in ordinary.lower()
    assert "висящему" in pending.lower()
    assert "новую тему" in pending.lower()
    assert "не оценивать" in ordinary.lower() and "не оценивать" in pending.lower()


def test_build_turn_messages_история_целиком(script):
    """Полная сборка кладёт всю переданную историю, без обрезки хвоста."""
    history = [
        HumanMessage(content=f"реплика {i}") if i % 2 == 0 else AIMessage(content=f"ответ {i}")
        for i in range(40)
    ]
    messages = build_turn_messages(
        script=script,
        steps=[script.step("name")],
        profile={},
        facts={},
        history=history,
        asides_done=[],
    )
    assert isinstance(messages[0], SystemMessage)
    assert messages[0].content
    assert len(messages) - 1 == 40
    assert messages[1].content == "реплика 0"
    assert messages[-1].content == "ответ 39"


def test_naturalness_правило_подводки_к_теме(script):
    """Одна тема за реплику; структурирующих подводок в правилах нет."""
    block = naturalness_block(ask_for_move=True)
    lowered = block.lower()
    assert "подводк" not in lowered
    assert "одна тема" in speech_rules_block().lower() or "одна тема" in lowered
    # «одна тема» живёт в SPEECH_RULES.
    assert "одна тема" in speech_rules_block().lower()
    messages = build_turn_messages(
        script=script,
        steps=[script.step("name")],
        profile={},
        facts={},
        history=[],
        asides_done=[],
    )
    content = messages[0].content.lower()
    assert "подводк" not in content
    assert "одна тема" in content


def test_naturalness_не_пересказывать_ответ_клиента(script):
    """Запрет пересказа сказанного клиентом — сразу по делу."""
    text = naturalness_block(ask_for_move=True).lower()
    assert "не повторять" in text
    assert "подтверждение выбора" in text or "вы выбрали" in text
    assert "вы выбрали механику" in text
    assert "сразу по делу" in text
    messages = build_turn_messages(
        script=script,
        steps=[script.step("name")],
        profile={},
        facts={},
        history=[],
        asides_done=[],
    )
    content = messages[0].content.lower()
    assert "не повторять" in content
    assert "не пересказывает сказанное клиентом" in content or "начало — сразу с сути" in content


def test_naturalness_запрет_пустых_проверок_в_конце(script):
    """Пустые проверки в конце запрещены; naturalness — ход при незакрытом вопросе."""
    text = naturalness_block(ask_for_move=True).lower()
    assert "пустая проверка" in text
    assert "продолжу?" in text
    assert "теорию удобнее" in text
    assert "обращением к человеку" in text
    assert "одинаковый конец не повторяется" in text
    messages = build_turn_messages(
        script=script,
        steps=[script.step("name")],
        profile={},
        facts={},
        history=[],
        asides_done=[],
    )
    content = messages[0].content.lower()
    assert "пустые проверки" in content or "пустая проверка" in content
    assert "разговор ведёт агент" in content
    assert "не спрашивает у собеседника, что тому рассказать" in content


def test_naturalness_добавлять_новое_а_не_пересказывать_вопрос(script):
    """Ответ обязан добавить новое; пересказ вопроса клиента — не ответ."""
    text = naturalness_block(ask_for_move=True).lower()
    assert "ещё не знает" in text
    assert "тавтология" in text or "другими словами — не ответ" in text
    assert "ручной коробкой" in text
    assert "лада гранта" in text
    messages = build_turn_messages(
        script=script,
        steps=[script.step("name")],
        profile={},
        facts={},
        history=[],
        asides_done=[],
    )
    # В полной сборке правило — в naturalness_block; в системном — SPEECH_RULES.
    assert "ещё не знает" in text
    content = messages[0].content.lower()
    assert "одна тема" in content


def test_naturalness_называть_данные_сразу_из_контекста(script):
    """Данные из контекста произносятся в этой же реплике, без отложенного обещания."""
    text = naturalness_block(ask_for_move=True).lower()
    assert "сейчас" in text and "целиком" in text
    assert "назову" in text
    assert "комендантская" in text
    assert "коломяжский" in text
    messages = build_turn_messages(
        script=script,
        steps=[script.step("name")],
        profile={},
        facts={},
        history=[],
        asides_done=[],
    )
    assert "целиком" in text
    content = messages[0].content.lower()
    assert "не обещать сходить за данными" in content


def test_обращение_по_имени_одно_жёсткое_правило(script):
    """Правило про имя — одно, в естественности: не с имени и не подряд."""
    natural = naturalness_block(ask_for_move=True)
    assert natural.lower().count("имя звучит один раз") == 1
    assert "начинать реплику с имени нельзя" in natural.lower()
    assert "две реплики подряд с именем" in natural.lower()
    assert "закреплении места" in natural.lower()
    assert "прощании" in natural.lower()
    assert "на «Вы» без имени" in natural or "на «вы» без имени" in natural.lower()

    persona = persona_block().lower()
    assert "имя звучит один раз" not in persona
    assert "начинать реплику с имени" not in persona
    assert "две реплики подряд с именем" not in persona
    assert "по имени — редко" not in persona
    assert "обращение по имени и только по имени" not in persona

    messages = build_turn_messages(
        script=script,
        steps=[script.step("name")],
        profile={},
        facts={},
        history=[],
        asides_done=[],
    )
    content = messages[0].content.lower()
    assert "имя звучит один раз" not in content  # только в naturalness_block
    assert "по имени — редко" not in content
    assert "колл-центр" not in content
    assert natural.lower().count("имя звучит один раз") == 1


def test_unknown_запрет_выдумывать_филиал(script):
    """Адрес филиала — только дословно из контекста."""
    from graph.prompts import unknown_block

    text = unknown_block(script).lower()
    assert "филиал" in text
    assert "дословно" in text
    assert "энгельса" in text or "просвещения" in text
    assert "уточню" in text or "уточнишь" in text


def test_closed_steps_в_системном_сообщении(script):
    """Закрытые шаги попадают в системное сообщение с запретом пересказывать."""
    from graph.prompts import build_turn_messages

    closed = [script.step("included"), script.step("practice")]
    messages = build_turn_messages(
        script=script,
        steps=[script.step("branch")],
        profile={"city": "Пермь"},
        facts={},
        history=[],
        asides_done=[],
        closed_steps=closed,
    )
    text = messages[0].content.lower()
    assert "закрытые шаги" in text
    assert "пересказывать" in text or "повторять" in text
    included_name = (getattr(script.step("included"), "goal", None) or "included").lower()
    practice_name = (getattr(script.step("practice"), "goal", None) or "practice").lower()
    # Названия шагов или их идентификаторы.
    assert "included" in text or any(w in text for w in included_name.split()[:2])
    assert "practice" in text or any(w in text for w in practice_name.split()[:2])


def test_правило_адресов_требует_дословного_присутствия(script):
    """Правило про адреса — только дословная строка из контекста."""
    rule = next(r for r in SPEECH_RULES if "утверждать наличие филиала" in r.lower())
    text = rule.lower()
    assert "дословно" in text
    assert "не называть адресов" in text
    assert "уточню" not in text and "уточнишь" not in text
    assert "плохо:" not in text and "хорошо:" not in text


def test_profile_block_pending_в_уточняется_не_в_неизвестном(script):
    from graph.prompts import profile_block

    text = profile_block(
        script,
        {"caller_name": "Мария"},
        pending_fields=["city"],
    )
    assert "уточняется" in text.lower()
    assert "city" in text
    # В секции неизвестного city не дублируется.
    unknown_part = text.split("Чего ещё не знаем")[-1] if "Чего ещё не знаем" in text else ""
    assert "- city" not in unknown_part


def test_build_waiting_messages_с_контекстом_и_без(script):
    """Waiting кладёт context_text в системное; пустой не ломает; лимит истории свой."""
    from langchain_core.messages import AIMessage, HumanMessage

    from graph.prompts import build_turn_messages, build_waiting_messages

    history = [
        AIMessage(content="В каком городе?"),
        HumanMessage(content="Пермь"),
        AIMessage(content="Какой район?"),
        HumanMessage(content="Просвещения"),
        AIMessage(content="Секунду."),
        HumanMessage(content="Ну что?"),
    ]
    waiting = build_waiting_messages(
        script,
        messages=history,
        profile={"city": "Пермь"},
        pending_fields=[],
        step=script.step("name"),
        history_limit=4,
        context_text="Статика: Пермь, автопарк Solaris",
    )
    empty_ctx = build_waiting_messages(
        script,
        messages=history,
        profile={"city": "Пермь"},
        pending_fields=[],
        step=script.step("name"),
        history_limit=4,
        context_text="",
    )
    full = build_turn_messages(
        script=script,
        steps=[script.step("branch")],
        profile={"city": "Пермь"},
        facts={"price_line": "Стоимость — 43900"},
        history=history,
        asides_done=[],
        context_text="Статика: Пермь, автопарк Solaris",
        pending_fields=["branch"],
    )
    content = waiting[0].content
    assert "Статика: Пермь" in content
    assert "автопарк Solaris" in content
    assert "Статика:" not in empty_ctx[0].content
    assert "43900" not in content
    assert "В истории — весь разговор" not in content
    assert "Требования:" not in content
    assert "предмет" in content.lower()
    assert "восьми слов" in content.lower() or "восемь слов" in content.lower()
    assert "придаточн" in content.lower()
    assert len(waiting) - 1 == 4
    assert len(content) < len(full[0].content)


def test_build_filler_messages_короче_без_статики_и_примеров(script):
    """Живая реакция: без статики/фактов/шапки/примеров, короче waiting."""
    from langchain_core.messages import AIMessage, HumanMessage

    from core.config import settings
    from graph.prompts import build_filler_messages, build_waiting_messages

    history = [
        AIMessage(content="Как вас зовут?"),
        HumanMessage(content="Меня Андрей зовут"),
        AIMessage(content="В каком городе?"),
        HumanMessage(content="Пермь"),
    ]
    filler = build_filler_messages(
        script,
        messages=history,
        history_limit=settings.filler_history_limit,
    )
    waiting = build_waiting_messages(
        script,
        messages=history,
        profile={"caller_name": "Андрей"},
        pending_fields=[],
        step=script.step("name"),
        history_limit=settings.waiting_history_limit,
    )
    content = filler[0].content
    assert "Статика" not in content
    assert "Шапка" not in content
    assert "Факты" not in content
    assert "сейчас подберу" not in content.lower()
    assert "например" not in content.lower()
    assert len(content) * 2 < len(waiting[0].content)
    assert len(filler) - 1 == settings.filler_history_limit
    assert filler[-1].content == "Пермь"


def test_build_silence_messages_короче_опирается_на_сказанное(script):
    """Молчание: без статики/фактов/шапки; опора на сказанное; запрет новых фактов."""
    from langchain_core.messages import AIMessage, HumanMessage

    from core.config import settings
    from graph.prompts import build_silence_messages, build_turn_messages

    history = [
        AIMessage(content="Как вас зовут?"),
        HumanMessage(content="Мария"),
        AIMessage(content="В каком городе будете учиться?"),
        HumanMessage(content="Пермь"),
        AIMessage(content="Обучение под ключ, доплат не будет. Теорию удобнее очно?"),
        HumanMessage(content="очно"),
    ]
    silence = build_silence_messages(
        script,
        messages=history,
        profile={"caller_name": "Мария", "city": "Пермь"},
        step=script.step("theory_format"),
        attempt=1,
        history_limit=settings.silence_history_limit,
    )
    full = build_turn_messages(
        script=script,
        steps=[script.step("theory_format")],
        profile={"caller_name": "Мария", "city": "Пермь"},
        facts={"price_line": "Стоимость — 43900"},
        history=history,
        asides_done=[],
        context_text="Статика: Пермь, автопарк Solaris",
    )
    content = silence[0].content
    lowered = content.lower()
    assert "Статика:" not in content
    assert "43900" not in content
    assert "В истории — весь разговор" not in content
    assert "Требования:" not in content
    assert "Факты этого хода" not in content
    assert "опираясь" in lowered or "о чём только что говорили" in lowered
    assert "новыми фактами" in lowered or "новых фактов" in lowered
    assert "вернуть" in lowered
    assert "молчание не означает" in lowered
    assert "бронировать" in lowered
    assert "решать за человека" in lowered
    assert "например" not in lowered
    assert "дежурные оклики" in lowered
    assert "алло" in lowered
    assert "вы тут" in lowered
    assert len(silence) - 1 == len(history)
    assert len(silence) - 1 > 4
    assert silence[-1].content == "очно"
    assert len(content) * 2 < len(full[0].content)

    long_history = [
        msg
        for i in range(7)
        for msg in (
            AIMessage(content=f"вопрос {i}?"),
            HumanMessage(content=f"ответ {i}"),
        )
    ]
    silence_long = build_silence_messages(
        script,
        messages=long_history,
        profile={},
        step=None,
        attempt=1,
        history_limit=settings.silence_history_limit,
    )
    assert len(silence_long) - 1 == settings.silence_history_limit
    assert settings.silence_history_limit > 4

    first = build_silence_messages(
        script,
        messages=history,
        profile={},
        step=None,
        attempt=1,
        history_limit=4,
    )[0].content.lower()
    second = build_silence_messages(
        script,
        messages=history,
        profile={},
        step=None,
        attempt=2,
        history_limit=4,
    )[0].content.lower()
    assert "первая попытка" in first
    assert "мягко" in first
    assert "сформулировать иначе" not in first
    assert "с другой стороны" in second
    assert "сформулировать иначе" in second
    assert "повторять" in second
    assert "первая попытка" not in second
    assert "мягко верни" not in second


def test_шапка_текущим_помечен_первый_не_последний(script):
    """Первый шаг — в «СЕЙЧАС», второй — в «ЕЩЁ НЕ ЗАКРЫТО»; без старых пометок."""
    t1, t2 = script.step("transmission"), script.step("terms")
    block = steps_block([t1, t2], {}, {})
    assert "# СЕЙЧАС ГОВОРИМ ОБ ЭТОМ" in block
    assert "# ЕЩЁ НЕ ЗАКРЫТО" in block
    assert f"## {t1.goal}" in block
    assert f"## {t2.goal}" in block
    lead_i = block.index(f"## {t1.goal}")
    hang_i = block.index(f"## {t2.goal}")
    assert lead_i < hang_i
    assert block.index("# СЕЙЧАС ГОВОРИМ ОБ ЭТОМ") < block.index("# ЕЩЁ НЕ ЗАКРЫТО")
    assert "**Требования**" in block
    assert "Ведущий шаг" not in block
    assert "Висящий шаг" not in block
    assert "новый вопрос" not in block
    assert "уже спрашивали" not in block


def test_step_block_с_next_step_текущим_ведущий(script):
    """step_block с next_step: первый в перечне, следующий — ориентир дальше."""
    block = step_block(
        script.step("terms"),
        {},
        {},
        next_step=script.step("theory_format"),
    )
    terms, nxt = script.step("terms"), script.step("theory_format")
    assert f"## {terms.goal}" in block
    assert f"## {nxt.goal}" in block
    assert block.index(f"## {terms.goal}") < block.index(f"## {nxt.goal}")
    # next_step — без примеров theory_format, если они есть только у следующего
    after_next = block.split(f"## {nxt.goal}", 1)[1]
    assert "**Требования**" in after_next
    assert "Ведущий шаг" not in block
    assert "новый вопрос" not in block


def test_шапка_не_убегать_вперёд_предпочтение(script):
    """Текущий шаг отрабатывают; предпочтительного порядка в тексте нет."""
    block = steps_block(
        [script.step("city"), script.step("who_studies")],
        {},
        {},
    )
    lowered = block.lower()
    assert "**требования**" in lowered
    assert script.step("city").goal.lower() in lowered
    assert "уже спрашивали, ответа нет" not in lowered
    assert "предпочтительн" not in lowered
    assert "необязательн" not in lowered
    assert "подсказка, куда двигаться" not in lowered
    messages = build_turn_messages(
        script=script,
        steps=[script.step("city"), script.step("who_studies")],
        profile={},
        facts={},
        history=[],
        asides_done=[],
    )
    content = messages[0].content.lower()
    assert "# сейчас говорим об этом" in content
    assert "# ещё не закрыто" in content
    assert script.step("city").goal.lower() in content
    assert script.step("who_studies").goal.lower() in content
    assert "отработать, а не произнести" in content
    assert "реплика строится по шагу" not in content


def test_ведущий_и_следующий_разные_роли_только_вопрос(script):
    """Первый в работе и «что дальше» — разные роли; следующий не в шагах."""
    included, nxt = script.step("included"), script.step("theory_format")
    block = step_block(included, {}, {}, next_step=nxt)
    assert f"## {included.goal}" in block
    assert f"## {nxt.goal}" in block
    assert block.index(f"## {included.goal}") < block.index(f"## {nxt.goal}")
    assert "**Требования**" in block
    # В steps_block next_step не попадает.
    only_head = steps_block([included], {}, {})
    assert f"## {nxt.goal}" not in only_head


def test_запрет_раскрывать_содержание_следующего_шага(script):
    """Шаг не диктует реплику; порядок — подсказка, не запрет рассказать следующее."""
    for phrase in (
        "реплика строится по ведущему шагу",
        "единственный источник содержания",
        "содержание висящих и следующего шага не рассказывать",
        "не раскрывая его содержание",
    ):
        assert not any(phrase in rule.lower() for rule in SPEECH_RULES)
    messages = build_turn_messages(
        script=script,
        steps=[script.step("included")],
        profile={},
        facts={},
        history=[],
        asides_done=[],
        next_step=script.step("theory_format"),
    )
    content = messages[0].content.lower()
    assert "# что дальше" in content
    assert "не раскрывая его содержание" not in content
    assert script.step("theory_format").goal.lower() in content


def test_ход_к_собеседнику_и_запрет_пустых_проверок_вместе(script):
    """Ведение разговора агентом и запрет пустых проверок — в правилах."""
    joined = _lead_speech_joined().lower()
    assert "разговор ведёт агент" in joined
    assert "продолжим" in joined
    assert "вопрос по делу" in joined
    natural = naturalness_block(ask_for_move=True)
    assert "Пустая проверка" in natural
    assert "либо ничем" not in natural.lower()


def test_запрет_повторять_и_разворачивать_уже_сказанное(script):
    """Уже сказанное в разговоре не повторять дословно — ни свои фразы, ни образцы."""
    rule = next(r for r in SPEECH_RULES if "не повторять дословно" in r.lower())
    assert "свои фразы" in rule.lower()
    assert "примеры" in rule.lower()
    messages = build_turn_messages(
        script=script,
        steps=[script.step("included")],
        profile={},
        facts={},
        history=[],
        asides_done=[],
    )
    content = messages[0].content.lower()
    assert "не повторять дословно" in content
    assert "ни свои фразы, ни примеры" in content


def test_реплика_всегда_ходом_и_запрет_пустых_проверок(script):
    """Реплика ведёт разговор дальше; пустые проверки и разрешения запрещены."""
    joined = _lead_speech_joined().lower()
    assert "разговор ведёт агент" in joined
    assert "не заканчивается в никуда" in joined
    assert "продолжим" in joined
    natural = naturalness_block(ask_for_move=True)
    assert "Пустая проверка" in natural
    assert "либо ничем" not in natural.lower()


def test_правило_речи_без_разрешения_продолжать(script):
    """Агент ведёт разговор; запрет разрешений; переход к следующему пункту."""
    joined = _lead_speech_joined().lower()
    assert "разрешения продолжать не спрашивают" in joined
    assert "рассказать подробнее" in joined
    assert "не запрет на вопросы вообще" in joined
    assert "говорить ли дальше" in joined
    assert "разговор ведёт агент" in joined
    assert "переходить к следующему" in joined
    assert "следующему пункту перечня" not in joined
    assert "Реплика всегда заканчивается обращением к человеку" not in joined
    assert "Заканчивать ничем нельзя" not in joined
    assert "Побуждать человека к ответу" not in joined
    messages = build_turn_messages(
        script=script,
        steps=[script.step("city")],
        profile={},
        facts={},
        history=[],
        asides_done=[],
    )
    content = messages[0].content
    content_lower = content.lower()
    assert "разрешения продолжать не спрашивают" in content_lower
    assert "не запрет на вопросы вообще" in content_lower
    assert "разговор ведёт агент" in content_lower
    assert "переходить к следующему" in content_lower
    assert "следующему пункту перечня" not in content_lower
    assert "Реплика всегда заканчивается обращением к человеку" not in content
    assert "Заканчивать ничем нельзя" not in content


def test_вопрос_в_конце_двигает_разговор_а_не_проверяет_понимание(script):
    """Вопрос по делу двигает разговор; пустые «что интересует» запрещены."""
    joined = _lead_speech_joined().lower()
    assert "двигает разговор дальше" in joined
    assert "выбор из двух вариантов" in joined
    assert "уточнение недостающего" in joined
    assert "предложение следующего шага" in joined
    assert "не проверка понимания" in joined
    assert "что осталось непонятным" in joined
    assert "что вас интересует" in joined
    assert "что бы вы хотели узнать" in joined
    assert "о чём рассказать" in joined
    assert "побуждать человека к ответу" not in joined
    assert "реплика заканчивается вопросом или конкретным предложением" not in joined
    messages = build_turn_messages(
        script=script,
        steps=[script.step("city")],
        profile={},
        facts={},
        history=[],
        asides_done=[],
    )
    content = messages[0].content.lower()
    assert "двигает разговор дальше" in content
    assert "не запрет на вопросы вообще" in content
    assert "разговор ведёт агент" in content


def test_имя_повторы_и_одна_тема_не_тронуты_усилением_хвоста():
    """Правила про имя, повторы и одну тему — прежние формулировки."""
    name_in_natural = naturalness_block(ask_for_move=True).lower()
    assert "имя звучит один раз" in name_in_natural
    assert "начинать реплику с имени нельзя" in name_in_natural
    assert "две реплики подряд с именем" in name_in_natural
    repeat_rule = next(r for r in SPEECH_RULES if "не повторять дословно" in r.lower())
    assert "ни свои фразы, ни примеры" in repeat_rule.lower()
    topic_rule = next(r for r in SPEECH_RULES if "одна тема за реплику" in r.lower())
    assert "два-три коротких предложения" in topic_rule.lower()
    assert len(SPEECH_RULES) == _SPEECH_RULES_COUNT


def test_правило_речи_ведёт_разговор_а_не_отдаёт_ход(script):
    """Агент ведёт разговор: три вида продолжения, обещание, без пустых вопросов."""
    joined = _lead_speech_joined().lower()
    assert "разговор ведёт агент" in joined
    assert "не спрашивает у собеседника, что тому рассказать" in joined
    assert "вопрос по делу" in joined
    assert "переход к следующей теме" in joined
    assert "обозначение того, о чём пойдёт речь" in joined
    assert "что пообещал рассказать — обязан рассказать" in joined
    assert "следующая реплика начинается именно с этого" in joined
    assert "что вас интересует" in joined
    assert "что осталось непонятным" in joined
    assert "молчание собеседника после реплики не проблема" not in joined
    assert "вымогать ответ вопросом ради вопроса" not in joined
    assert "если по текущей теме нового не осталось" in joined
    assert "разрешения продолжать не спрашивают" in joined
    assert "реплика заканчивается вопросом или конкретным предложением" not in joined
    assert "побуждать человека к ответу" not in joined
    assert "обязательным такой конец не является" not in joined
    messages = build_turn_messages(
        script=script,
        steps=[script.step("city")],
        profile={},
        facts={},
        history=[],
        asides_done=[],
    )
    content = messages[0].content.lower()
    assert "разговор ведёт агент" in content
    assert "не спрашивает у собеседника, что тому рассказать" in content
    assert "что пообещал рассказать — обязан рассказать" in content
    assert "что вас интересует" in content
    assert "что осталось непонятным" in content
    assert "каждая реплика заканчивается вопросом или конкретным предложением" not in content
    assert "побуждать человека к ответу" not in content


def test_исключение_ожидания_без_хода_к_собеседнику(script):
    """Сборка ожидания — исключение: ход к человеку не требуется."""
    from graph.prompts import build_waiting_messages

    end_rule = next(rule for rule in SPEECH_RULES if "реплика ожидания" in rule.lower())
    assert "ход к человеку не нужен" in end_rule.lower()
    waiting = build_waiting_messages(
        script,
        messages=[HumanMessage(content="Пермь")],
        profile={"city": "Пермь"},
        pending_fields=[],
        step=script.step("branch"),
        history_limit=2,
    )
    content = waiting[0].content.lower()
    assert "заканчивать ничем нельзя" not in content
    assert "всегда заканчивается обращением к человеку" not in content


def test_точные_данные_вместо_расплывчатых(script):
    """Точное число/адрес/срок/цена — называть, не подменять «несколькими»."""
    rule = next(r for r in SPEECH_RULES if "точное число" in r.lower())
    assert "несколько" in rule
    assert "плохо:" not in rule.lower()
    assert "хорошо:" not in rule.lower()
    messages = build_turn_messages(
        script=script,
        steps=[script.step("city")],
        profile={},
        facts={},
        history=[],
        asides_done=[],
    )
    content = messages[0].content
    assert "точное число" in content.lower()
    assert "заменять на «несколько»" in content.lower()


def test_шапка_не_повторять_сказанное_и_ход_к_человеку(script):
    """В рамке перечня — реплика из всего разговора; агент ведёт разговор."""
    block = steps_block([script.step("included")], {}, {}).lower()
    assert "**требования**" in block
    assert script.step("included").goal.lower() in block
    assert "разговор ведёт агент" in speech_rules_block().lower()
    messages = build_turn_messages(
        script=script,
        steps=[script.step("included")],
        profile={},
        facts={},
        history=[],
        asides_done=[],
    )
    content = messages[0].content.lower()
    assert "# сейчас говорим об этом" in content
    assert "одна тема" in content
    assert "разговор ведёт агент" in content


def test_запрет_служебных_слов_в_эфире(script):
    """Служебные слова скрипта вслух не произносятся."""
    rule = next(r for r in SPEECH_RULES if "служебные слова" in r.lower())
    assert "шаг" in rule.lower()
    assert "пункт" in rule.lower()
    assert "скрипт" in rule.lower()
    assert "перечень" in rule.lower()
    assert "нумерованных" in rule.lower() or "нумерованн" in rule.lower()
    messages = build_turn_messages(
        script=script,
        steps=[script.step("name")],
        profile={},
        facts={},
        history=[],
        asides_done=[],
    )
    content = messages[0].content.lower()
    assert "служебные слова" in content
    assert "нумерованных перечислений" in content


def test_naturalness_длина_потолок_не_ориентир(script):
    """Два-три предложения — потолок длины реплики."""
    text = naturalness_block(ask_for_move=True).lower()
    assert "потолок" in text
    assert "два-три" in text
    messages = build_turn_messages(
        script=script,
        steps=[script.step("included")],
        profile={},
        facts={},
        history=[],
        asides_done=[],
    )
    assert "потолок" in messages[0].content.lower()


#: Фразы старой рамки, назначавшие шаг источником реплики.
_STEP_AS_SOURCE_PHRASES: tuple[str, ...] = (
    "реплика строится по ведущему шагу",
    "единственный источник содержания",
    "содержание висящих и следующего шага не рассказывать",
    "не раскрывая его содержание",
    "вперёд по ним не забегать",
    "если по шагам перечня сказать нечего, реплика строится по последней",
    "ведущий шаг",
    "висящий шаг",
)


def test_системное_без_назначения_шага_источником_реплики(script):
    """В системном сообщении нет формулировок, где шаг диктует реплику."""
    messages = build_turn_messages(
        script=script,
        steps=[script.step("included"), script.step("practice")],
        profile={},
        facts={},
        history=[],
        asides_done=[],
        next_step=script.step("theory_format"),
    )
    content = messages[0].content.lower()
    for phrase in _STEP_AS_SOURCE_PHRASES:
        assert phrase not in content, phrase


def test_рамка_разговор_целиком_порядок_подсказка_ответ_на_вопрос(script):
    """Есть рамка: весь разговор, порядок как подсказка, ответить на вопрос."""
    messages = build_turn_messages(
        script=script,
        steps=[script.step("city")],
        profile={},
        facts={},
        history=[],
        asides_done=[],
    )
    content = messages[0].content.lower()
    assert "# сейчас говорим об этом" in content
    assert "**требования**" in content
    assert "разговор ведёт агент" in content


def test_образцы_всех_шагов_шапки_полностью_и_в_порядке(script_v4):
    """Примеры только у текущего шага; у незакрытого примеров нет."""
    current = script_v4.step("city")
    hang = script_v4.step("who_studies")
    assert current.examples and hang.examples
    messages = build_turn_messages(
        script=script_v4,
        steps=[current, hang],
        profile={},
        facts={},
        history=[],
        asides_done=[],
    )
    content = messages[0].content
    now = content.split("# СЕЙЧАС ГОВОРИМ ОБ ЭТОМ", 1)[1].split("# ЕЩЁ НЕ ЗАКРЫТО", 1)[0]
    open_part = _top_section(content, "ЕЩЁ НЕ ЗАКРЫТО")
    positions: list[int] = []
    for example in current.examples:
        assert example in now, example
        positions.append(now.index(example))
    assert positions == sorted(positions)
    for example in hang.examples:
        assert example not in open_part


def test_приказ_не_совпадать_с_образцами_дословно(script_v4):
    """Рядом с образцами — прямой приказ не совпадать с ними дословно."""
    step = script_v4.step("city")
    assert step.examples
    messages = build_turn_messages(
        script=script_v4,
        steps=[step],
        profile={},
        facts={},
        history=[],
        asides_done=[],
    )
    content = messages[0].content
    assert "**Примеры**" in content
    assert SPEECH_RULES[0].startswith("ВНИМАНИЕ: примеры")
    assert "дословное или почти дословное совпадение с примером — ошибка" in content.lower()
    assert content.index("**Примеры**") < content.index(step.examples[0])


def test_жёсткий_запрет_фактов_вне_данных(script):
    """В полной сборке — правило про данные как есть; жёсткий запрет — в коротких."""
    from graph.prompts import _HARD_FACT_BAN, build_filler_messages

    messages = build_turn_messages(
        script=script,
        steps=[script.step("city")],
        profile={},
        facts={},
        history=[],
        asides_done=[],
    )
    content = messages[0].content
    assert "из переданных данных как есть" in content
    filler = build_filler_messages(script, messages=[], history_limit=2)
    assert _HARD_FACT_BAN in filler[0].content


def test_запрет_служебных_слов_и_обещания_сходить_за_данными(script):
    """Есть запрет служебных слов вслух и запрет обещать сходить за данными."""
    messages = build_turn_messages(
        script=script,
        steps=[script.step("name")],
        profile={},
        facts={},
        history=[],
        asides_done=[],
    )
    content = messages[0].content.lower()
    assert "служебные слова" in content
    assert "шаг" in content and "скрипт" in content and "перечень" in content
    assert "не обещать сходить за данными" in content


def test_persona_и_naturalness_суммарно_не_больше_7500():
    """Суммарный размер persona_block и naturalness_block ≤ 7500 символов."""
    total = len(persona_block()) + len(naturalness_block(ask_for_move=True))
    assert total <= 7500, total


def test_сборка_молчания_запрещает_решать_и_бронировать(script):
    """Сборка молчания: молчание не согласие; нельзя решать и бронировать."""
    from graph.prompts import build_silence_messages

    silence = build_silence_messages(
        script,
        messages=[AIMessage(content="Записать вас на ближайший набор?")],
        profile={},
        step=script.step("name"),
        attempt=1,
        history_limit=4,
    )
    content = silence[0].content.lower()
    assert "молчание не означает" in content
    assert "решать за человека" in content
    assert "бронировать" in content
    assert "возвращает человека в разговор" in content or "вернуть" in content


def test_silence_содержит_запрет_фактов_вне_данных(script):
    """Системное сообщение молчания содержит запрет называть факты вне данных."""
    from graph.prompts import _HARD_FACT_BAN, build_silence_messages

    silence = build_silence_messages(
        script,
        messages=[AIMessage(content="Стоимость пока уточняю.")],
        profile={},
        step=script.step("price"),
        attempt=1,
        history_limit=4,
    )
    assert _HARD_FACT_BAN in silence[0].content


def test_filler_содержит_запрет_фактов_вне_данных(script):
    """Системное сообщение filler содержит запрет называть факты вне данных."""
    from graph.prompts import _HARD_FACT_BAN, build_filler_messages

    filler = build_filler_messages(
        script,
        messages=[HumanMessage(content="сколько стоит?")],
        history_limit=2,
    )
    assert _HARD_FACT_BAN in filler[0].content


def test_waiting_содержит_запрет_фактов_вне_данных(script):
    """Системное сообщение waiting содержит запрет называть факты вне данных."""
    from graph.prompts import _HARD_FACT_BAN, build_waiting_messages

    waiting = build_waiting_messages(
        script,
        messages=[HumanMessage(content="сколько стоит?")],
        profile={},
        pending_fields=[],
        step=script.step("price"),
        history_limit=2,
    )
    assert _HARD_FACT_BAN in waiting[0].content


#: Маркеры указания при «не нашлось» — только в dynamic_status_block.
_MISSING_STATUS_MARKERS: tuple[str, ...] = (
    "не объявлять о пробеле",
    "вслух об этом не сообщать",
)


def test_правила_речи_и_короткие_сборки_без_указания_о_пробеле(script):
    """В SPEECH_RULES и коротких сборках указания «не объявлять о пробеле» нет."""
    from graph.prompts import (
        build_filler_messages,
        build_silence_messages,
        build_waiting_messages,
    )

    for rule in SPEECH_RULES:
        lowered = rule.lower()
        for marker in _MISSING_STATUS_MARKERS:
            assert marker not in lowered, marker

    silence = build_silence_messages(
        script,
        messages=[AIMessage(content="Стоимость пока уточняю.")],
        profile={},
        step=script.step("price"),
        attempt=1,
        history_limit=4,
    )[0].content.lower()
    filler = build_filler_messages(
        script,
        messages=[HumanMessage(content="сколько стоит?")],
        history_limit=2,
    )[0].content.lower()
    waiting = build_waiting_messages(
        script,
        messages=[HumanMessage(content="сколько стоит?")],
        profile={},
        pending_fields=[],
        step=script.step("price"),
        history_limit=2,
    )[0].content.lower()
    for content in (silence, filler, waiting):
        for marker in _MISSING_STATUS_MARKERS:
            assert marker not in content, marker


def test_жёсткий_запрет_фактов_во_всех_сборках(script):
    """Жёсткий запрет на факты вне данных — в коротких сборках; в полной — правило данных."""
    from graph.prompts import (
        _HARD_FACT_BAN,
        build_filler_messages,
        build_silence_messages,
        build_waiting_messages,
    )

    full = build_turn_messages(
        script=script,
        steps=[script.step("city")],
        profile={},
        facts={},
        history=[],
        asides_done=[],
    )[0].content
    assert "из переданных данных как есть" in full
    silence = build_silence_messages(
        script,
        messages=[AIMessage(content="?")],
        profile={},
        step=script.step("price"),
        attempt=1,
        history_limit=2,
    )[0].content
    filler = build_filler_messages(
        script,
        messages=[HumanMessage(content="?")],
        history_limit=2,
    )[0].content
    waiting = build_waiting_messages(
        script,
        messages=[HumanMessage(content="?")],
        profile={},
        pending_fields=[],
        step=script.step("price"),
        history_limit=2,
    )[0].content
    for content in (silence, filler, waiting):
        assert _HARD_FACT_BAN in content


#: Заголовки верхнего уровня полной сборки — порядок разделов.
_TOP_LEVEL_SECTIONS: tuple[str, ...] = (
    "# КАК РАБОТАТЬ",
    "# КУДА ВЕДЁМ РАЗГОВОР",
    "# ПРАВИЛА РЕЧИ",
    "# КОНТЕКСТ",
    "# СЕЙЧАС ГОВОРИМ ОБ ЭТОМ",
    "# ЕЩЁ НЕ ЗАКРЫТО",
    "# ЧТО ДАЛЬШЕ",
    "# ФОРМА ОТВЕТА",
)


def _top_section(content: str, title: str) -> str:
    """Тело раздела верхнего уровня ``# title`` до следующего такого заголовка."""
    marker = f"# {title}"
    assert marker in content, title
    after = content.split(marker, 1)[1]
    if after.startswith("\n"):
        after = after[1:]
    match = re.search(r"^# [^#\n]", after, flags=re.MULTILINE)
    return after[: match.start()] if match else after


def test_системное_разделы_верхнего_уровня_в_порядке(script_v4):
    """В системном сообщении все разделы верхнего уровня в порядке из файла."""
    current = script_v4.step("city")
    hang = script_v4.step("who_studies")
    nxt = script_v4.step("experience")
    messages = build_turn_messages(
        script=script_v4,
        steps=[current, hang],
        profile={"city": "Пермь"},
        facts={"city": {"name": "Пермь"}},
        history=[],
        asides_done=[],
        next_step=nxt,
        context_text="Город: Пермь",
        dynamic_status="не нашлось",
        closed_steps=[script_v4.step("greeting")],
    )
    content = messages[0].content
    positions = [content.index(title) for title in _TOP_LEVEL_SECTIONS]
    assert positions == sorted(positions)


def test_пустой_раздел_не_выводится(script_v4):
    """Раздел без содержимого не выводится: пустых заголовков нет."""
    messages = build_turn_messages(
        script=script_v4,
        steps=[script_v4.step("greeting")],
        profile={},
        facts={},
        history=[],
        asides_done=[],
    )
    content = messages[0].content
    assert "# ЧТО ДАЛЬШЕ" not in content
    assert "# ЕЩЁ НЕ ЗАКРЫТО" not in content
    assert "# КАК РАБОТАТЬ" in content
    assert "# ПРАВИЛА РЕЧИ" in content
    assert "# СЕЙЧАС ГОВОРИМ ОБ ЭТОМ" in content
    assert "# ФОРМА ОТВЕТА" in content
    for title in _TOP_LEVEL_SECTIONS:
        if title in content:
            # После заголовка есть непустая строка содержимого.
            after = content.split(title, 1)[1].lstrip("\n")
            first = after.split("\n", 1)[0].strip()
            assert first, f"пустой раздел: {title}"


def test_шаги_со_своими_заголовками_и_подразделами(script_v4):
    """Текущий шаг — с примерами; незакрытый — только название и требования."""
    current = script_v4.step("city")
    hang = script_v4.step("who_studies")
    messages = build_turn_messages(
        script=script_v4,
        steps=[current, hang],
        profile={},
        facts={},
        history=[],
        asides_done=[],
    )
    content = messages[0].content
    now = _top_section(content, "СЕЙЧАС ГОВОРИМ ОБ ЭТОМ")
    open_part = _top_section(content, "ЕЩЁ НЕ ЗАКРЫТО")
    assert f"## {current.name}" in now
    assert current.requirements in now
    assert "**Требования**" in now
    assert "**Примеры**" in now
    for example in current.examples:
        assert example in now
    assert f"## {hang.name}" in open_part
    assert hang.requirements in open_part
    assert "**Требования**" in open_part
    assert "**Примеры**" not in open_part
    for example in hang.examples:
        assert example not in open_part


def test_системное_без_пометок_ведущий_хорошо_плохо(script_v4):
    """Нет пометок «Ведущий шаг», «Висящий шаг», «Хорошо:», «Плохо:»."""
    messages = build_turn_messages(
        script=script_v4,
        steps=[script_v4.step("city"), script_v4.step("who_studies")],
        profile={},
        facts={},
        history=[],
        asides_done=[],
        next_step=script_v4.step("experience"),
    )
    content = messages[0].content
    assert "Ведущий шаг" not in content
    assert "Висящий шаг" not in content
    assert "Плохо:" not in content
    assert "Хорошо:" not in content
    for rule in SPEECH_RULES:
        assert "Плохо:" not in rule
        assert "Хорошо:" not in rule


def test_правила_нумерованный_список_пункт_про_примеры_первый(script):
    """Правила выведены нумерованным списком, пункт про примеры — первым."""
    assert SPEECH_RULES[0].startswith("ВНИМАНИЕ: примеры")
    messages = build_turn_messages(
        script=script,
        steps=[script.step("city")],
        profile={},
        facts={},
        history=[],
        asides_done=[],
    )
    content = messages[0].content
    rules_part = content.split("# ПРАВИЛА РЕЧИ", 1)[1].split("# ", 1)[0]
    assert rules_part.strip().startswith("0. ВНИМАНИЕ: примеры")
    for index in range(len(SPEECH_RULES)):
        assert f"\n{index}. " in f"\n{rules_part}" or rules_part.startswith(f"{index}. ")


def test_нет_формулировки_молчание_не_проблема(script):
    """Формулировки о том, что молчание собеседника не проблема, нет."""
    joined = _lead_speech_joined().lower()
    assert "молчание собеседника" not in joined
    messages = build_turn_messages(
        script=script,
        steps=[script.step("city")],
        profile={},
        facts={},
        history=[],
        asides_done=[],
    )
    assert "молчание собеседника после реплики не проблема" not in messages[0].content.lower()


def test_реплика_ведёт_дальше_три_вида_продолжения(script):
    """Есть требование, что реплика ведёт разговор дальше, и три вида продолжения."""
    joined = _lead_speech_joined().lower()
    assert "не заканчивается в никуда" in joined
    assert "вопрос по делу" in joined
    assert "переход к следующей теме" in joined
    assert "обозначение того, о чём пойдёт речь" in joined


def test_строить_реплику_по_уже_известному(script):
    """Есть указание строить реплику по уже известному, когда нового по теме нет."""
    joined = _lead_speech_joined().lower()
    assert "если по текущей теме нового не осталось" in joined
    assert "уже известно из разговора" in joined


def test_пункты_speech_rules_не_длиннее_500():
    """Ни один пункт SPEECH_RULES не длиннее 500 символов."""
    for index, rule in enumerate(SPEECH_RULES):
        assert len(rule) <= 500, f"пункт {index}: {len(rule)} символов"


def test_девять_правил_ведения_в_системном_сообщении(script):
    """Все девять правил разбивки присутствуют в системном сообщении отдельно."""
    messages = build_turn_messages(
        script=script,
        steps=[script.step("city")],
        profile={},
        facts={},
        history=[],
        asides_done=[],
    )
    content = messages[0].content
    assert "Разговор ведёт агент: сам решает, о чём говорить дальше" in content
    assert "Реплика не заканчивается в никуда" in content
    assert "Что пообещал рассказать — обязан рассказать" in content
    assert "Если по текущей теме нового не осталось" in content
    assert "Вопрос по делу двигает разговор дальше" in content
    assert "Вопросы вида «что бы Вы хотели узнать»" in content
    assert "Разрешения продолжать не спрашивают" in content
    assert "Пустые проверки «что скажете?»" in content
    assert "переходить к следующему, а не спрашивать разрешения" in content
    for rule in _lead_speech_rules():
        assert rule in content


def test_бессвязная_реплика_про_искажённые_названия(script):
    """Правило про бессвязную реплику упоминает искажённые названия."""
    rule = next(r for r in SPEECH_RULES if "бессвязна" in r.lower())
    assert "искажённ" in rule.lower()
    assert "не принимают за верные" in rule.lower()


def test_отвеченное_упоминает_сведения_из_формы(script):
    """Правило про отвеченное упоминает сведения из формы."""
    rule = next(r for r in SPEECH_RULES if "на отвеченное" in r.lower())
    lowered = rule.lower()
    assert "форме" in lowered
    assert "город" in lowered
    assert "имя" in lowered
    assert "коробк" in lowered


def test_адреса_филиалов_без_обещания_уточнить_и_вернуться(script):
    """Правило про адреса филиалов не содержит обещания уточнить и вернуться."""
    rule = next(r for r in SPEECH_RULES if "утверждать наличие филиала" in r.lower())
    lowered = rule.lower()
    assert "уточню" not in lowered
    assert "вернус" not in lowered
    assert "плохо:" not in lowered
    assert "хорошо:" not in lowered


def test_короткие_сборки_без_разметки_разделов(script):
    """Короткие сборки не изменились: без заголовков разделов полной сборки."""
    from graph.prompts import (
        build_filler_messages,
        build_silence_messages,
        build_waiting_messages,
    )

    filler = build_filler_messages(script, messages=[HumanMessage(content="?")], history_limit=2)[
        0
    ].content
    silence = build_silence_messages(
        script,
        messages=[AIMessage(content="?")],
        profile={},
        step=script.step("city"),
        attempt=1,
        history_limit=2,
    )[0].content
    waiting = build_waiting_messages(
        script,
        messages=[HumanMessage(content="?")],
        profile={},
        pending_fields=[],
        step=script.step("city"),
        history_limit=2,
    )[0].content
    for content in (filler, silence, waiting):
        assert "# КАК РАБОТАТЬ" not in content
        assert "# ПРАВИЛА РЕЧИ" not in content
        assert "# СЕЙЧАС ГОВОРИМ ОБ ЭТОМ" not in content
        assert "# ЕЩЁ НЕ ЗАКРЫТО" not in content
        assert "# ШАГИ В РАБОТЕ" not in content


def test_текущий_шаг_в_сейчас_с_требованиями_и_примерами(script_v4):
    """Первый шаг из steps — в «СЕЙЧАС ГОВОРИМ ОБ ЭТОМ» с требованиями и примерами."""
    current = script_v4.step("city")
    hang = script_v4.step("who_studies")
    assert current.examples
    messages = build_turn_messages(
        script=script_v4,
        steps=[current, hang],
        profile={},
        facts={},
        history=[],
        asides_done=[],
    )
    content = messages[0].content
    now = _top_section(content, "СЕЙЧАС ГОВОРИМ ОБ ЭТОМ")
    assert f"## {current.name}" in now
    assert "**Требования**" in now
    assert current.requirements in now
    assert "**Примеры**" in now
    for example in current.examples:
        assert example in now


def test_остальные_шаги_в_ещё_не_закрыто_без_примеров(script_v4):
    """Остальные шаги — в «ЕЩЁ НЕ ЗАКРЫТО» только с названием и требованиями."""
    current = script_v4.step("city")
    hang = script_v4.step("who_studies")
    assert hang.examples
    messages = build_turn_messages(
        script=script_v4,
        steps=[current, hang],
        profile={},
        facts={},
        history=[],
        asides_done=[],
    )
    content = messages[0].content
    assert "# ЕЩЁ НЕ ЗАКРЫТО" in content
    open_part = _top_section(content, "ЕЩЁ НЕ ЗАКРЫТО")
    assert f"## {hang.name}" in open_part
    assert "**Требования**" in open_part
    assert hang.requirements in open_part
    assert "**Примеры**" not in open_part
    for example in hang.examples:
        assert example not in open_part


def test_один_шаг_без_раздела_ещё_не_закрыто(script_v4):
    """При одном шаге в steps раздел «ЕЩЁ НЕ ЗАКРЫТО» не выводится."""
    messages = build_turn_messages(
        script=script_v4,
        steps=[script_v4.step("city")],
        profile={},
        facts={},
        history=[],
        asides_done=[],
    )
    content = messages[0].content
    assert "# СЕЙЧАС ГОВОРИМ ОБ ЭТОМ" in content
    assert "# ЕЩЁ НЕ ЗАКРЫТО" not in content


def test_что_дальше_из_next_step_после_разделов_шагов(script_v4):
    """«ЧТО ДАЛЬШЕ» строится из next_step и идёт после текущих/незакрытых."""
    current = script_v4.step("city")
    hang = script_v4.step("who_studies")
    nxt = script_v4.step("experience")
    messages = build_turn_messages(
        script=script_v4,
        steps=[current, hang],
        profile={},
        facts={},
        history=[],
        asides_done=[],
        next_step=nxt,
    )
    content = messages[0].content
    assert "# ЧТО ДАЛЬШЕ" in content
    assert f"## {nxt.name}" in content.split("# ЧТО ДАЛЬШЕ", 1)[1]
    assert content.index("# СЕЙЧАС ГОВОРИМ ОБ ЭТОМ") < content.index("# ЕЩЁ НЕ ЗАКРЫТО")
    assert content.index("# ЕЩЁ НЕ ЗАКРЫТО") < content.index("# ЧТО ДАЛЬШЕ")


def test_вступление_требует_отработать_шаг_а_не_произнести(script_v4):
    """Во вступлении — отработать шаг, а не произнести; способы через разговор."""
    messages = build_turn_messages(
        script=script_v4,
        steps=[script_v4.step("city"), script_v4.step("who_studies")],
        profile={},
        facts={},
        history=[],
        asides_done=[],
    )
    content = messages[0].content.lower()
    assert "реплика строится по шагу" not in content
    assert "отработать, а не произнести" in content
    assert "ответить на сказанное" in content
    assert "отработать сомнение" in content
    assert "наводящий вопрос" in content
    assert "переспросить" in content
    assert "к незакрытому из раздела «ещё не закрыто» возвращаются" in content


def test_системное_без_предпочтительного_порядка_и_пометок_ведущий(script_v4):
    """Нет предпочтительного порядка и пометок «Ведущий» / «Висящий»."""
    messages = build_turn_messages(
        script=script_v4,
        steps=[script_v4.step("city"), script_v4.step("who_studies")],
        profile={},
        facts={},
        history=[],
        asides_done=[],
        next_step=script_v4.step("experience"),
    )
    content = messages[0].content
    lowered = content.lower()
    assert "предпочтительн" not in lowered
    assert "необязательн" not in lowered
    assert "подсказка, куда двигаться" not in lowered
    assert "не обязательно брать первый" not in lowered
    assert "Ведущий шаг" not in content
    assert "Висящий шаг" not in content
    assert "# ШАГИ В РАБОТЕ" not in content


def test_сводка_в_системном_между_как_работать_и_правилами(script_v4):
    """Раздел «КУДА ВЕДЁМ РАЗГОВОР» есть, стоит после «КАК РАБОТАТЬ» и до «ПРАВИЛА РЕЧИ»."""
    messages = build_turn_messages(
        script=script_v4,
        steps=[script_v4.step("city")],
        profile={},
        facts={},
        history=[],
        asides_done=[],
    )
    content = messages[0].content
    assert "# КУДА ВЕДЁМ РАЗГОВОР" in content
    how_i = content.index("# КАК РАБОТАТЬ")
    dest_i = content.index("# КУДА ВЕДЁМ РАЗГОВОР")
    rules_i = content.index("# ПРАВИЛА РЕЧИ")
    assert how_i < dest_i < rules_i
    body = _top_section(content, "КУДА ВЕДЁМ РАЗГОВОР")
    assert script_v4.summary.strip() in body
    assert "федеральную сеть автошкол" in body


def test_пустая_сводка_раздел_не_выводится(script_v4):
    """Пустая сводка — раздела «КУДА ВЕДЁМ РАЗГОВОР» нет."""
    from dataclasses import replace

    empty = replace(script_v4, summary="")
    messages = build_turn_messages(
        script=empty,
        steps=[empty.step("city")],
        profile={},
        facts={},
        history=[],
        asides_done=[],
    )
    assert "# КУДА ВЕДЁМ РАЗГОВОР" not in messages[0].content


def test_правило_опоры_на_диалог_сразу_после_примеров():
    """Правило про опору на диалог стоит сразу после пункта про примеры."""
    assert SPEECH_RULES[0].startswith("ВНИМАНИЕ: примеры")
    dialog = SPEECH_RULES[1]
    lowered = dialog.lower()
    assert "к чему пришёл разговор" in lowered
    assert "шаги обязательны" in lowered
    assert "возражения" in lowered
    assert "наводящие вопросы" in lowered
    assert "переспрос" in lowered
    assert "способ отработать" in lowered
    assert SPEECH_RULES[2].startswith("К клиенту обращение только на «Вы»")


def test_как_работать_описывает_из_чего_строится_реплика(script_v4):
    """В «КАК РАБОТАТЬ» — реплика строится из разговора, сводки, известного и шагов."""
    messages = build_turn_messages(
        script=script_v4,
        steps=[script_v4.step("greeting")],
        profile={},
        facts={},
        history=[],
        asides_done=[],
    )
    how = _top_section(messages[0].content, "КАК РАБОТАТЬ").lower()
    assert "весь разговор целиком" in how
    assert "куда ведём разговор" in how
    assert "реплика строится из всего этого разом" in how
    assert "реплика строится по шагу" not in messages[0].content.lower()
