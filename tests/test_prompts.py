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
    _HARD_FACT_BAN,
    _MISSING_KNOWLEDGE_GUARD,
    _NO_MECHANICS,
    _STEPS_CONVERSATION_FIRST,
    _STEPS_HANGING_NOTE,
    LEAD_PULL_OVERRIDES,
    LEAD_REPEAT_INTRO,
    LEAD_REPEAT_OVERRIDES,
    PULL_TASK,
    RULE_MOVE_ON,
    RULE_NO_OPEN_QUESTIONS,
    RULE_NO_VERBATIM,
    RULE_QUESTION_MOVES,
    SPEECH_RULES,
    _context_has_fact,
    _describe_step,
    aside_block,
    build_turn_messages,
    context_block,
    continuation_block,
    dynamic_status_block,
    facts_block,
    fill_facts,
    naturalness_block,
    next_step_block,
    persona_block,
    profile_block,
    speech_rules_block,
    step_block,
    steps_block,
)

#: Число правил речи, включая пункт 0 про примеры и пункт про опору на диалог.
_SPEECH_RULES_COUNT = 28

#: Начала десяти правил про ведение разговора — подряд после «одна тема».
_LEAD_SPEECH_RULE_STARTS: tuple[str, ...] = (
    "Разговор ведёт агент:",
    "Реплика не заканчивается в никуда",
    "Реплика, которая ждёт ответа прямо сейчас",
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
    assert "прощаться первым" in content.lower()
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


def test_требование_последовательности_разговора_в_промпте_хода(script):
    """Требование «разговор важнее порядка шагов» есть в собранном промпте хода."""
    messages = build_turn_messages(
        script=script,
        steps=[script.step("city"), script.step("who_studies")],
        profile={},
        facts={},
        history=[],
        asides_done=[],
    )
    content = messages[0].content
    assert _STEPS_CONVERSATION_FIRST in content
    steps_at = content.index("# СЕЙЧАС ГОВОРИМ ОБ ЭТОМ")
    assert content.index(_STEPS_CONVERSATION_FIRST) > steps_at


def test_требование_последовательности_разговора_несёт_три_смысла():
    """Незаконченное, шаг как направление и продолжение разговора — все три в тексте."""
    assert "Начатое доводится до конца" in _STEPS_CONVERSATION_FIRST
    assert "через несколько реплик" in _STEPS_CONVERSATION_FIRST
    assert "не брать его только потому, что он выдан" in _STEPS_CONVERSATION_FIRST
    assert "продолжает разговор с того места, где он повис" in _STEPS_CONVERSATION_FIRST


def test_требование_не_разрешает_пропускать_шаги():
    """Оговорка про отложить-не-отменить на месте: молчать по сценарию нельзя."""
    assert "Отложить — не значит отменить" in _STEPS_CONVERSATION_FIRST
    assert "Молча пропускать шаги нельзя" in _STEPS_CONVERSATION_FIRST


def test_требование_запрещает_делать_сделанное_дважды():
    """Прощание по кругу закрыто прямым пунктом про уже сделанное."""
    assert "не сделано ли по нему уже всё" in _STEPS_CONVERSATION_FIRST
    assert "попрощался" in _STEPS_CONVERSATION_FIRST


def test_требование_остаётся_при_одном_шаге_в_шапке(script):
    """Один шаг в шапке — раздела незакрытых нет, требование всё равно есть."""
    block = steps_block([script.step("city")], {}, {})
    assert _STEPS_CONVERSATION_FIRST in block
    assert _STEPS_HANGING_NOTE not in block


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
    initiative_i = content.index("# ИНИЦИАТИВА")
    assert how_i < rules_i < ctx_i < steps_i < form_i < initiative_i
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


def test_continuation_block_факт_молчания_без_предписаний():
    """continuation_block: факт молчания для continuation/silence, пусто для client."""
    from graph.prompts import continuation_block

    silence_fact = "Реплики человека не было: он молчит, разговор продолжается."
    forbidden = (
        "Комендантская",
        "Просвещения",
        "Плохо:",
        "Хорошо:",
        "не переспрашивать",
        "Не здороваться",
    )

    assert continuation_block(turn_kind="client") == ""

    cont = continuation_block(turn_kind="continuation")
    assert cont == silence_fact
    for fragment in forbidden:
        assert fragment not in cont

    silence = continuation_block(turn_kind="silence")
    assert silence == cont
    for fragment in forbidden:
        assert fragment not in silence


def test_build_turn_messages_silence_кладёт_факт_молчания(script):
    """build_turn_messages: silence кладёт continuation_block, client — нет."""
    from graph.prompts import build_turn_messages, continuation_block

    silence_fact = continuation_block(turn_kind="silence")
    assert silence_fact

    with_silence = build_turn_messages(
        script=script,
        steps=[script.step("theory_format")],
        profile={},
        facts={},
        history=[],
        asides_done=[],
        turn_kind="silence",
    )
    with_client = build_turn_messages(
        script=script,
        steps=[script.step("theory_format")],
        profile={},
        facts={},
        history=[],
        asides_done=[],
        turn_kind="client",
    )
    assert silence_fact in with_silence[0].content
    assert silence_fact not in with_client[0].content


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
    """В полной сборке — правило про данные как есть и жёсткий запрет; в коротких — запрет."""
    from graph.prompts import build_filler_messages

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
    assert _HARD_FACT_BAN in content
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


def test_filler_содержит_запрет_фактов_вне_данных(script):
    """Системное сообщение filler содержит запрет называть факты вне данных."""
    from graph.prompts import _HARD_FACT_BAN, build_filler_messages

    filler = build_filler_messages(
        script,
        messages=[HumanMessage(content="сколько стоит?")],
        history_limit=2,
    )
    assert _HARD_FACT_BAN in filler[0].content


def test_filler_ограничение_длины_и_запреты_оценки_рассуждений(script):
    """Заглушка: лимит длины, без оценки, рассуждений, вопросов и фактов."""
    from graph.prompts import _HARD_FACT_BAN, build_filler_messages

    content = build_filler_messages(
        script,
        messages=[HumanMessage(content="Санкт-Петербург")],
        history_limit=2,
    )[0].content
    lowered = content.lower()
    assert "ограничение длины" in lowered
    assert "два-три слова" in lowered or "два–три слова" in lowered
    assert "не оценивать" in lowered
    assert "замечательный выбор" in lowered
    assert "звучит основательно" in lowered
    assert "не рассуждать" in lowered
    assert "чувствах" in lowered
    assert "не задавать вопросов" in lowered
    assert "не сообщать фактов" in lowered
    assert "не повторять" in lowered
    assert _HARD_FACT_BAN in content
    assert "не реплика" in lowered


def test_сборки_ожидания_и_основная_не_из_filler(script):
    """Ожидание и основная сборка — свои маркеры, не текст заглушки."""
    from graph.prompts import (
        build_filler_messages,
        build_waiting_messages,
    )

    filler = build_filler_messages(
        script,
        messages=[HumanMessage(content="?")],
        history_limit=2,
    )[0].content.lower()
    waiting = build_waiting_messages(
        script,
        messages=[HumanMessage(content="?")],
        profile={},
        pending_fields=[],
        step=script.step("price"),
        history_limit=2,
    )[0].content
    full = build_turn_messages(
        script=script,
        steps=[script.step("price")],
        profile={},
        facts={},
        history=[],
        asides_done=[],
    )[0].content

    assert "восьми слов" in waiting.lower() or "восемь слов" in waiting.lower()
    assert "реплика ожидания" in waiting.lower()
    assert "ограничение длины" not in waiting.lower()
    assert "не оценивать сказанное" not in waiting.lower()

    assert "# КАК РАБОТАТЬ" in full
    assert "# ПРАВИЛА РЕЧИ" in full
    assert "ограничение длины" not in full.lower()
    assert filler.count("ограничение длины") == 1


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
        build_waiting_messages,
    )

    for rule in SPEECH_RULES:
        lowered = rule.lower()
        for marker in _MISSING_STATUS_MARKERS:
            assert marker not in lowered, marker

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
    for content in (filler, waiting):
        for marker in _MISSING_STATUS_MARKERS:
            assert marker not in content, marker


def test_жёсткий_запрет_фактов_во_всех_сборках(script):
    """Жёсткий запрет на факты вне данных — в полной сборке и в коротких."""
    from graph.prompts import (
        build_filler_messages,
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
    assert _HARD_FACT_BAN in full
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
    for content in (filler, waiting, full):
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
    "# ИНИЦИАТИВА",
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
        build_waiting_messages,
    )

    filler = build_filler_messages(script, messages=[HumanMessage(content="?")], history_limit=2)[
        0
    ].content
    waiting = build_waiting_messages(
        script,
        messages=[HumanMessage(content="?")],
        profile={},
        pending_fields=[],
        step=script.step("city"),
        history_limit=2,
    )[0].content
    for content in (filler, waiting):
        assert "# КАК РАБОТАТЬ" not in content
        assert "# ПРАВИЛА РЕЧИ" not in content
        assert "# СЕЙЧАС ГОВОРИМ ОБ ЭТОМ" not in content
        assert "# ЕЩЁ НЕ ЗАКРЫТО" not in content
        assert "# ШАГИ В РАБОТЕ" not in content
        assert "# ИНИЦИАТИВА" not in content


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


def test_speech_rules_block_без_флага_и_normal_одинаковы():
    """Без аргумента и с ``mode="normal"`` — один и тот же штатный текст."""
    assert speech_rules_block() == speech_rules_block(mode="normal")


def test_speech_rules_block_repeat_подменяет_четыре_правила():
    """При ``mode="repeat"`` число правил то же, четыре текста — другие."""
    staff = speech_rules_block(mode="normal")
    repeated = speech_rules_block(mode="repeat")
    staff_lines = staff.splitlines()
    repeated_lines = repeated.splitlines()
    assert len(staff_lines) == len(repeated_lines) == len(SPEECH_RULES) + 1
    changed = sum(
        1 for left, right in zip(staff_lines, repeated_lines, strict=True) if left != right
    )
    assert changed == 4
    for staff_rule, override in LEAD_REPEAT_OVERRIDES.items():
        assert staff_rule in staff
        assert override in repeated
        assert staff_rule != override


def test_speech_rules_block_repeat_без_штатных_подменённых():
    """При ``mode="repeat"`` штатные тексты четырёх правил в блоке не встречаются."""
    text = speech_rules_block(mode="repeat")
    for rule in (
        RULE_MOVE_ON,
        RULE_QUESTION_MOVES,
        RULE_NO_OPEN_QUESTIONS,
        RULE_NO_VERBATIM,
    ):
        assert rule not in text
        assert LEAD_REPEAT_OVERRIDES[rule] in text


def test_steps_block_repeat_врезка_и_примеры(script_v4):
    """Врезка до «Требования» только при repeat; примеры есть в обоих режимах."""
    step = script_v4.step("city")
    assert step.examples
    with_flag = steps_block([step], {}, {}, mode="repeat")
    without = steps_block([step], {}, {}, mode="normal")
    assert LEAD_REPEAT_INTRO in with_flag
    assert with_flag.index(LEAD_REPEAT_INTRO) < with_flag.index("Требования")
    assert LEAD_REPEAT_INTRO not in without
    for example in step.examples:
        assert example in with_flag
        assert example in without


def test_speech_rules_block_pull_подменяет_четыре_правила():
    """При ``mode="pull"`` подменённые правила отличаются от штатных, число то же."""
    staff = speech_rules_block(mode="normal")
    pulled = speech_rules_block(mode="pull")
    assert len(staff.splitlines()) == len(pulled.splitlines()) == len(SPEECH_RULES) + 1
    assert staff != pulled
    changed = sum(
        1
        for left, right in zip(staff.splitlines(), pulled.splitlines(), strict=True)
        if left != right
    )
    assert changed == 4


def test_steps_block_pull_задача_перед_шагом(script_v4):
    """При ``mode="pull"`` — задача вытаскивания перед штатным разделом шага."""
    from graph.prompts import PULL_TASK

    step = script_v4.step("city")
    assert step.examples
    text = steps_block([step], {}, {}, mode="pull")
    assert PULL_TASK in text
    assert "ЗАДАЧА ЭТОГО ХОДА" in text
    assert "СЕЙЧАС ГОВОРИМ ОБ ЭТОМ" in text
    assert "Требования" in text
    assert "Примеры" in text
    assert text.index("ЗАДАЧА ЭТОГО ХОДА") < text.index("СЕЙЧАС ГОВОРИМ ОБ ЭТОМ")
    for example in step.examples:
        assert example in text


def test_steps_block_repeat_сохраняет_текущий_раздел(script_v4):
    """При ``mode="repeat"`` раздел «СЕЙЧАС ГОВОРИМ ОБ ЭТОМ» и врезка на месте."""
    from graph.prompts import PULL_TASK

    step = script_v4.step("city")
    text = steps_block([step], {}, {}, mode="repeat")
    assert "СЕЙЧАС ГОВОРИМ ОБ ЭТОМ" in text
    assert "Требования" in text
    assert "Примеры" in text
    assert LEAD_REPEAT_INTRO in text
    assert PULL_TASK not in text


def test_steps_block_normal_без_врезок(script_v4):
    """При ``mode="normal"`` нет ни ``PULL_TASK``, ни ``LEAD_REPEAT_INTRO``."""
    from graph.prompts import PULL_TASK

    step = script_v4.step("city")
    text = steps_block([step], {}, {}, mode="normal")
    assert PULL_TASK not in text
    assert LEAD_REPEAT_INTRO not in text
    assert "СЕЙЧАС ГОВОРИМ ОБ ЭТОМ" in text
    assert "Требования" in text


def test_правило_про_знак_вопроса_есть_в_правилах_речи():
    """В штатном блоке правил речи есть правило про знак вопроса."""
    block = speech_rules_block()
    assert "ждёт ответа прямо сейчас" in block
    assert "заканчивается вопросом со знаком вопроса" in block


def test_правило_про_знак_вопроса_есть_в_режимах_повтора_и_вытаскивания():
    """Правило про знак вопроса не теряется при подмене правил в repeat/pull."""
    for mode in ("repeat", "pull"):
        block = speech_rules_block(mode=mode)
        assert "ждёт ответа прямо сейчас" in block
        assert "заканчивается вопросом со знаком вопроса" in block


def test_правило_называет_обратный_случай():
    """Правило явно запрещает знак вопроса там, где отвечать нечего."""
    block = speech_rules_block()
    assert "знаком вопроса не заканчиваются" in block
    assert "повиснет в тишине" in block


def test_правило_называет_просьбу_на_потом():
    """Просьба о будущем действии ответа сейчас не требует — знак вопроса вреден."""
    assert "просьба написать или дать знать потом" in speech_rules_block()


def test_естественность_требует_знак_вопроса_когда_ответ_нужен():
    """В блоке «Естественность» при обычном ask_for_move — требование знака вопроса."""
    block = naturalness_block(ask_for_move=True, pending_only=False)
    assert "Ответ нужен сейчас — это вопрос со знаком вопроса" in block


def test_естественность_pending_only_не_изменилась():
    """Ветка pending_only не получила новую формулировку про знак вопроса."""
    block = naturalness_block(ask_for_move=True, pending_only=True)
    assert "по уже висящему вопросу" in block
    assert "Ответ нужен сейчас" not in block


def test_число_правил_речи_выросло_на_одно():
    """После добавления правила про знак вопроса правил стало на одно больше."""
    assert len(SPEECH_RULES) == _SPEECH_RULES_COUNT


def test_правило_запрещает_прощаться_первым():
    """В правилах речи — запрет прощаться первым, включая «до встречи»."""
    block = speech_rules_block()
    assert "Прощаться первым" in block
    assert "до встречи" in block
    assert "хорошего дня" in block
    assert "буду ждать Вас" in block


def test_правило_разрешает_попрощаться_в_ответ():
    """Попрощаться можно только после того, как простился собеседник."""
    assert "после того, как простился сам собеседник" in speech_rules_block()


def test_подтверждение_встречи_не_прощание():
    """Подтверждение договорённости о встрече прощанием не является."""
    assert "Подтверждение договорённости о встрече прощанием не является" in speech_rules_block()


def test_правило_про_прощание_во_всех_режимах():
    """Запрет прощаться первым действует и в repeat, и в pull."""
    for mode in ("repeat", "pull"):
        block = speech_rules_block(mode=mode)
        assert "Прощаться первым" in block
        assert "до встречи" in block
        assert "хорошего дня" in block
        assert "буду ждать Вас" in block
        assert "после того, как простился сам собеседник" in block
        assert "Подтверждение договорённости о встрече прощанием не является" in block


#: Прежний текст ``dynamic_status_block`` при ``DYN_MISSING`` — без изменений.
_MISSING_STATUS_TEXT = (
    "По нужному факту в данных ничего нет. Вслух об этом не сообщать, "
    "не извиняться и не объявлять о пробеле. Вести разговор дальше по "
    "тому, что известно — не выдумывать. Если по смыслу уместно — "
    "предложить прислать подробности в переписку, но только там, где "
    "разговор до мессенджера дошёл, а не посреди другой темы."
)

#: Редакция правила 17 для ``repeat`` — не менялась.
_REPEAT_MOVE_ON = (
    "По текущей теме человек ещё не ответил, и ответ сейчас нужен. Уходить "
    "на следующую тему рано: реплика обязана вести к слову собеседника. "
    "Исключение — реплика ожидания, пока данные готовятся: ход к человеку не нужен."
)


def test_speech_rules_block_жёсткий_запрет_во_всех_режимах():
    """В каждом режиме последний пункт — жёсткий запрет; нумерация и порядок на месте."""
    last_index = len(SPEECH_RULES)
    for mode in ("normal", "repeat", "pull"):
        block = speech_rules_block(mode=mode)
        lines = block.splitlines()
        assert _HARD_FACT_BAN in block
        assert lines[-1] == f"{last_index}. {_HARD_FACT_BAN}"
        assert len(lines) == last_index + 1
        for index, line in enumerate(lines):
            assert line.startswith(f"{index}. ")
        if mode == "normal":
            for index, rule in enumerate(SPEECH_RULES):
                assert lines[index] == f"{index}. {rule}"


def test_next_step_block_без_context_text_прежнее_поведение(script_v4):
    """Без ``context_text`` для SalesStep — ровно название и требования."""
    step = script_v4.step("terms")
    assert step.knowledge
    expected = f"## {step.name}\n\n**Требования**\n{step.requirements}"
    assert next_step_block(step, {}, {}) == expected


def test_next_step_block_нехватка_данных_по_контексту(script_v4):
    """С контекстом без данных — раздел о нехватке; с данными — без него."""
    step = script_v4.step("terms")
    assert step.knowledge
    missing = next_step_block(step, {}, {}, context_text="Город: Пермь")
    assert "**Не хватает данных**" in missing
    assert "В контексте нет данных:" in missing
    assert step.name in missing
    present_ctx = "срок обучения по городу: 2 месяца; время до первого занятия по вождению: 3 дня"
    present = next_step_block(step, {}, {}, context_text=present_ctx)
    assert "**Не хватает данных**" not in present
    assert present == f"## {step.name}\n\n**Требования**\n{step.requirements}"


def test_steps_block_висящий_шаг_нехватка_данных(script_v4):
    """У висящего шага раздел о нехватке появляется, если данных в контексте нет."""
    current = script_v4.step("city")
    hang = script_v4.step("terms")
    assert hang.knowledge
    block = steps_block([current, hang], {}, {}, context_text="Город: Пермь")
    hang_part = block.split("# ЕЩЁ НЕ ЗАКРЫТО", 1)[1]
    assert "**Не хватает данных**" in hang_part
    assert "В контексте нет данных:" in hang_part
    filled = "срок обучения по городу: 2 месяца; время до первого занятия по вождению: 3 дня"
    ok = steps_block([current, hang], {}, {}, context_text=filled)
    hang_ok = ok.split("# ЕЩЁ НЕ ЗАКРЫТО", 1)[1]
    assert "**Не хватает данных**" not in hang_ok


def test_dynamic_status_block_поиск_и_не_нашлось():
    """``DYN_SEARCHING`` — непустой текст; ``DYN_MISSING`` — прежний текст."""
    from graph.context import DYN_MISSING, DYN_SEARCHING

    searching = dynamic_status_block(status=DYN_SEARCHING)
    assert searching
    lowered = searching.lower()
    assert "готовятся" in lowered
    assert "цифры" in lowered
    assert "уточняем" in lowered
    assert dynamic_status_block(status=DYN_MISSING) == _MISSING_STATUS_TEXT


def test_continuation_block_pull_та_же_отметка_что_у_молчания():
    """На ``pull`` — та же отметка молчания, что на continuation и silence."""
    silence_fact = "Реплики человека не было: он молчит, разговор продолжается."
    assert continuation_block(turn_kind="pull") == silence_fact
    assert continuation_block(turn_kind="continuation") == silence_fact
    assert continuation_block(turn_kind="silence") == silence_fact
    assert continuation_block(turn_kind="client") == ""


def test_pull_task_без_обещания_и_запрета_новой_темы():
    """``PULL_TASK`` не держит модель на уже сказанном и не запрещает новую тему."""
    lowered = PULL_TASK.lower()
    assert "расскажи, что обещал" not in lowered
    assert "если ты обещал" not in lowered
    assert "не начинай новую тему" not in lowered
    assert "если не обещал" not in lowered


def test_правило_17_pull_отличается_от_repeat():
    """Правило 17 в ``pull`` своё; редакция ``repeat`` не изменилась."""
    move_index = SPEECH_RULES.index(RULE_MOVE_ON)
    assert move_index == 17
    assert LEAD_REPEAT_OVERRIDES[RULE_MOVE_ON] == _REPEAT_MOVE_ON
    repeat_lines = speech_rules_block(mode="repeat").splitlines()
    pull_lines = speech_rules_block(mode="pull").splitlines()
    assert repeat_lines[move_index] == f"{move_index}. {_REPEAT_MOVE_ON}"
    assert pull_lines[move_index] != repeat_lines[move_index]
    assert "Уходить на следующую тему рано" in repeat_lines[move_index]
    assert "Уходить на следующую тему рано" not in pull_lines[move_index]
    assert LEAD_PULL_OVERRIDES[RULE_MOVE_ON] in pull_lines[move_index]
    assert "не топтаться" in pull_lines[move_index]


def test_сборка_pull_молчание_запрет_нехватка(script_v4):
    """Pull, два шага, пустой контекст: молчание, запрет фактов, нехватка у обоих."""
    current = script_v4.step("terms")
    hang = script_v4.step("theory_format")
    assert current.knowledge and hang.knowledge
    silence_fact = continuation_block(turn_kind="pull")
    messages = build_turn_messages(
        script=script_v4,
        steps=[current, hang],
        profile={},
        facts={},
        history=[],
        asides_done=[],
        context_text="",
        mode="pull",
        turn_kind="pull",
    )
    content = messages[0].content
    assert silence_fact in content
    assert _HARD_FACT_BAN in content
    now = _top_section(content, "СЕЙЧАС ГОВОРИМ ОБ ЭТОМ")
    hang_part = _top_section(content, "ЕЩЁ НЕ ЗАКРЫТО")
    assert "**Не хватает данных**" in now
    assert "**Не хватает данных**" in hang_part


#: Правило про разрешение продолжать — ищется по началу, номер не важен.
def _rule_no_permission() -> str:
    """Возвращает пункт правил речи про разрешение продолжать."""
    return next(rule for rule in SPEECH_RULES if rule.startswith("Разрешения продолжать"))


#: Редакции четырёх подменяемых правил для повтора — меняться не должны.
_REPEAT_EDITION: dict[str, str] = {
    RULE_MOVE_ON: _REPEAT_MOVE_ON,
    RULE_QUESTION_MOVES: (
        "Вопрос по делу двигает разговор дальше: это выбор из двух вариантов, "
        "уточнение недостающего или предложение следующего шага. Сейчас вопрос "
        "обязателен, и берётся он из всего разговора — из того, что человек уже "
        "сказал и что осталось непрояснённым, а не из требований текущей темы."
    ),
    RULE_NO_OPEN_QUESTIONS: (
        "Вопросы вида «что бы Вы хотели узнать», «что Вас интересует», «о чём "
        "рассказать», «что осталось непонятным» запрещены: это перекладывание "
        "разговора на собеседника. Спросить о том, что уже прозвучало в разговоре, "
        "и предложить конкретный следующий шаг — можно и нужно."
    ),
    RULE_NO_VERBATIM: (
        "Эту тему в разговоре уже поднимали. Повторять её той же стороной — ошибка, "
        "и пересказ прежней мысли другими словами — тоже ошибка. Сказать надо то, "
        "чего человек про неё ещё не слышал."
    ),
}


def test_вопрос_остались_ли_вопросы_запрещён_без_оговорок():
    """В правиле про вопросы к собеседнику нет разрешающей оговорки «изредка»."""
    assert "изредка" not in RULE_NO_OPEN_QUESTIONS
    assert "остались ли вопросы, тоже нельзя" in RULE_NO_OPEN_QUESTIONS
    assert "ни в любой другой момент" in RULE_NO_OPEN_QUESTIONS
    # Смысл правила — запрет перекладывать ведение разговора — остался.
    assert "перекладывание разговора на собеседника" in RULE_NO_OPEN_QUESTIONS
    assert "что осталось непонятным" in RULE_NO_OPEN_QUESTIONS
    # Приглашение спрашивать — тот же приём, только без вопросительного знака.
    assert "«спрашивайте, я отвечу»" in RULE_NO_OPEN_QUESTIONS
    assert "«если что-то интересно — расскажу»" in RULE_NO_OPEN_QUESTIONS
    assert len(RULE_NO_OPEN_QUESTIONS) <= 500
    for mode in ("normal", "repeat", "pull"):
        assert "изредка" not in speech_rules_block(mode=mode)
    for rule in SPEECH_RULES:
        assert "изредка" not in rule.lower()


def test_разрешение_продолжать_запрещено_классом_а_не_перечнем():
    """Правило запрещает класс вопросов, а прежние обороты остались примерами."""
    rule = _rule_no_permission()
    assert "Запрещён весь класс" in rule
    assert "любой вопрос или приглашение спросить" in rule
    assert "просит согласия говорить дальше" in rule
    assert "отдаёт выбор темы собеседнику" in rule
    assert "какими бы словами это ни было сказано" in rule
    assert "Примеры, не весь перечень:" in rule
    for phrase in (
        "«рассказать подробнее?»",
        "«могу назвать?»",
        "«продолжать?»",
        "«продолжим?»",
        "«интересно узнать подробнее?»",
    ):
        assert phrase in rule
    assert rule.index("Запрещён весь класс") < rule.index("Примеры, не весь перечень:")
    assert "не запрет на вопросы вообще" in rule
    assert len(rule) <= 500


def test_нумерация_правил_сплошная_во_всех_режимах():
    """Номера правил идут подряд без дублей и разрывов; число правил прежнее."""
    assert len(SPEECH_RULES) == _SPEECH_RULES_COUNT
    for mode in ("normal", "repeat", "pull"):
        lines = speech_rules_block(mode=mode).splitlines()
        numbers = [int(line.split(".", 1)[0]) for line in lines]
        assert numbers == list(range(len(SPEECH_RULES) + 1))
        assert len(numbers) == len(set(numbers))
        assert lines[-1] == f"{len(SPEECH_RULES)}. {_HARD_FACT_BAN}"


def test_жёсткий_запрет_фактов_в_трёх_режимах_правил():
    """Жёсткий запрет на выдуманные факты остаётся во всех трёх режимах."""
    for mode in ("normal", "repeat", "pull"):
        assert _HARD_FACT_BAN in speech_rules_block(mode=mode)


def test_переопределения_подменяют_те_же_пункты_редакция_повтора_прежняя():
    """Repeat и pull подменяют одни и те же четыре пункта; повтор не изменился."""
    keys = {RULE_MOVE_ON, RULE_QUESTION_MOVES, RULE_NO_OPEN_QUESTIONS, RULE_NO_VERBATIM}
    assert set(LEAD_REPEAT_OVERRIDES) == keys
    assert set(LEAD_PULL_OVERRIDES) == keys
    assert LEAD_REPEAT_OVERRIDES == _REPEAT_EDITION
    for rule in keys - {RULE_MOVE_ON}:
        assert LEAD_PULL_OVERRIDES[rule] == LEAD_REPEAT_OVERRIDES[rule]
    assert LEAD_PULL_OVERRIDES[RULE_MOVE_ON] != LEAD_REPEAT_OVERRIDES[RULE_MOVE_ON]


def test_указание_вытаскивания_без_ссылки_на_отсутствующий_раздел():
    """Ни задача хода, ни правило перехода не ссылаются на раздел по названию."""
    assert "ЕЩЁ НЕ ЗАКРЫТО" not in PULL_TASK
    assert "ЕЩЁ НЕ ЗАКРЫТО" not in LEAD_PULL_OVERRIDES[RULE_MOVE_ON]
    assert "Незакрытых тем ниже нет" in PULL_TASK
    assert "уже известно из разговора" in PULL_TASK
    assert "Закончи вопросом, который требует выбора или решения" in PULL_TASK
    assert "Незакрытых тем ниже нет" in LEAD_PULL_OVERRIDES[RULE_MOVE_ON]
    assert "не топтаться на месте" in LEAD_PULL_OVERRIDES[RULE_MOVE_ON]


def test_сборка_вытаскивания_с_одним_шагом_исполнима(script_v4):
    """Один шаг в шапке: раздела нет и ссылок на него в промпте тоже нет."""
    one = build_turn_messages(
        script=script_v4,
        steps=[script_v4.step("terms")],
        profile={},
        facts={},
        history=[],
        asides_done=[],
        mode="pull",
        turn_kind="pull",
    )
    content = one[0].content
    assert PULL_TASK in content
    assert "ЕЩЁ НЕ ЗАКРЫТО" not in content
    assert _STEPS_HANGING_NOTE not in content
    assert "Незакрытых тем ниже нет" in content

    two = build_turn_messages(
        script=script_v4,
        steps=[script_v4.step("terms"), script_v4.step("group")],
        profile={},
        facts={},
        history=[],
        asides_done=[],
        mode="pull",
        turn_kind="pull",
    )
    with_hang = two[0].content
    assert "# ЕЩЁ НЕ ЗАКРЫТО" in with_hang
    assert _STEPS_HANGING_NOTE in with_hang


def test_обычный_ход_с_одним_шагом_без_ссылки_на_незакрытое(script_v4):
    """Хвост вступления про незакрытое выводится только вместе с разделом."""
    one = build_turn_messages(
        script=script_v4,
        steps=[script_v4.step("city")],
        profile={},
        facts={},
        history=[],
        asides_done=[],
    )
    assert "ЕЩЁ НЕ ЗАКРЫТО" not in one[0].content
    two = build_turn_messages(
        script=script_v4,
        steps=[script_v4.step("city"), script_v4.step("terms")],
        profile={},
        facts={},
        history=[],
        asides_done=[],
    )
    content = two[0].content
    assert _STEPS_HANGING_NOTE in content
    assert "# ЕЩЁ НЕ ЗАКРЫТО" in content


def test_четыре_шага_пустой_контекст_без_четырёх_одинаковых_блоков(script_v4):
    """Нехватка данных отмечена у каждого шага, а указание печатается один раз."""
    ids = ("terms", "theory_format", "included", "group")
    steps = [script_v4.step(step_id) for step_id in ids]
    for step in steps:
        assert step.knowledge
    messages = build_turn_messages(
        script=script_v4,
        steps=steps,
        profile={},
        facts={},
        history=[],
        asides_done=[],
        context_text="",
    )
    content = messages[0].content
    assert content.count("**Не хватает данных**") == len(steps)
    assert content.count(_MISSING_KNOWLEDGE_GUARD) == 1
    blocks = [part.splitlines()[1] for part in content.split("**Не хватает данных**")[1:]]
    assert len(set(blocks)) == len(steps)
    for step in steps:
        for fact in step.knowledge:
            assert fact in content
    now = _top_section(content, "СЕЙЧАС ГОВОРИМ ОБ ЭТОМ")
    assert _MISSING_KNOWLEDGE_GUARD in now


def test_указание_о_нехватке_достаётся_первому_шагу_с_пробелом(script_v4):
    """Ведущему шагу данных хватает — указание уходит первому висящему."""
    block = steps_block(
        [script_v4.step("city"), script_v4.step("terms"), script_v4.step("group")],
        {},
        {},
        context_text="Город: Пермь",
    )
    assert block.count(_MISSING_KNOWLEDGE_GUARD) == 1
    now, hang = block.split("# ЕЩЁ НЕ ЗАКРЫТО", 1)
    assert "**Не хватает данных**" not in now
    assert hang.count("**Не хватает данных**") == 2
    terms_part, group_part = hang.split(f"## {script_v4.step('group').name}", 1)
    assert _MISSING_KNOWLEDGE_GUARD in terms_part
    assert _MISSING_KNOWLEDGE_GUARD not in group_part
    assert "В контексте нет данных: расписание ближайших стартов по филиалу" in group_part


def test_проверка_факта_требует_все_значимые_слова():
    """Совпадения двух случайных слов из разных тем для факта не хватает."""
    price_ctx = "Цена: Стоимость — от 43900 рублей. Срок действия цены — до конца месяца."
    assert not _context_has_fact(price_ctx, {}, "стоимость и срок второй категории")
    assert not _context_has_fact(price_ctx, {}, "категории обучения кроме легковой")
    assert _context_has_fact("срок обучения по городу: 2 месяца", {}, "срок обучения по городу")
    # Падежи сравнению не мешают: слова сопоставляются по основам.
    assert _context_has_fact("Формат теории в городах: очно", {}, "форматы теории в городе")
    assert _context_has_fact("", {"tariffs": "линейка тарифов города"}, "линейка тарифов города")
    assert not _context_has_fact(
        "",
        {"price": "стоимость 43900, срок 2 месяца"},
        "стоимость и срок второй категории",
    )
    assert _context_has_fact("что угодно", {}, "   ")


def test_естественность_без_образца_с_разрешением_рассказать():
    """В блоке естественности нет образца, спрашивающего разрешения рассказать."""
    block = naturalness_block(ask_for_move=True)
    assert "Рассказать, что входит?" not in block
    assert "Обучение под ключ — всё включено, доплат нет" in block


def test_полная_сборка_содержит_оба_изменённых_правила(script_v4):
    """Оба переписанных правила попадают в системное сообщение целиком."""
    messages = build_turn_messages(
        script=script_v4,
        steps=[script_v4.step("terms")],
        profile={},
        facts={},
        history=[],
        asides_done=[],
    )
    content = messages[0].content
    assert RULE_NO_OPEN_QUESTIONS in content
    assert _rule_no_permission() in content


#: Правила 14, 15 и 17 — дословно, без изменений.
_RULE_14_VERBATIM = RULE_NO_OPEN_QUESTIONS
_RULE_15_VERBATIM = (
    "Разрешения продолжать не спрашивают: есть что рассказать — рассказывают. "
    "Запрещён весь класс: любой вопрос или приглашение спросить, которым агент "
    "просит согласия говорить дальше или отдаёт выбор темы собеседнику, какими "
    "бы словами это ни было сказано. Примеры, не весь перечень: «рассказать "
    "подробнее?», «интересно узнать подробнее?», «могу назвать?», "
    "«продолжать?», «продолжим?». Это не запрет на вопросы вообще, а запрет "
    "перекладывать на собеседника решение, говорить ли дальше."
)
_RULE_17_VERBATIM = RULE_MOVE_ON

#: Примеры запрещённых оборотов из боевых звонков — в разделе инициативы.
_INITIATIVE_CALL_EXAMPLES: tuple[str, ...] = (
    "«Если хотите, расскажу подробнее?»",
    "«Готова рассказать, если интересно?»",
)


def test_правила_14_15_17_не_изменились():
    """Правила 14, 15 и 17 в SPEECH_RULES остались дословно прежними."""
    assert SPEECH_RULES[14] == _RULE_14_VERBATIM
    assert SPEECH_RULES[15] == _RULE_15_VERBATIM
    assert SPEECH_RULES[17] == _RULE_17_VERBATIM


def test_инициатива_последний_раздел_полной_сборки(script):
    """Раздел «ИНИЦИАТИВА» есть в полной сборке и стоит после «ФОРМА ОТВЕТА»."""
    content = build_turn_messages(
        script=script,
        steps=[script.step("city")],
        profile={},
        facts={},
        history=[],
        asides_done=[],
    )[0].content
    form_i = content.index("# ФОРМА ОТВЕТА")
    initiative_i = content.index("# ИНИЦИАТИВА")
    assert form_i < initiative_i
    assert content[initiative_i:].startswith("# ИНИЦИАТИВА")
    after_initiative = content[initiative_i + len("# ИНИЦИАТИВА") :]
    assert not re.search(r"^# [^#\n]", after_initiative, flags=re.MULTILINE)


def test_инициатива_короткий_раздел(script):
    """Раздел «ИНИЦИАТИВА» — несколько строк, не длинный перечень."""
    content = build_turn_messages(
        script=script,
        steps=[script.step("city")],
        profile={},
        facts={},
        history=[],
        asides_done=[],
    )[0].content
    body = _top_section(content, "ИНИЦИАТИВА").strip()
    lines = [line for line in body.splitlines() if line.strip()]
    assert 1 <= len(lines) <= 8


def test_инициатива_требования_и_запрет_разрешений(script):
    """В разделе инициативы — вопрос по делу и запрет спрашивать разрешения."""
    body = _top_section(
        build_turn_messages(
            script=script,
            steps=[script.step("city")],
            profile={},
            facts={},
            history=[],
            asides_done=[],
        )[0].content,
        "ИНИЦИАТИВА",
    ).lower()
    assert "вопросом, который двигает дело" in body
    assert "выбор" in body
    assert "решение" in body
    assert "недостающие данные" in body
    assert "спрашивать разрешения рассказать" in body
    assert "работа по сценарию" in body


def test_инициатива_примеры_из_звонков(script):
    """В разделе инициативы есть примеры запрещённых оборотов из боевых звонков."""
    body = _top_section(
        build_turn_messages(
            script=script,
            steps=[script.step("city")],
            profile={},
            facts={},
            history=[],
            asides_done=[],
        )[0].content,
        "ИНИЦИАТИВА",
    )
    assert "Примеры, не весь перечень:" in body
    for example in _INITIATIVE_CALL_EXAMPLES:
        assert example in body


def test_короткие_сборки_без_раздела_инициативы(script):
    """Вытаскивание, заглушка и реплика ожидания не содержат раздел «ИНИЦИАТИВА»."""
    from graph.prompts import (
        build_filler_messages,
        build_pull_messages,
        build_waiting_messages,
    )

    filler = build_filler_messages(script, messages=[HumanMessage(content="?")], history_limit=2)[
        0
    ].content
    waiting = build_waiting_messages(
        script,
        messages=[HumanMessage(content="?")],
        profile={},
        pending_fields=[],
        step=script.step("city"),
        history_limit=2,
    )[0].content
    pull = build_pull_messages(
        script,
        messages=[HumanMessage(content="?")],
        profile={},
        step=script.step("city"),
    )[0].content
    for content in (filler, waiting, pull):
        assert "# ИНИЦИАТИВА" not in content
        for example in _INITIATIVE_CALL_EXAMPLES:
            assert example not in content


def test_инициатива_признак_негодного_вопроса_и_прежние_примеры(script):
    """В разделе инициативы есть признак негодного вопроса; примеры про «расскажу подробнее» на месте."""
    body = _top_section(
        build_turn_messages(
            script=script,
            steps=[script.step("city")],
            profile={},
            facts={},
            history=[],
            asides_done=[],
        )[0].content,
        "ИНИЦИАТИВА",
    )
    assert "про сам разговор, а не про дело" in body
    assert "Интересно узнать, как проходит вождение?" in body
    assert "Хотите узнать, когда стартует группа?" in body
    assert "Хотели бы узнать подробнее?" in body
    for example in _INITIATIVE_CALL_EXAMPLES:
        assert example in body


def test_инициатива_вопрос_по_делу_даже_если_нет(script):
    """В разделе инициативы вопрос по делу годится, даже если на него можно ответить «нет»."""
    body = _top_section(
        build_turn_messages(
            script=script,
            steps=[script.step("city")],
            profile={},
            facts={},
            history=[],
            asides_done=[],
        )[0].content,
        "ИНИЦИАТИВА",
    )
    assert "даже если на него можно ответить «нет»" in body
    assert "нельзя спрашивать, говорить ли" in body
    assert "спрашивать, что решаем" in body


def test_инициатива_не_закругляться_держанием_связи(script):
    """Пока шаги незакрыты, нельзя заканчивать держанием связи; прощаться по смыслу можно."""
    body = _top_section(
        build_turn_messages(
            script=script,
            steps=[script.step("city")],
            profile={},
            facts={},
            history=[],
            asides_done=[],
        )[0].content,
        "ИНИЦИАТИВА",
    )
    assert "незакрытые шаги" in body
    assert "держанием связи" in body
    assert "«всегда на связи»" in body
    assert "«если что — пишите»" in body
    assert "«дайте знать»" in body
    assert "«напомню за день»" in body
    assert "сам простился" in body
    assert "не нужно" in body


def test_добивка_без_новых_строк_инициативы(script):
    """Промпт добивки не содержит новых строк раздела инициативы."""
    from graph.prompts import build_pull_messages

    pull = build_pull_messages(
        script,
        messages=[HumanMessage(content="?")],
        profile={},
        step=script.step("city"),
    )[0].content
    assert "про сам разговор, а не про дело" not in pull
    assert "Признак негодного вопроса" not in pull
    assert "Вопрос по делу годится всегда" not in pull
    assert "держанием связи" not in pull
    assert "«всегда на связи»" not in pull
    assert "«напомню за день»" not in pull
