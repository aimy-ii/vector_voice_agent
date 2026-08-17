"""Короткая сборка хода вытаскивания и точка её выбора в ``nodes``.

Ход вытаскивания включается, когда предыдущая реплика бота вопроса не
содержала и человек после неё молчит. Нужна следующая реплика разговора,
поэтому промпт свой, а не полный ход продажи с заплатками.

Промпт собран вокруг порядка рассуждения — прочитать разговор, найти
незакрытое, продолжить с последней реплики бота, сказать новое с вопросом.
Шапка, закрытые шаги и следующий шаг приходят тем же путём, что в
полную сборку. Запрет повтора — отдельный пункт рядом с требованием
вопроса.
"""

from __future__ import annotations

import inspect
import re
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from core.config import settings
from graph import nodes as nodes_module
from graph.prompts import (
    _HARD_FACT_BAN,
    _PULL_BOUNDS,
    _PULL_LAST_INTRO,
    _PULL_NEXT_INTRO,
    _PULL_NO_REPEAT,
    _PULL_QUESTION,
    _PULL_SITUATION,
    _PULL_STEPS_INTRO,
    _PULL_STEPS_NONE,
    _PULL_THINKING,
    _STEPS_FOCUS_INTRO,
    PULL_TASK,
    SPEECH_RULES,
    _gender_speech_rule,
    build_filler_messages,
    build_pull_messages,
    build_turn_messages,
    build_waiting_messages,
    closed_steps_block,
    profile_block,
)
from graph.state import new_state_defaults

#: Контекст хода: столько же справочного материала, сколько у полной сборки.
_CONTEXT = (
    "Цена: стоимость обучения — от 43900 рублей.\n"
    "Срок обучения: 2,5 месяца.\n"
    "Форматы теории: очно, дистанционно."
)

#: Хвост звонка, на котором разговор повис: бот сказал утверждение, человек молчит.
_HISTORY = [
    HumanMessage(content="Сколько всё это стоит?"),
    AIMessage(content="Обучение стоит от 43900 рублей."),
    HumanMessage(content="Понятно."),
    AIMessage(content="И ещё можно вернуть тринадцать процентов через налоговый вычет."),
]

#: Разговор длиннее прежней обрезки в шесть сообщений: начало обязано доходить
#: до модели целиком, иначе она не видит, что уже обещано.
_LONG_HISTORY = [
    HumanMessage(content="Добрый день, звоню про обучение."),
    AIMessage(content="Добрый день, академия «Вектор». Как к Вам обращаться?"),
    HumanMessage(content="Андрей."),
    AIMessage(content="Андрей, а где Вам удобно заниматься?"),
    HumanMessage(content="У Просвещения."),
    AIMessage(content="Подберу ближайший филиал. Сколько времени готовы уделять?"),
    HumanMessage(content="Вечерами после работы."),
    AIMessage(content="Хорошо, отправлю информацию в Max."),
]


def _pull(script, **extra: Any) -> list[Any]:
    """Собирает короткое вытаскивание на общих для тестов данных."""
    kwargs: dict[str, Any] = {
        "messages": _HISTORY,
        "profile": {"caller_name": "Андрей"},
        "pending_fields": [],
        "step": script.step("terms"),
        "facts": {},
        "context_text": _CONTEXT,
    }
    kwargs.update(extra)
    return build_pull_messages(script, **kwargs)


def _full(script, **extra: Any) -> list[Any]:
    """Собирает полный ход на тех же данных — для сравнения длин."""
    kwargs: dict[str, Any] = {
        "script": script,
        "steps": [script.step("terms"), script.step("theory_format")],
        "profile": {"caller_name": "Андрей"},
        "facts": {},
        "history": _HISTORY,
        "asides_done": [],
        "context_text": _CONTEXT,
        "turn_kind": "pull",
        "mode": "pull",
    }
    kwargs.update(extra)
    return build_turn_messages(**kwargs)


def test_вытаскивание_короче_полного_хода_на_тех_же_данных(script_v4):
    """На тех же шапке, закрытых и следующем системное сообщение короче полного хода."""
    extra = {
        "steps": [script_v4.step("terms"), script_v4.step("theory_format")],
        "closed_steps": [script_v4.step("greeting")],
        "next_step": script_v4.step("price"),
    }
    pull = _pull(script_v4, **extra)[0].content
    full = _full(script_v4, **extra)[0].content
    assert isinstance(_pull(script_v4)[0], SystemMessage)
    assert len(pull) < len(full)


def test_промпт_ведёт_по_порядку_рассуждения(script_v4):
    """Ядро промпта — последовательность действий, и шаги идут в своём порядке."""
    content = _pull(script_v4)[0].content
    intro = content.index("Как рассуждать перед репликой")
    read = content.index("Прочитай разговор целиком")
    hanging = content.index("Найди, на чём разговор повис")
    cont = content.index("Продолжай с последней своей реплики")
    lead = content.index("Веди разговор к тому, что ещё не сделано")
    fresh = content.index("Скажи то, чего в этом разговоре ещё не было")
    assert content.index(_PULL_SITUATION) < intro < read < hanging < cont < lead < fresh
    for number, opening in enumerate(("Прочитай", "Найди", "Продолжай", "Веди", "Скажи"), start=1):
        assert f"{number}. {opening}" in content


def test_тема_берётся_из_шагов_сценария(script_v4):
    """В порядке рассуждения — прямое указание брать тему из шагов, а не выдумывать."""
    content = _pull(script_v4)[0].content
    topic = content[content.index("Веди разговор к тому, что ещё не сделано") :]
    assert "к незакрытому шагу сценария из списка ниже или к следующему" in topic
    assert "Посторонних тем не бывает" in topic
    assert "либо шаг сценария, либо то, что человек сам поднял" in topic
    assert "Выдумывать тему нельзя" in topic


def test_требование_вопроса_на_месте(script_v4):
    """Реплика заканчивается вопросом; пустые вопросы отсечены поимённо."""
    content = _pull(script_v4)[0].content
    assert _PULL_QUESTION in content
    assert "требует от человека выбора, решения или конкретных данных" in content
    for empty in ("«всё понятно?»", "«остались вопросы?»", "«продолжим?»", "«что скажете?»"):
        assert empty in content
    assert "вопросом не считаются" in content
    # Требования проверяют готовую реплику, поэтому идут после порядка.
    assert content.index(_PULL_THINKING) < content.index(_PULL_QUESTION)


def test_последняя_реплика_бота_стоит_выше_порядка_рассуждения(script_v4):
    """Промпт добивки содержит последнюю реплику бота дословно, выше порядка рассуждения."""
    last_reply = (
        "Коломяжский проспект работает каждый день с десяти до восьми, перерыв с двух до трёх."
    )
    content = _pull(
        script_v4,
        messages=[
            HumanMessage(content="Когда работает филиал?"),
            AIMessage(content=last_reply),
        ],
    )[0].content
    assert last_reply in content
    assert content.index(_PULL_SITUATION) < content.index(_PULL_LAST_INTRO)
    assert content.index(_PULL_LAST_INTRO) < content.index(last_reply)
    assert content.index(last_reply) < content.index(_PULL_THINKING)


def test_без_реплик_бота_раздел_последней_реплики_не_печатается(script_v4):
    """Нет реплик бота — раздела нет, остальные части промпта на месте."""
    content = _pull(
        script_v4,
        messages=[HumanMessage(content="Сколько стоит обучение?")],
    )[0].content
    assert _PULL_LAST_INTRO not in content
    assert _PULL_SITUATION in content
    assert _PULL_THINKING in content
    assert _PULL_QUESTION in content
    assert _PULL_NO_REPEAT in content
    assert _HARD_FACT_BAN in content
    assert _PULL_BOUNDS in content


def test_запрет_пересказа_остался_запрета_подхвата_нет(script_v4):
    """Запрет пересказа в промпте на месте, запрета подхвата больше нет."""
    content = _pull(script_v4)[0].content
    assert _PULL_NO_REPEAT in content
    question = content.index(_PULL_QUESTION)
    repeat = content.index(_PULL_NO_REPEAT)
    ban = content.index(_HARD_FACT_BAN)
    assert question < repeat < ban
    assert content[question:ban] == f"{_PULL_QUESTION}\n{_PULL_NO_REPEAT}\n"
    assert "подхвата собственной последней мысли" not in content


def test_ограничения_по_числу_слов_в_промпте_нет(script_v4):
    """Потолок длины не вернулся: объём задан формой реплики, а не счётом слов."""
    content = _pull(script_v4)[0].content
    assert "двадцати пяти слов" not in content
    assert "одна-две короткие фразы" not in content
    assert re.search(r"(не длиннее|не больше|не более|максимум)[^.]{0,40}слов", content) is None
    assert "Одна мысль и вопрос к ней" in content
    assert "выкладывать три темы подряд не нужно" in content


def test_предписаний_о_чём_говорить_из_прошлой_версии_нет(script_v4):
    """Старые разборы случаев сняты; тему берёт порядок рассуждения из шагов."""
    content = _pull(script_v4)[0].content
    for prescription in (
        "продвинуть текущий шаг",
        "продвигать его",
        "взять другую тему разговора",
        "Чем растормошить — выбор твой",
        "тоже твоё решение",
    ):
        assert prescription not in content
    assert "ЕЩЁ НЕ ЗАКРЫТО" not in content
    assert "перечисленных ниже" not in content


def test_жёсткий_запрет_фактов_на_месте(script_v4):
    """Второе требование — не называть данных вне переданных — из промпта не выпало."""
    content = _pull(script_v4)[0].content
    assert _HARD_FACT_BAN in content


def test_шапка_подчинена_порядку_рассуждения(script_v4):
    """Требования шага на месте, но поданы меню тем, а не текстом хода."""
    step = script_v4.step("terms")
    content = _pull(script_v4)[0].content
    assert _PULL_STEPS_INTRO in content
    assert "меню тем к рассуждению выше" in content
    assert "не текст реплики" in content
    assert step.name in content
    assert step.requirements in content
    assert content.index(_PULL_THINKING) < content.index(_PULL_STEPS_INTRO)
    assert content.index(_PULL_STEPS_INTRO) < content.index(step.requirements)
    assert "Шаг, который сейчас ведём:" not in content


def test_вытаскивание_без_правил_речи_и_сводки(script_v4):
    """Полного перечня правил, сводки сценария и разделов полной сборки нет."""
    content = _pull(script_v4)[0].content
    for rule in SPEECH_RULES:
        assert rule not in content
    assert script_v4.summary
    assert script_v4.summary not in content
    assert "# ПРАВИЛА РЕЧИ" not in content
    assert "ЕЩЁ НЕ ЗАКРЫТО" not in content
    assert _STEPS_FOCUS_INTRO not in content
    assert PULL_TASK not in content


def test_в_промпте_вся_шапка_а_не_только_ведущий(script_v4):
    """Все шаги шапки на месте с требованиями; ведущий выделен как основной."""
    lead = script_v4.step("terms")
    hang = script_v4.step("theory_format")
    content = _pull(script_v4, steps=[lead, hang])[0].content
    assert lead.name in content
    assert lead.requirements in content
    assert hang.name in content
    assert hang.requirements in content
    assert "Основной:" in content
    assert "Незакрытые темы:" in content
    assert content.index("Основной:") < content.index(lead.name)
    assert content.index(lead.name) < content.index("Незакрытые темы:")
    assert content.index("Незакрытые темы:") < content.index(hang.name)


def test_ведущий_выделен_и_при_одном_шаге_шапки(script_v4):
    """Один шаг в шапке — помечен основным, раздела незакрытых нет."""
    lead = script_v4.step("terms")
    content = _pull(script_v4, steps=[lead])[0].content
    assert "Основной:" in content
    assert lead.name in content
    assert "Незакрытые темы:" not in content


def test_закрытые_шаги_списком_названий(script_v4):
    """Закрытые шаги — названиями, без требований; пустой список раздела не даёт."""
    closed = [script_v4.step("greeting"), script_v4.step("city")]
    content = _pull(script_v4, closed_steps=closed)[0].content
    assert closed_steps_block(closed) in content
    assert "Уже закрытые шаги" in content
    assert script_v4.step("greeting").name in content
    assert script_v4.step("city").name in content
    assert script_v4.step("greeting").requirements not in content
    assert script_v4.step("city").requirements not in content

    empty = _pull(script_v4, closed_steps=[])[0].content
    assert "Уже закрытые шаги" not in empty
    assert closed_steps_block([]) == ""


def test_следующий_шаг_ориентиром(script_v4):
    """Раздел следующего шага есть при передаче и пропадает, если шага нет."""
    nxt = script_v4.step("price")
    content = _pull(script_v4, next_step=nxt)[0].content
    assert _PULL_NEXT_INTRO in content
    assert "Ориентир, не задание этого хода" in content
    assert nxt.name in content
    assert nxt.requirements in content
    assert content.index(_PULL_STEPS_INTRO) < content.index(_PULL_NEXT_INTRO)

    missing = _pull(script_v4, next_step=None)[0].content
    assert _PULL_NEXT_INTRO not in missing
    assert nxt.name not in missing


def test_следующий_шаг_из_шапки_отдельным_разделом_не_дублируется(script_v4):
    """Шаг, который уже в шапке, вторым разом как «что дальше» не печатается."""
    lead = script_v4.step("terms")
    content = _pull(script_v4, steps=[lead], next_step=lead)[0].content
    assert _PULL_NEXT_INTRO not in content
    assert content.count(lead.name) == 1


def test_вытаскивание_без_повторяющихся_блоков_нехватки(script_v4):
    """Нехватка данных отмечается не больше одного раза — у ведущего шага."""
    without_context = _pull(script_v4, context_text="")[0].content
    with_context = _pull(script_v4)[0].content
    assert without_context.count("**Не хватает данных**") == 1
    assert with_context.count("**Не хватает данных**") <= 1
    full = _full(script_v4, context_text="")[0].content
    assert full.count("**Не хватает данных**") > 1


def test_роль_профиль_и_отметка_молчания_на_месте(script_v4):
    """Роль, тон, «Вы», род, отметка молчания и форма разговора остаются как были."""
    content = _pull(script_v4)[0].content
    assert content.startswith(f"Роль: {settings.agent_name}, {settings.agent_role}")
    assert settings.agent_tone in content
    assert "К клиенту только на «Вы», всегда." in content
    assert _gender_speech_rule(settings.agent_gender) in content
    assert "Реплики человека не было" in content
    assert profile_block(script_v4, {"caller_name": "Андрей"}, pending_fields=[]) in content


def test_границы_хода_на_месте(script_v4):
    """Строка про приветствие и прощание из промпта не выпала."""
    content = _pull(script_v4)[0].content
    assert _PULL_BOUNDS in content


def test_история_разговора_уходит_целиком(script_v4):
    """Разговор длиннее шести реплик доходит до модели весь, включая первую."""
    built = _pull(script_v4, messages=_LONG_HISTORY)
    tail = built[1:]
    assert len(_LONG_HISTORY) > 6
    assert len(tail) == len(_LONG_HISTORY)
    assert [message.content for message in tail] == [message.content for message in _LONG_HISTORY]
    assert tail[0].content == _LONG_HISTORY[0].content

    full = build_turn_messages(
        script=script_v4,
        steps=[script_v4.step("terms")],
        profile={},
        facts={},
        history=_LONG_HISTORY,
        asides_done=[],
        context_text=_CONTEXT,
        turn_kind="pull",
    )
    assert [message.content for message in built[1:]] == [message.content for message in full[1:]]


def test_обрезки_истории_в_сборке_не_осталось(script_v4):
    """Ни параметра обрезки, ни константы хвоста у вытаскивания больше нет."""
    import graph.prompts as prompts_module

    assert "history_limit" not in inspect.signature(build_pull_messages).parameters
    assert not hasattr(prompts_module, "PULL_HISTORY_TURNS")


def test_хвост_диалога_как_в_реплике_ожидания(script_v4):
    """Пустая история даёт ту же отметку молчания, что и реплика ожидания."""
    pull = build_pull_messages(
        script_v4,
        messages=[],
        profile={},
        step=script_v4.step("terms"),
    )
    waiting = build_waiting_messages(
        script_v4,
        messages=[],
        profile={},
        pending_fields=[],
        step=script_v4.step("terms"),
        history_limit=4,
    )
    assert pull[-1].content == waiting[-1].content == "(клиент молчит)"
    assert len(pull) == 2
    assert "Реплики человека не было" in pull[0].content


def _respond(script, **extra: Any) -> list[Any]:
    """Дёргает точку выбора сборки в ``nodes`` с общими аргументами."""
    kwargs: dict[str, Any] = {
        "prompt_kind": "full",
        "script": script,
        "state": new_state_defaults(),
        "history": _HISTORY,
        "profile": {},
        "facts": {},
        "lead": script.step("terms"),
        "head": [script.step("terms")],
        "context_text": _CONTEXT,
        "dynamic_status": "",
        "pending_fields": [],
        "turn_kind": "client",
    }
    kwargs.update(extra)
    return nodes_module._build_respond_messages(**kwargs)


def test_точка_выбора_на_pull_даёт_короткую_сборку(script_v4, monkeypatch):
    """При ``turn_kind="pull"`` собирается вытаскивание, а не полный ход."""
    monkeypatch.setattr(nodes_module.settings, "script_version", "4")
    content = _respond(script_v4, turn_kind="pull")[0].content
    assert _PULL_SITUATION in content
    assert _PULL_THINKING in content
    assert PULL_TASK not in content
    assert "# ПРАВИЛА РЕЧИ" not in content


def test_точка_выбора_на_pull_передаёт_шапку_закрытые_и_следующий(script_v4, monkeypatch):
    """Шапка, закрытые шаги и следующий шаг доходят до короткой сборки тем же путём."""
    monkeypatch.setattr(nodes_module.settings, "script_version", "4")
    lead = script_v4.step("terms")
    hang = script_v4.step("theory_format")
    nxt = script_v4.step("price")
    state = {
        **new_state_defaults(),
        "step_status": {"greeting": "closed", "city": "closed"},
        "next_step": nxt.id,
    }
    content = _respond(
        script_v4,
        turn_kind="pull",
        state=state,
        lead=lead,
        head=[lead, hang],
    )[0].content
    assert lead.name in content
    assert lead.requirements in content
    assert hang.name in content
    assert hang.requirements in content
    assert "Основной:" in content
    assert "Незакрытые темы:" in content
    assert script_v4.step("greeting").name in content
    assert script_v4.step("city").name in content
    assert "Уже закрытые шаги" in content
    assert nxt.name in content
    assert _PULL_NEXT_INTRO in content
    assert content.index("Основной:") < content.index("Незакрытые темы:")
    assert content.index("Незакрытые темы:") < content.index(_PULL_NEXT_INTRO)


def test_точка_выбора_на_остальных_видах_хода_не_изменилась(script_v4):
    """Полный ход, заглушка и ожидание собираются как прежде."""
    for turn_kind in ("client", "continuation", "silence"):
        content = _respond(script_v4, turn_kind=turn_kind)[0].content
        expected = build_turn_messages(
            script=script_v4,
            steps=[script_v4.step("terms")],
            profile={},
            facts={},
            history=_HISTORY,
            asides_done=[],
            next_step=None,
            context_text=_CONTEXT,
            dynamic_status="",
            pending_fields=[],
            turn_kind=turn_kind,
            closed_steps=[],
            mode="normal",
        )[0].content
        assert content == expected

    waiting = _respond(script_v4, prompt_kind="waiting", turn_kind="client")[0].content
    assert (
        waiting
        == build_waiting_messages(
            script_v4,
            messages=_HISTORY,
            profile={},
            pending_fields=[],
            step=script_v4.step("terms"),
            history_limit=nodes_module.settings.waiting_history_limit,
            turn_kind="client",
            context_text=_CONTEXT,
        )[0].content
    )

    filler = _respond(script_v4, prompt_kind="filler", turn_kind="client")[0].content
    assert (
        filler
        == build_filler_messages(
            script_v4,
            messages=_HISTORY,
            history_limit=nodes_module.settings.filler_history_limit,
        )[0].content
    )


def test_повтор_ведущего_шага_остаётся_за_полной_сборкой(script_v4, monkeypatch):
    """Ведущий шаг повторяется выше порога — вытаскивание уступает режиму повтора."""
    from graph.prompts import LEAD_REPEAT_INTRO

    monkeypatch.setattr(nodes_module.settings, "lead_repeat_threshold", 2)
    content = _respond(
        script_v4,
        state={**new_state_defaults(), "lead_repeat": 2},
        turn_kind="pull",
    )[0].content
    assert LEAD_REPEAT_INTRO in content
    assert _PULL_THINKING not in content


def test_шапки_нет_но_указание_откуда_брать_тему_не_висит_в_пустоту(script_v4):
    """Шагов не осталось — вместо шапки печатается прямая замена, а не пустота."""
    content = build_pull_messages(
        script_v4,
        messages=_HISTORY,
        profile={},
        pending_fields=[],
        steps=[],
        facts={},
        context_text=_CONTEXT,
    )[0].content
    assert _PULL_STEPS_NONE in content
    assert "Основной:" not in content
    assert "Незакрытые темы:" not in content
    assert "Если шагов сценария ниже нет" in content
    assert content.index("Если шагов сценария ниже нет") < content.index(_PULL_STEPS_NONE)


def test_полная_сборка_на_пустой_шапке_не_изменилась(script_v4):
    """Правка добивки не тронула поведение полного хода без шагов."""
    content = _full(script_v4, steps=[])[0].content
    assert "Все шаги скрипта закрыты" in content
    assert _PULL_STEPS_NONE not in content
