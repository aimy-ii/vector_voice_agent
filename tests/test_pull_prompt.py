"""Короткая сборка хода вытаскивания и точка её выбора в ``nodes``.

Ход вытаскивания включается, когда предыдущая реплика бота вопроса не
содержала и человек после неё молчит. Нужна одна добивка с вопросом,
поэтому промпт свой, а не полный ход продажи с заплатками.
"""

from __future__ import annotations

import inspect
import re
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from graph import nodes as nodes_module
from graph.prompts import (
    _HARD_FACT_BAN,
    _STEPS_FOCUS_INTRO,
    PULL_TASK,
    SPEECH_RULES,
    build_filler_messages,
    build_pull_messages,
    build_turn_messages,
    build_waiting_messages,
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


def test_вытаскивание_кратно_короче_полного_хода(script_v4):
    """На тех же данных системное сообщение добивки в разы короче полного хода."""
    pull = _pull(script_v4)[0].content
    full = _full(script_v4)[0].content
    assert isinstance(_pull(script_v4)[0], SystemMessage)
    assert len(pull) * 3 < len(full)


def test_вытаскивание_требует_вопроса_в_конце(script_v4):
    """Главное требование — вопрос — стоит до прочих и назван браком."""
    content = _pull(script_v4)[0].content
    assert "реплика заканчивается вопросом со знаком вопроса" in content
    assert "Реплика без вопроса — брак" in content
    assert "требует выбора, решения или конкретных данных" in content
    # Проверка понимания и согласие слушать дальше вопросом не считаются.
    assert "«всё понятно?»" in content
    assert "«продолжим?»" in content
    assert "вопросом не считаются" in content
    task_index = content.index("Задача этого хода одна")
    demand_index = content.index("ГЛАВНОЕ:")
    assert task_index < demand_index < content.index(_HARD_FACT_BAN)


def test_запрет_повтора_стоит_среди_первых_требований(script_v4):
    """Запрет пересказа — второе главное требование, сразу за требованием вопроса."""
    content = _pull(script_v4)[0].content
    assert "последнюю мысль не повторять и не пересказывать другими словами" in content
    assert "уже названное заново не перечислять" in content
    task_index = content.index("Задача этого хода одна")
    question_index = content.index("ГЛАВНОЕ:")
    repeat_index = content.index("ВТОРОЕ ГЛАВНОЕ:")
    fact_ban_index = content.index(_HARD_FACT_BAN)
    assert task_index < question_index < repeat_index < fact_ban_index
    # Рядом с требованием вопроса: между ними ничего постороннего не вклинилось.
    between = content[question_index:repeat_index]
    assert between.count("\n") == 1


def test_запрет_подхватывать_свою_последнюю_мысль(script_v4):
    """Отдельным пунктом закрыт вход через связку-пересказ, с примером из звонка."""
    content = _pull(script_v4)[0].content
    assert "Реплика не начинается с подхвата собственной последней мысли" in content
    assert "«поняла»" in content
    assert "«как и сказала»" in content
    assert "тот же пересказ под новой шапкой" in content
    assert "Поняла, отправляю всё в Max" in content
    repeat_index = content.index("ВТОРОЕ ГЛАВНОЕ:")
    pickup_index = content.index("Реплика не начинается с подхвата")
    assert repeat_index < pickup_index < content.index(_HARD_FACT_BAN)


def test_ограничения_по_числу_слов_в_промпте_нет(script_v4):
    """Потолок длины убран: счёта слов и фраз в требованиях режима не осталось."""
    content = _pull(script_v4)[0].content
    assert "двадцати пяти слов" not in content
    assert "одна-две короткие фразы" not in content
    assert re.search(r"(не длиннее|не больше|не более|максимум)[^.]{0,40}слов", content) is None
    assert "сколько нужно, чтобы человек снова заговорил" in content


def test_предписаний_о_чём_говорить_нет(script_v4):
    """Разбор случаев убран: чем растормошить, модель решает по всему разговору."""
    content = _pull(script_v4)[0].content
    assert "продвинуть текущий шаг" not in content
    assert "продвигать его" not in content
    assert "взять другую тему разговора" not in content
    assert "Чем растормошить — выбор твой" in content
    assert "Прочитай весь разговор целиком" in content
    assert "тоже твоё решение" in content
    # Формулировка самодостаточна: ссылок на разделы, которых в промпте нет.
    for reference in ("перечисленных ниже", "из перечисленных", "ЕЩЁ НЕ ЗАКРЫТО", "раздел"):
        assert reference not in content


def test_вытаскивание_держит_запрет_фактов_и_требования_шага(script_v4):
    """Жёсткий запрет на факты и требования ведущего шага остаются."""
    step = script_v4.step("terms")
    content = _pull(script_v4)[0].content
    assert _HARD_FACT_BAN in content
    assert step.name in content
    assert step.requirements in content
    assert "Шаг, который сейчас ведём:" in content


def test_вытаскивание_без_правил_речи_сводки_и_шапки(script_v4):
    """Полного перечня правил, сводки сценария и второго шага в промпте нет."""
    content = _pull(script_v4)[0].content
    for rule in SPEECH_RULES:
        assert rule not in content
    assert script_v4.summary
    assert script_v4.summary not in content
    assert "# ПРАВИЛА РЕЧИ" not in content
    assert "ЕЩЁ НЕ ЗАКРЫТО" not in content
    assert _STEPS_FOCUS_INTRO not in content
    assert PULL_TASK not in content
    assert script_v4.step("theory_format").requirements not in content


def test_вытаскивание_без_повторяющихся_блоков_нехватки(script_v4):
    """Нехватка данных отмечается не больше одного раза — у ведущего шага."""
    without_context = _pull(script_v4, context_text="")[0].content
    with_context = _pull(script_v4)[0].content
    assert without_context.count("**Не хватает данных**") == 1
    assert with_context.count("**Не хватает данных**") <= 1
    full = _full(script_v4, context_text="")[0].content
    assert full.count("**Не хватает данных**") > 1


def test_профиль_и_запрет_прощаться_первым_на_месте(script_v4):
    """Форма разговора и границы режима из промпта не выпали."""
    content = _pull(script_v4)[0].content
    assert profile_block(script_v4, {"caller_name": "Андрей"}, pending_fields=[]) in content
    assert "не прощаться первым" in content
    assert "Не здороваться заново" in content


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
    assert "Задача этого хода одна: растормошить человека" in content
    assert PULL_TASK not in content
    assert "# ПРАВИЛА РЕЧИ" not in content


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
    assert "Задача этого хода одна" not in content
