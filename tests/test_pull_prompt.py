"""Короткая сборка хода вытаскивания и точка её выбора в ``nodes``.

Ход вытаскивания включается, когда предыдущая реплика бота вопроса не
содержала и человек после неё молчит. Нужна одна добивка с вопросом,
поэтому промпт свой, а не полный ход продажи с заплатками.
"""

from __future__ import annotations

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


def _pull(script, **extra: Any) -> list[Any]:
    """Собирает короткое вытаскивание на общих для тестов данных."""
    kwargs: dict[str, Any] = {
        "messages": _HISTORY,
        "profile": {"caller_name": "Андрей"},
        "pending_fields": [],
        "step": script.step("terms"),
        "facts": {},
        "history_limit": 6,
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


def test_вытаскивание_ограничивает_длину_и_запрещает_повтор(script_v4):
    """В промпте есть потолок длины и запрет пересказывать последнюю мысль."""
    content = _pull(script_v4)[0].content
    assert "одна-две короткие фразы" in content
    assert "не длиннее двадцати пяти слов" in content
    assert "Последнюю мысль не повторять и не пересказывать другими словами" in content
    assert "уже названное заново не перечислять" in content


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


def test_указание_о_выборе_темы_исполнимо_при_одном_шаге(script_v4):
    """Выбор темы описан самодостаточно: ссылок на отсутствующие разделы нет."""
    content = _pull(script_v4)[0].content
    assert "есть чем продвинуть текущий шаг — продвигать его" in content
    assert "взять другую тему разговора, которая ещё не закрыта" in content
    assert "продвигать текущий шаг по тому, что уже известно из разговора" in content
    for reference in ("перечисленных ниже", "из перечисленных", "ЕЩЁ НЕ ЗАКРЫТО", "раздел"):
        assert reference not in content


def test_хвост_диалога_как_в_реплике_ожидания(script_v4):
    """Пустая история даёт ту же отметку молчания, что и реплика ожидания."""
    pull = build_pull_messages(
        script_v4,
        messages=[],
        profile={},
        step=script_v4.step("terms"),
        history_limit=6,
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

    limited = _pull(script_v4, history_limit=2)
    assert len(limited) - 1 == 2
    assert limited[-1].content == _HISTORY[-1].content
    assert "Реплики человека не было" in limited[0].content


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
    assert "Задача этого хода одна: вытянуть человека на ответ" in content
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
