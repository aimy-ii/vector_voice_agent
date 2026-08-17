"""Короткая сборка хода вытаскивания и точка её выбора в ``nodes``.

Ход вытаскивания включается, когда предыдущая реплика бота вопроса не
содержала и человек после неё молчит. Нужна следующая реплика разговора,
поэтому промпт свой, а не полный ход продажи с заплатками.

Промпт собран вокруг порядка рассуждения — прочитать разговор, найти
незакрытое, сказать новое с вопросом, — а не вокруг перечня запретов:
на перечне модель выдавала пересказ своей же последней мысли с вопросом
на конце, формально не нарушая ни одного пункта. Тесты держат этот порядок
и два оставшихся требования.
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
    _PULL_QUESTION,
    _PULL_SITUATION,
    _PULL_STEP_INTRO,
    _PULL_THINKING,
    _STEPS_FOCUS_INTRO,
    PULL_TASK,
    SPEECH_RULES,
    _gender_speech_rule,
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


def test_промпт_ведёт_по_порядку_рассуждения(script_v4):
    """Ядро промпта — последовательность действий, и шаги идут в своём порядке."""
    content = _pull(script_v4)[0].content
    intro = content.index("Как рассуждать перед репликой")
    read = content.index("Прочитай разговор целиком")
    hanging = content.index("Найди, на чём он повис")
    fresh = content.index("Скажи то, чего в этом разговоре ещё не было")
    assert content.index(_PULL_SITUATION) < intro < read < hanging < fresh
    # Каждый шаг пронумерован: это порядок действий, а не перечень пунктов.
    for number, opening in enumerate(("Прочитай", "Найди", "Скажи"), start=1):
        assert f"{number}. {opening}" in content


def test_первый_шаг_опирается_на_весь_разговор(script_v4):
    """Шаг «прочитать разговор» перечисляет, что именно в нём искать."""
    content = _pull(script_v4)[0].content
    assert "с самой первой реплики" in content
    assert "что ему уже объяснили, что пообещали сделать" in content
    assert "Разговор подан полностью именно для этого" in content


def test_второй_шаг_ищет_незакрытое(script_v4):
    """Место, с которого продолжают, — незакрытое, а не последняя своя мысль."""
    content = _pull(script_v4)[0].content
    hanging = content[content.index("Найди, на чём он повис") :]
    for mark in (
        "Вопрос, оставшийся без ответа",
        "Мысль, которую не договорили",
        "Обещанное и не выполненное",
    ):
        assert mark in hanging
    assert "Это и есть место, с которого продолжают" in hanging


def test_запрет_повтора_живёт_в_порядке_а_не_отдельным_пунктом(script_v4):
    """Пункты «ГЛАВНОЕ» и «ВТОРОЕ ГЛАВНОЕ» сняты, повтор закрывает третий шаг.

    Прежний промпт запрещал пересказ отдельным требованием, и модель обходила
    запрет: подхватывала собственную мысль и приделывала вопрос. Теперь того
    же добивается порядок — сказать надо то, чего в разговоре ещё не было.
    """
    content = _pull(script_v4)[0].content
    assert "ГЛАВНОЕ:" not in content
    assert "последнюю мысль не повторять" not in content
    assert "Человек это слышал — потому и молчит" not in content
    assert "Скажи то, чего в этом разговоре ещё не было" in content


def test_подхват_своей_мысли_свёрнут_в_третий_шаг(script_v4):
    """Связка-шапка закрыта, но как часть описания реплики, а не пунктом списка."""
    content = _pull(script_v4)[0].content
    third = content[content.index("Скажи то, чего в этом разговоре ещё не было") :]
    assert "Начинай сразу с нового" in third
    assert "«как и сказала»" in third
    assert "шапка про уже сказанное" in third
    # Отдельного пункта с разбором плохого примера из звонка больше нет.
    assert "Реплика не начинается с подхвата собственной последней мысли" not in content
    assert "Поняла, отправляю всё в Max" not in content


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


def test_требований_к_реплике_осталось_два(script_v4):
    """Кроме вопроса и запрета фактов, требований в промпте нет.

    Девять указаний подряд прошлой версии сведены к двум: между ними и
    границами хода ничего постороннего не вклинивается.
    """
    content = _pull(script_v4)[0].content
    question = content.index(_PULL_QUESTION)
    ban = content.index(_HARD_FACT_BAN)
    bounds = content.index(_PULL_BOUNDS)
    assert question < ban < bounds
    assert content[question:bounds].count("\n") == 2


def test_ограничения_по_числу_слов_в_промпте_нет(script_v4):
    """Потолок длины не вернулся: объём задан формой реплики, а не счётом слов."""
    content = _pull(script_v4)[0].content
    assert "двадцати пяти слов" not in content
    assert "одна-две короткие фразы" not in content
    assert re.search(r"(не длиннее|не больше|не более|максимум)[^.]{0,40}слов", content) is None
    assert "Одна мысль и вопрос к ней" in content
    assert "выкладывать три темы подряд не нужно" in content


def test_предписаний_о_чём_говорить_нет(script_v4):
    """Ни разбора случаев, ни «выбор твой»: вместо них порядок рассуждения."""
    content = _pull(script_v4)[0].content
    for prescription in (
        "продвинуть текущий шаг",
        "продвигать его",
        "взять другую тему разговора",
        "Чем растормошить — выбор твой",
        "тоже твоё решение",
    ):
        assert prescription not in content
    # Формулировки самодостаточны: ссылок на разделы, которых в промпте нет.
    for reference in ("перечисленных ниже", "из перечисленных", "ЕЩЁ НЕ ЗАКРЫТО", "раздел"):
        assert reference not in content


def test_жёсткий_запрет_фактов_на_месте(script_v4):
    """Второе требование — не называть данных вне переданных — из промпта не выпало."""
    content = _pull(script_v4)[0].content
    assert _HARD_FACT_BAN in content


def test_блок_шага_подчинён_порядку_рассуждения(script_v4):
    """Требования шага на месте, но поданы ориентиром, а не текстом хода.

    Блок шага конкретнее любого рассуждения и перетягивал ход на себя:
    предыдущая реплика бота выросла из того же шага, и модель отрабатывала
    его второй раз подряд. Врезка перед блоком ставит его на место.
    """
    step = script_v4.step("terms")
    content = _pull(script_v4)[0].content
    assert _PULL_STEP_INTRO in content
    assert "не текст, который надо произнести сейчас" in content
    assert "предыдущая твоя реплика выросла отсюда же" in content
    assert step.name in content
    assert step.requirements in content
    assert content.index(_PULL_STEP_INTRO) < content.index(step.requirements)
    # Прежней командной шапки над блоком не осталось.
    assert "Шаг, который сейчас ведём:" not in content


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


def test_роль_профиль_и_отметка_молчания_на_месте(script_v4):
    """Роль, тон, «Вы», род, отметка молчания и форма разговора остаются как были."""
    content = _pull(script_v4)[0].content
    assert content.startswith(f"Роль: {settings.agent_name}, {settings.agent_role}")
    assert settings.agent_tone in content
    assert "К клиенту только на «Вы», всегда." in content
    assert _gender_speech_rule(settings.agent_gender) in content
    assert "Реплики человека не было" in content
    assert profile_block(script_v4, {"caller_name": "Андрей"}, pending_fields=[]) in content


def test_границы_хода_свёрнуты_в_одну_строку(script_v4):
    """Приветствие и прощание остались, но одной строкой вместо отдельного пункта."""
    content = _pull(script_v4)[0].content
    assert _PULL_BOUNDS in content
    assert "заново не здороваться" in content
    assert "не прощаться первым" in content
    assert "\n" not in _PULL_BOUNDS


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
