"""Тесты синхронного чекера."""

from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage

from graph.checker import (
    CheckerVerdict,
    checker_system_prompt,
    close_delivered_inform,
    history_slice_for,
    run_checker,
)
from script.store import ScriptProgress


class FakeChecker:
    """Заглушка модели чекера."""

    def __init__(self, verdicts: list[CheckerVerdict | None]) -> None:
        self.verdicts = list(verdicts)
        self.calls: list[dict] = []

    async def judge(
        self,
        *,
        history_slice,
        client_reply,
        step,
        step_text,
        attempts: int = 0,
        age: int = 0,
        in_work: bool = False,
    ):
        self.calls.append(
            {
                "history_slice": history_slice,
                "client_reply": client_reply,
                "step_id": step.id,
                "step_text": step_text,
                "attempts": attempts,
                "age": age,
                "in_work": in_work,
            }
        )
        if not self.verdicts:
            return None
        return self.verdicts.pop(0)


async def test_вход_разделён_история_и_реплика_не_склеены(script):
    client = FakeChecker([CheckerVerdict(reply_usable=True, step_closed=False)])
    messages = [
        AIMessage(content="Как к вам обращаться?"),
        HumanMessage(content="Меня зовут Андрей"),
    ]
    progress = ScriptProgress(
        status={"name": "pending"}, attempts={"name": 1}, taken_turn={"name": 1}
    )
    await run_checker(
        script=script,
        progress=progress,
        messages=messages,
        profile={},
        turn=2,
        client=client,
    )
    assert client.calls
    call = client.calls[0]
    assert "Меня зовут Андрей" == call["client_reply"]
    assert "Меня зовут Андрей" not in call["history_slice"]
    assert "Как к вам обращаться" in call["history_slice"]


async def test_срез_от_взятия_самого_старого(script):
    steps = [script.step("name"), script.step("city")]
    progress = ScriptProgress(
        attempts={"name": 1, "city": 1},
        taken_turn={"name": 1, "city": 3},
    )
    messages = [
        HumanMessage(content="привет"),
        AIMessage(content="имя?"),
        HumanMessage(content="Андрей"),
        AIMessage(content="город?"),
        HumanMessage(content="Пермь"),
    ]
    sliced = history_slice_for(messages, steps=steps, progress=progress, turn=4)
    # Последняя реплика клиента отрезана.
    assert sliced[-1].type != "human" or sliced[-1].content != "Пермь"


async def test_для_шага_не_в_работе_история_с_начала(script):
    steps = [script.step("city")]
    progress = ScriptProgress(attempts={"city": 0}, in_work=[])
    messages = [
        HumanMessage(content="я из Перми"),
        AIMessage(content="как зовут?"),
        HumanMessage(content="Игорь"),
    ]
    sliced = history_slice_for(messages, steps=steps, progress=progress, turn=2)
    assert sliced[0].content == "я из Перми" or any("Перми" in str(m.content) for m in sliced[:-1])


def test_срез_отрезает_хвост_при_разной_пунктуации(script):
    """Хвост уходит, если реплика отличается от human только знаками."""
    steps = [script.step("name")]
    progress = ScriptProgress(attempts={"name": 1}, taken_turn={"name": 1})
    original = "Да, механика."
    messages = [
        AIMessage(content="Какая коробка?"),
        HumanMessage(content=original),
    ]
    sliced = history_slice_for(
        messages,
        steps=steps,
        progress=progress,
        turn=2,
        reply="да механика",
    )
    assert not any(m.type == "human" and m.content == original for m in sliced)


def test_срез_отрезает_хвост_при_регистре_и_ё(script):
    """Хвост уходит при отличии только регистром и «ё»/«е»."""
    steps = [script.step("name")]
    progress = ScriptProgress(attempts={"name": 1}, taken_turn={"name": 1})
    original = "Всё понятно"
    messages = [
        AIMessage(content="Вопрос?"),
        HumanMessage(content=original),
    ]
    sliced = history_slice_for(
        messages,
        steps=steps,
        progress=progress,
        turn=2,
        reply="все понятно",
    )
    assert not any(m.type == "human" and m.content == original for m in sliced)


def test_срез_не_отрезает_чужой_хвост(script):
    """Хвост остаётся, если последняя реплика в истории действительно другая."""
    steps = [script.step("city")]
    progress = ScriptProgress(attempts={"city": 1}, taken_turn={"city": 1})
    messages = [
        AIMessage(content="имя?"),
        HumanMessage(content="Андрей"),
        AIMessage(content="город?"),
    ]
    sliced = history_slice_for(
        messages,
        steps=steps,
        progress=progress,
        turn=3,
        reply="я из Перми",
    )
    assert any(m.content == "Андрей" for m in sliced)
    assert sliced[-1].type == "ai"


def test_срез_не_меняет_текст_сообщений(script):
    """Тексты в срезе совпадают с исходными символ в символ."""
    steps = [script.step("name")]
    progress = ScriptProgress(attempts={"name": 1}, taken_turn={"name": 1})
    messages = [
        AIMessage(content="Как к вам обращаться?"),
        HumanMessage(content="Меня зовут Андрей!"),
        AIMessage(content="Из какого города вы звоните?"),
        HumanMessage(content="Да, механика."),
    ]
    sliced = history_slice_for(
        messages,
        steps=steps,
        progress=progress,
        turn=2,
        reply="да механика",
    )
    # Хвост отрезан; оставшиеся тексты — без нормализации.
    assert [m.content for m in sliced] == [
        "Как к вам обращаться?",
        "Меня зовут Андрей!",
        "Из какого города вы звоните?",
    ]
    for original, kept in zip(messages[:-1], sliced, strict=True):
        assert kept.content == original.content


def test_срез_при_reply_none_отрезает_хвостовой_human(script):
    """При ``reply is None`` хвостовой human отрезается безусловно."""
    steps = [script.step("name")]
    progress = ScriptProgress(attempts={"name": 1}, taken_turn={"name": 1})
    messages = [
        AIMessage(content="Как зовут?"),
        HumanMessage(content="Совершенно другая фраза"),
    ]
    sliced = history_slice_for(
        messages,
        steps=steps,
        progress=progress,
        turn=2,
        reply=None,
    )
    assert not any(m.type == "human" for m in sliced)
    assert sliced[-1].content == "Как зовут?"


async def test_реплика_не_годится_цикл_рвётся(script):
    client = FakeChecker([CheckerVerdict(reply_usable=False, step_closed=False)])
    progress = ScriptProgress(status={"name": "pending"}, attempts={"name": 1})
    updated, _ = await run_checker(
        script=script,
        progress=progress,
        messages=[HumanMessage(content="э-э")],
        profile={},
        turn=1,
        client=client,
    )
    assert updated.status.get("name") != "closed"
    assert len(client.calls) == 1


async def test_цикл_останавливается_на_первом_незакрывшемся(script):
    client = FakeChecker(
        [
            CheckerVerdict(reply_usable=True, step_closed=False),
            CheckerVerdict(reply_usable=True, step_closed=True),
        ]
    )
    progress = ScriptProgress(
        status={"name": "pending", "city": "pending"},
        attempts={"name": 1, "city": 1},
    )
    # caller_name уже в профиле — name закрывает код до модели.
    # city: модель не закрыла — цикл рвётся, who_studies не смотрим.
    updated, _ = await run_checker(
        script=script,
        progress=progress,
        messages=[HumanMessage(content="Андрей, Пермь")],
        profile={"caller_name": "Андрей"},
        turn=2,
        client=client,
    )
    assert updated.status.get("name") == "closed"
    assert updated.status.get("city") != "closed"
    assert len(client.calls) == 1
    assert client.calls[0]["step_id"] == "city"


async def test_модель_не_ответила_шаги_не_тронуты(script):
    client = FakeChecker([None])
    progress = ScriptProgress(status={"name": "pending"}, attempts={"name": 1})
    updated, _ = await run_checker(
        script=script,
        progress=progress,
        messages=[HumanMessage(content="Андрей")],
        profile={},
        turn=1,
        client=client,
    )
    assert updated.status.get("name") == "pending"


async def test_порог_попыток_сам_по_себе_не_закрывает(script):
    """Большой счётчик без вердикта судьи шаг не закрывает."""
    client = FakeChecker([CheckerVerdict(reply_usable=True, step_closed=False)])
    progress = ScriptProgress(status={"name": "pending"}, attempts={"name": 2})
    updated, closures = await run_checker(
        script=script,
        progress=progress,
        messages=[HumanMessage(content="потом скажу")],
        profile={},
        turn=6,
        client=client,
        attempt_limit=2,
    )
    assert updated.status.get("name") != "closed"
    assert ("name", "счётчик") not in closures
    assert len(client.calls) == 1
    assert client.calls[0]["step_id"] == "name"


async def test_закрытие_по_счётчику_больше_не_срабатывает(script):
    """Порог attempts не закрывает шаг; без вердикта остаётся открытым."""
    client = FakeChecker([CheckerVerdict(reply_usable=True, step_closed=False)])
    progress = ScriptProgress(status={"name": "pending"}, attempts={"name": 2})
    updated, closures = await run_checker(
        script=script,
        progress=progress,
        messages=[HumanMessage(content="не сейчас")],
        profile={},
        turn=5,
        client=client,
        attempt_limit=2,
    )
    assert updated.status.get("name") != "closed"
    assert ("name", "счётчик") not in closures
    assert len(client.calls) == 1

    progress2 = ScriptProgress(status={"name": "pending"}, attempts={"name": 1})
    updated2, closures2 = await run_checker(
        script=script,
        progress=progress2,
        messages=[HumanMessage(content="не сейчас")],
        profile={},
        turn=5,
        client=client,
        attempt_limit=2,
    )
    assert updated2.status.get("name") != "closed"
    assert ("name", "счётчик") not in closures2


async def test_не_в_работе_модель_не_вызывается(script):
    client = FakeChecker([CheckerVerdict(reply_usable=True, step_closed=True)])
    progress = ScriptProgress(status={}, attempts={}, in_work=[])
    updated, _ = await run_checker(
        script=script,
        progress=progress,
        messages=[HumanMessage(content="Меня зовут Андрей, я из Перми")],
        profile={},
        turn=1,
        client=client,
    )
    assert client.calls == []
    assert updated.status.get("name") != "closed"


async def test_question_с_пустыми_fills_закрывается_по_вердикту(script):
    """Вердикт чекера окончательный: пустой профиль закрытию не мешает."""
    client = FakeChecker([CheckerVerdict(reply_usable=True, step_closed=True)])
    progress = ScriptProgress(
        status={"theory_format": "pending"},
        attempts={"theory_format": 1},
    )
    # Предшественники закрыты, чтобы theory_format был доступен.
    for step_id in ("name", "city", "who_studies", "experience", "transmission", "terms"):
        progress.status[step_id] = "closed"
        progress.attempts[step_id] = 1
    updated, closures = await run_checker(
        script=script,
        progress=progress,
        messages=[HumanMessage(content="Очно в классе")],
        profile={
            "caller_name": "Мария",
            "city": "Пермь",
            "student_is_caller": "да",
            "experience": "впервые",
            "transmission": "механика",
        },
        turn=5,
        client=client,
    )
    assert updated.status.get("theory_format") == "closed"
    assert ("theory_format", "диалог") in closures
    assert any(c["step_id"] == "theory_format" for c in client.calls)


async def test_закрытие_не_зависит_от_порядка_заполнителя(script):
    """Заполнитель раньше или позже — вердикт модели даёт один итог."""
    progress_base = ScriptProgress(
        status={"city": "pending"},
        attempts={"city": 1},
    )
    progress_base.status["name"] = "closed"
    progress_base.attempts["name"] = 1
    reply = "В Санкт-Петербурге"
    verdicts = [CheckerVerdict(reply_usable=True, step_closed=True)]

    before_fill, closures_before = await run_checker(
        script=script,
        progress=ScriptProgress.from_mapping(progress_base.to_dict()),
        messages=[HumanMessage(content=reply)],
        profile={"caller_name": "Андрей"},
        turn=2,
        client=FakeChecker(list(verdicts)),
    )
    after_fill, closures_after = await run_checker(
        script=script,
        progress=ScriptProgress.from_mapping(progress_base.to_dict()),
        messages=[HumanMessage(content=reply)],
        profile={"caller_name": "Андрей", "city": "Санкт-Петербург"},
        turn=2,
        client=FakeChecker(list(verdicts)),
    )
    assert before_fill.status.get("city") == "closed"
    assert after_fill.status.get("city") == "closed"
    assert ("city", "диалог") in closures_before
    assert ("city", "диалог") in closures_after


async def test_question_с_заполненным_fills_закрывается(script):
    """Поле профиля заполнилось — шаг закрывает код без модели."""
    client = FakeChecker([CheckerVerdict(reply_usable=True, step_closed=True)])
    progress = ScriptProgress(
        status={"theory_format": "pending"},
        attempts={"theory_format": 1},
    )
    for step_id in ("name", "city", "who_studies", "experience", "transmission", "terms"):
        progress.status[step_id] = "closed"
        progress.attempts[step_id] = 1
    updated, _ = await run_checker(
        script=script,
        progress=progress,
        messages=[HumanMessage(content="Очно в классе")],
        profile={
            "caller_name": "Мария",
            "city": "Пермь",
            "student_is_caller": "да",
            "experience": "впервые",
            "transmission": "механика",
            "theory_format": "очно",
        },
        turn=5,
        client=client,
    )
    assert updated.status["theory_format"] == "closed"
    assert not any(c["step_id"] == "theory_format" for c in client.calls)


async def test_inform_не_попадает_в_pending(script):
    """Inform чекеру на суд не отдаём — закрытие только по доставке."""
    client = FakeChecker([CheckerVerdict(reply_usable=True, step_closed=True)])
    progress = ScriptProgress(
        status={"terms": "pending"},
        attempts={"terms": 1},
    )
    for step_id in ("name", "city", "who_studies", "experience", "transmission"):
        progress.status[step_id] = "closed"
        progress.attempts[step_id] = 1
    updated, _ = await run_checker(
        script=script,
        progress=progress,
        messages=[HumanMessage(content="У меня проспект Просвещения, адрес")],
        profile={
            "caller_name": "Мария",
            "city": "Пермь",
            "student_is_caller": "да",
            "experience": "впервые",
            "transmission": "механика",
        },
        turn=4,
        client=client,
    )
    assert not any(c["step_id"] == "terms" for c in client.calls)
    assert updated.status.get("terms") != "closed"


async def test_inform_check_закрывается_ответом_на_проверку(script):
    """inform_check закрывается чекером по ответу на проверочный вопрос."""
    from graph.checker import closure_criterion

    practice = script.step("practice")
    assert practice.kind == "inform_check"
    assert practice.check_question
    assert "проверочн" in closure_criterion(practice)

    client = FakeChecker([CheckerVerdict(reply_usable=True, step_closed=True)])
    progress = ScriptProgress(
        status={"practice": "pending"},
        attempts={"practice": 1},
    )
    for step_id in (
        "name",
        "city",
        "who_studies",
        "experience",
        "transmission",
        "terms",
        "theory_format",
        "included",
    ):
        progress.status[step_id] = "closed"
        progress.attempts[step_id] = 1
    updated, _ = await run_checker(
        script=script,
        progress=progress,
        messages=[HumanMessage(content="Да, такой подход мне подходит")],
        profile={
            "caller_name": "Мария",
            "city": "Пермь",
            "student_is_caller": "да",
            "experience": "впервые",
            "transmission": "механика",
            "theory_format": "очно",
        },
        turn=6,
        client=client,
    )
    assert client.calls and client.calls[0]["step_id"] == "practice"
    assert updated.status["practice"] == "closed"


def test_close_delivered_inform_только_inform(script):
    """Доставка закрывает inform; inform_check остаётся на чекере."""
    progress = ScriptProgress(
        status={"terms": "pending", "practice": "pending"},
        attempts={"terms": 1, "practice": 1},
    )
    after_inform = close_delivered_inform(
        script=script,
        progress=progress,
        pending_step="terms",
        delivered=True,
    )
    assert after_inform.status["terms"] == "closed"

    after_check = close_delivered_inform(
        script=script,
        progress=progress,
        pending_step="practice",
        delivered=True,
    )
    assert after_check.status.get("practice") != "closed"


def test_checker_system_prompt_ветка_in_work():
    """Промпт для шага в работе содержит возраст и отличается от остального."""
    in_work = checker_system_prompt(in_work=True)
    idle = checker_system_prompt(in_work=False)

    assert in_work != idle
    assert "возраст" in in_work.lower()
    assert "бессмысленно" in in_work
    assert "где угодно" in idle
    assert "срезе" in idle


async def test_фейковый_судья_получает_возраст(script):
    """В судью уходит возраст шага и пометка in_work."""
    client = FakeChecker([CheckerVerdict(reply_usable=True, step_closed=False)])
    progress = ScriptProgress(
        status={"name": "pending"},
        attempts={"name": 2},
        taken_turn={"name": 1},
        in_work=["name"],
    )
    await run_checker(
        script=script,
        progress=progress,
        messages=[HumanMessage(content="Андрей")],
        profile={},
        turn=4,
        client=client,
        attempt_limit=3,
    )
    assert client.calls
    assert client.calls[0]["attempts"] == 2
    assert client.calls[0]["age"] == 3
    assert client.calls[0]["in_work"] is True


async def test_просьба_повторить_не_закрывает_и_цикл_рвётся(script):
    """step_closed=false на просьбе повторить — шаг pending, дальше по шапке нет."""
    client = FakeChecker(
        [
            CheckerVerdict(reply_usable=True, step_closed=False),
            CheckerVerdict(reply_usable=True, step_closed=True),
        ]
    )
    progress = ScriptProgress(
        status={"name": "pending", "city": "pending"},
        attempts={"name": 1, "city": 1},
    )
    updated, _ = await run_checker(
        script=script,
        progress=progress,
        messages=[HumanMessage(content="отвлёкся, не услышал, повторите")],
        profile={},
        turn=2,
        client=client,
    )
    assert updated.status.get("name") == "pending"
    assert updated.status.get("city") != "closed"
    assert len(client.calls) == 1
    assert client.calls[0]["step_id"] == "name"


async def test_на_проверку_только_взятые_в_работу(script, caplog):
    """Судье идут только in_work и незакрытые; остальные — «не в работе»."""
    import logging

    client = FakeChecker([CheckerVerdict(reply_usable=True, step_closed=False)])
    progress = ScriptProgress(
        status={"name": "pending", "city": "pending"},
        attempts={"name": 1},
        taken_turn={"name": 1},
        in_work=["name"],
    )
    with caplog.at_level(logging.INFO, logger="graph.checker"):
        await run_checker(
            script=script,
            progress=progress,
            messages=[HumanMessage(content="Андрей")],
            profile={},
            turn=2,
            client=client,
        )
    assert len(client.calls) == 1
    assert client.calls[0]["step_id"] == "name"
    pending_logs = [r.message for r in caplog.records if "[check|pending]" in r.message]
    assert pending_logs
    assert "на проверку: [name(1)]" in pending_logs[0]
    assert "city — не в работе" in pending_logs[0]


async def test_asking_pointless_закрывает_с_основанием_бессмысленно(script):
    """Вердикт «спрашивать бессмысленно» закрывает шаг с отдельным основанием."""
    from graph.log_fmt import format_check_done

    client = FakeChecker(
        [CheckerVerdict(reply_usable=True, step_closed=False, asking_pointless=True)]
    )
    progress = ScriptProgress(
        status={"name": "pending"},
        attempts={"name": 2},
        taken_turn={"name": 1},
        in_work=["name"],
    )
    updated, closures = await run_checker(
        script=script,
        progress=progress,
        messages=[HumanMessage(content="не хочу отвечать, хватит")],
        profile={},
        turn=5,
        client=client,
    )
    assert updated.status["name"] == "closed"
    assert ("name", "бессмысленно") in closures
    assert ("name", "диалог") not in closures
    assert "бессмысленно" in format_check_done(closures)
