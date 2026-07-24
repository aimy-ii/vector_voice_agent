"""Тесты синхронного чекера."""

from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage

from graph.checker import (
    CheckerVerdict,
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

    async def judge(self, *, history_slice, client_reply, step, step_text):
        self.calls.append(
            {
                "history_slice": history_slice,
                "client_reply": client_reply,
                "step_id": step.id,
                "step_text": step_text,
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


async def test_для_счётчика_ноль_история_с_начала(script):
    steps = [script.step("city")]
    progress = ScriptProgress(attempts={"city": 0})
    messages = [
        HumanMessage(content="я из Перми"),
        AIMessage(content="как зовут?"),
        HumanMessage(content="Игорь"),
    ]
    sliced = history_slice_for(messages, steps=steps, progress=progress, turn=2)
    assert sliced[0].content == "я из Перми" or any("Перми" in str(m.content) for m in sliced[:-1])


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
            CheckerVerdict(reply_usable=True, step_closed=True),
            CheckerVerdict(reply_usable=True, step_closed=False),
            CheckerVerdict(reply_usable=True, step_closed=True),
        ]
    )
    progress = ScriptProgress(
        status={"name": "pending", "city": "pending"},
        attempts={"name": 1, "city": 1},
    )
    # caller_name уже в профиле — name закрывает код до модели.
    # city без поля в профиле: модель сказала «закрыт» — не верим, цикл рвётся.
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


async def test_порог_исчерпан_закрывает_без_модели(script):
    client = FakeChecker([CheckerVerdict(reply_usable=True, step_closed=False)])
    progress = ScriptProgress(status={"name": "pending"}, attempts={"name": 5})
    updated, _ = await run_checker(
        script=script,
        progress=progress,
        messages=[HumanMessage(content="потом скажу")],
        profile={},
        turn=6,
        client=client,
        attempt_limit=5,
    )
    assert updated.status["name"] == "closed"
    assert not any(c["step_id"] == "name" for c in client.calls)


async def test_счётчик_ноль_модель_не_вызывается(script):
    client = FakeChecker([CheckerVerdict(reply_usable=True, step_closed=True)])
    progress = ScriptProgress(status={}, attempts={})
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


async def test_question_с_пустыми_fills_не_закрывается(script):
    """Модель сказала «закрыт», поля профиля пустые — не верим."""
    client = FakeChecker([CheckerVerdict(reply_usable=True, step_closed=True)])
    progress = ScriptProgress(
        status={"theory_format": "pending"},
        attempts={"theory_format": 1},
    )
    # Предшественники закрыты, чтобы theory_format был доступен.
    for step_id in ("name", "city", "who_studies", "experience", "transmission", "terms"):
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
        turn=5,
        client=client,
    )
    assert updated.status.get("theory_format") != "closed"
    assert any(c["step_id"] == "theory_format" for c in client.calls)


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
