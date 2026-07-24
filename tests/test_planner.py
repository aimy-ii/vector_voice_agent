"""Тесты планировщика: шапка, статусы pending/closed, пропуск без статуса."""

from __future__ import annotations

from script.planner import (
    blocked_by,
    exhausted,
    is_available,
    is_closed,
    next_attempt,
    peek_next_step,
    pick_step,
    profile_has,
    render_step_text,
    script_head,
    should_skip,
    steps_to_skip,
)


def test_первым_идёт_имя(script):
    step = pick_step(script, status={}, profile={})
    assert step is not None
    assert step.id == "name"


def test_шаг_филиала_не_открывается_без_города(script):
    step = script.step("branch")
    assert is_available(step, status={}, profile={}) is False
    status = {
        sid: "closed"
        for sid in (
            "name",
            "who_studies",
            "experience",
            "transmission",
            "terms",
            "theory_format",
            "included",
            "practice",
        )
    }
    assert is_available(step, status=status, profile={"city": "perm"}) is True


def test_причина_блокировки_называет_поле_и_владельца(script):
    reasons = blocked_by(script, script.step("branch"), status={}, profile={})
    assert "нужно поле city (заполняет city)" in reasons


def test_имя_можно_не_спрашивать_если_оно_уже_известно(script):
    profile = {"caller_name": "Мария"}
    assert should_skip(script.step("name"), profile) is True
    assert "name" in steps_to_skip(script, status={}, profile=profile)
    step = pick_step(script, status={}, profile=profile)
    assert step is not None
    assert step.id == "city"


def test_порядок_подстраивается_под_уже_известное(script):
    """Имя уже известно — разговор уходит к городу, не к имени."""
    profile = {"caller_name": "Иван"}
    step = pick_step(script, status={}, profile=profile)
    assert step is not None
    assert step.id == "city"


def test_срок_ждёт_коробку(script):
    profile = {"city": "perm", "caller_name": "Мария"}
    status = {
        "name": "closed",
        "city": "closed",
        "who_studies": "closed",
        "experience": "closed",
    }
    step = pick_step(script, status=status, profile=profile)
    assert step is not None
    assert step.id == "transmission"

    status["transmission"] = "closed"
    profile["transmission"] = "механика"
    step = pick_step(script, status=status, profile=profile)
    assert step is not None
    assert step.id == "terms"


def test_пропуск_не_ставит_статус_а_отсеивает_из_шапки(script):
    profile = {"caller_name": "Мария"}
    head = script_head(script, status={}, attempts={}, profile=profile)
    assert all(s.id != "name" for s in head)
    assert "name" not in {s.id for s in head}


def test_закрытый_шаг_пускает_дальше(script):
    profile = {"city": "perm"}
    status = {"name": "closed", "city": "closed"}
    step = pick_step(script, status=status, profile=profile)
    assert step is not None
    assert step.id == "who_studies"


def test_счётчик_ноль_значит_не_задавали(script):
    step = script.step("name")
    assert exhausted(step, {}, limit=2) is False
    assert exhausted(step, {"name": 1}, limit=2) is False
    assert exhausted(step, {"name": 2}, limit=2) is True
    assert next_attempt({"name": 1}, "name") == 2


def test_статусов_ровно_два():
    assert is_closed("closed")
    assert is_closed("done")  # наследие v1 в слепке
    assert is_closed("pending") is False
    assert is_closed("open") is False
    assert is_closed(None) is False
    assert profile_has({"city": " "}, "city") is False
    assert profile_has({"city": "perm"}, "city") is True


def test_шапка_берёт_заданные_и_один_новый(script):
    status = {"name": "pending"}
    attempts = {"name": 1}
    profile: dict[str, str] = {}
    head = script_head(script, status=status, attempts=attempts, profile=profile)
    ids = [s.id for s in head]
    assert ids[0] == "name"
    assert ids.count("city") == 1
    assert ids == ["name", "city"]


def test_закрытые_и_отсеянные_не_в_шапке(script):
    status = {"name": "closed", "city": "closed"}
    profile = {"city": "perm", "caller_name": "Мария", "student_is_caller": "да"}
    head = script_head(script, status=status, attempts={}, profile=profile)
    assert all(s.id not in {"name", "city", "who_studies"} for s in head)


def test_текст_шага_ветвится_по_значению(script):
    step = script.step("theory_format")
    очно = render_step_text(step, {"theory_format": "очно"})
    дистанционно = render_step_text(step, {"theory_format": "дистанционно"})
    комбинированно = render_step_text(step, {"theory_format": "комбинированно"})
    по_умолчанию = render_step_text(step, {})

    assert "небольших групп" in очно
    assert "приложении" in дистанционно
    assert "популярный" in комбинированно
    assert "на стоимость это не влияет" in по_умолчанию
    assert len({очно, дистанционно, комбинированно, по_умолчанию}) == 4


def test_когда_закрывать_нечего_шага_нет(script):
    status = {step_id: "closed" for step_id in script.step_order}
    assert pick_step(script, status=status, profile={}) is None


def test_после_имени_следующим_идёт_город(script):
    nxt = peek_next_step(
        script,
        current=script.step("name"),
        status={},
        profile={},
    )
    assert nxt is not None
    assert nxt.id == "city"


def test_после_последнего_шага_следующего_нет(script):
    status = {step_id: "closed" for step_id in script.step_order if step_id != "messenger"}
    nxt = peek_next_step(
        script,
        current=script.step("messenger"),
        status=status,
        profile={"city": "perm"},
    )
    assert nxt is None


def test_peek_учитывает_пропуск_по_признаку(script):
    """Имя уже известно — после закрытия имени peek отдаёт город."""
    nxt = peek_next_step(
        script,
        current=script.step("name"),
        status={},
        profile={"caller_name": "Мария"},
    )
    assert nxt is not None
    assert nxt.id == "city"


def test_возврат_после_справки_уводит_обратно_на_шаг(script):
    profile = {"caller_name": "Мария", "city": "perm"}
    status = {"name": "closed", "city": "closed"}
    step = pick_step(script, status=status, profile=profile, resume="experience")
    assert step is not None
    assert step.id == "experience"
