"""Тесты планировщика: порядок шагов выводится, а не задан списком."""

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
    should_skip,
    steps_to_skip,
)


def test_первым_идёт_город(script):
    step = pick_step(script, status={}, profile={})
    assert step is not None
    assert step.id == "city"


def test_шаг_филиала_не_открывается_без_города(script):
    step = script.step("branch")
    assert is_available(step, status={}, profile={}) is False
    assert is_available(step, status={}, profile={"city": "perm"}) is True


def test_причина_блокировки_называет_поле_и_владельца(script):
    reasons = blocked_by(script, script.step("branch"), status={}, profile={})
    assert reasons == ["нужно поле city (заполняет city)"]


def test_имя_можно_не_спрашивать_если_оно_уже_известно(script):
    profile = {"city": "perm", "caller_name": "Мария"}
    assert should_skip(script.step("name"), profile) is True
    assert "name" in steps_to_skip(script, status={}, profile=profile)


def test_порядок_подстраивается_под_уже_известное(script):
    """Калининград: имя не спросили вовсе, разговор ушёл к опыту."""
    profile = {"city": "kaliningrad", "caller_name": "Иван"}
    status = {"city": "done"}
    step = pick_step(script, status=status, profile=profile)
    assert step is not None
    assert step.id == "experience"


def test_презентация_ждёт_коробку(script):
    profile = {"city": "perm"}
    status = {"city": "done", "name": "done", "experience": "done"}
    step = pick_step(script, status=status, profile=profile)
    assert step is not None
    assert step.id == "transmission"

    status["transmission"] = "done"
    profile["transmission"] = "механика"
    step = pick_step(script, status=status, profile=profile)
    assert step is not None
    assert step.id == "presentation"


def test_презентация_пропускается_готовому_клиенту(script):
    profile = {"city": "perm", "urgency": "готов записаться"}
    assert should_skip(script.step("presentation"), profile) is True


def test_отказ_закрывает_шаг_и_пускает_дальше(script):
    profile = {"city": "perm"}
    status = {"city": "done", "name": "refused"}
    step = pick_step(script, status=status, profile=profile)
    assert step is not None
    assert step.id == "experience"


def test_счётчик_попыток_предохраняет_от_зацикливания(script):
    step = script.step("name")
    assert step.max_attempts == 2
    assert exhausted(step, {}) is False
    assert exhausted(step, {"name": 1}) is False
    assert exhausted(step, {"name": 2}) is True
    assert next_attempt({"name": 1}, "name") == 2


def test_возврат_после_справки_уводит_обратно_на_шаг(script):
    profile = {"city": "perm"}
    status = {"city": "done"}
    step = pick_step(script, status=status, profile=profile, resume="experience")
    assert step is not None
    assert step.id == "experience"


def test_возврат_игнорируется_если_шаг_уже_закрыт(script):
    profile = {"city": "perm", "caller_name": "Мария"}
    status = {"city": "done", "experience": "done"}
    step = pick_step(script, status=status, profile=profile, resume="experience")
    assert step is not None
    assert step.id != "experience"


def test_закрытый_и_пустой_профиль(script):
    assert is_closed("done") and is_closed("refused") and is_closed("skipped")
    assert is_closed("open") is False and is_closed(None) is False
    assert profile_has({"city": " "}, "city") is False
    assert profile_has({"city": "perm"}, "city") is True


def test_текст_шага_ветвится_по_значению(script):
    step = script.step("theory_format")
    очно = render_step_text(step, {"theory_format": "очно"})
    дистанционно = render_step_text(step, {"theory_format": "дистанционно"})
    комбинированно = render_step_text(step, {"theory_format": "комбинированно"})
    по_умолчанию = render_step_text(step, {})

    assert "небольшие группы" in очно
    assert "приложении" in дистанционно
    assert "частый" in комбинированно
    assert "на стоимость это не влияет" in по_умолчанию
    assert len({очно, дистанционно, комбинированно, по_умолчанию}) == 4


def test_когда_закрывать_нечего_шага_нет(script):
    status = {step_id: "done" for step_id in script.step_order}
    assert pick_step(script, status=status, profile={}) is None


def test_после_города_следующим_идёт_имя(script):
    """Закрыли город — peek отдаёт следующий открытый шаг."""
    nxt = peek_next_step(
        script,
        current=script.step("city"),
        status={},
        profile={},
    )
    assert nxt is not None
    assert nxt.id == "name"


def test_после_последнего_шага_следующего_нет(script):
    status = {step_id: "done" for step_id in script.step_order if step_id != "closing"}
    nxt = peek_next_step(
        script,
        current=script.step("closing"),
        status=status,
        profile={"city": "perm"},
    )
    assert nxt is None


def test_peek_учитывает_пропуск_по_признаку(script):
    """Имя уже известно — после города сразу опыт, не имя."""
    nxt = peek_next_step(
        script,
        current=script.step("city"),
        status={},
        profile={"caller_name": "Мария"},
    )
    assert nxt is not None
    assert nxt.id == "experience"


def test_peek_пропускает_презентацию_готовому_клиенту(script):
    status = {
        "city": "done",
        "name": "done",
        "experience": "done",
    }
    nxt = peek_next_step(
        script,
        current=script.step("transmission"),
        status=status,
        profile={"city": "perm", "urgency": "готов записаться"},
    )
    assert nxt is not None
    assert nxt.id != "presentation"
    assert nxt.id == "theory_format"
