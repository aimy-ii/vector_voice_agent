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
    # Формат теории ждёт закрытия terms; вопросов больше нет — terms идёт сам.
    step = pick_step(script, status=status, profile=profile)
    assert step is not None
    assert step.id == "terms"

    status["terms"] = "closed"
    step = pick_step(script, status=status, profile=profile)
    assert step is not None
    assert step.id == "theory_format"


def test_theory_format_ждёт_terms(script):
    """theory_format недоступен, пока terms не закрыт, и открывается после."""
    profile = {
        "city": "perm",
        "caller_name": "Мария",
        "student_is_caller": "да",
        "experience": "впервые",
        "transmission": "механика",
    }
    status = {
        "name": "closed",
        "city": "closed",
        "who_studies": "closed",
        "experience": "closed",
        "transmission": "closed",
    }
    assert is_available(script.step("theory_format"), status=status, profile=profile) is False
    status["terms"] = "closed"
    assert is_available(script.step("theory_format"), status=status, profile=profile) is True


def test_информирующий_без_повода_не_в_шапке(script):
    profile = {
        "city": "perm",
        "caller_name": "Мария",
        "student_is_caller": "да",
        "experience": "впервые",
        "transmission": "механика",
        "theory_format": "очно",
    }
    status = {
        "name": "closed",
        "city": "closed",
        "who_studies": "closed",
        "experience": "closed",
        "transmission": "closed",
        "terms": "closed",
        "theory_format": "closed",
        "included": "closed",
        "practice": "closed",
    }
    # Есть свежий вопрос branch — информ price без повода не в шапке.
    head = script_head(script, status=status, attempts={}, profile=profile)
    assert head[0].id == "branch"
    assert all(s.id != "price" for s in head)


def test_информирующий_по_вопросу_клиента(script):
    from script.planner import client_asks_inform

    assert client_asks_inform("А что входит в обучение?")
    assert not client_asks_inform("Для себя")

    profile = {
        "city": "perm",
        "caller_name": "Мария",
        "student_is_caller": "да",
        "experience": "впервые",
        "transmission": "механика",
    }
    status = {
        "name": "closed",
        "city": "closed",
        "who_studies": "closed",
        "experience": "closed",
        "transmission": "closed",
    }
    head = script_head(script, status=status, attempts={}, profile=profile, inform_reason=True)
    assert head[0].id == "terms"


def test_информирующий_после_проверочного_ответа(script):
    from script.planner import answered_inform_check

    profile = {
        "city": "perm",
        "caller_name": "Мария",
        "student_is_caller": "да",
        "experience": "впервые",
        "transmission": "механика",
        "theory_format": "очно",
    }
    status = {
        "name": "closed",
        "city": "closed",
        "who_studies": "closed",
        "experience": "closed",
        "transmission": "closed",
        "terms": "closed",
        "theory_format": "closed",
        "included": "closed",
        "practice": "closed",
    }
    assert answered_inform_check(script, status=status, pending_step="practice")
    # Оба свежие — без повода только вопрос; информ ждёт.
    head_no = script_head(script, status=status, attempts={}, profile=profile)
    assert head_no[0].id == "branch"
    assert all(s.id != "price" for s in head_no)
    # С поводом информ всё ещё после вопроса по приоритету, но когда вопрос
    # уже в шапке как заданный — price входит.
    head = script_head(
        script,
        status=status,
        attempts={"branch": 1},
        profile=profile,
        inform_reason=True,
    )
    assert any(s.id == "price" for s in head)


def test_информирующий_когда_вопросов_не_осталось(script):
    profile = {"city": "perm", "caller_name": "Мария", "payment_pref": "целиком"}
    status = {
        sid: "closed"
        for sid in script.step_order
        if sid not in {"tax_deduction", "closing", "messenger"}
    }
    head = script_head(script, status=status, attempts={}, profile=profile)
    assert head
    assert head[0].id == "tax_deduction"


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


def test_exhausted_только_по_порогу_из_аргумента(script):
    """Порог живёт в настройках; поля max_attempts у шага больше нет."""
    from script.models import Step

    step = script.step("city")
    assert "max_attempts" not in Step.model_fields
    assert exhausted(step, {"city": 2}, limit=2) is True
    assert exhausted(step, {"city": 2}, limit=3) is False
    assert exhausted(step, {"city": 3}, limit=3) is True


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
