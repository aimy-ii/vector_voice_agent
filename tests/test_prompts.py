"""Тесты сборки промпта и подготовки фактов справочника."""

from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage

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
    aside_block,
    build_turn_messages,
    facts_block,
    fill_facts,
    persona_block,
    profile_block,
    step_block,
)


def test_подстановка_фактов_в_текст():
    assert (
        fill_facts("Стоимость: {price_line}", {"price_line": "от 43900"}) == "Стоимость: от 43900"
    )


def test_неизвестный_плейсхолдер_не_роняет_ход():
    """В звонке пустое место лучше исключения."""
    assert fill_facts("Адрес: {branch_address}", {}) == "Адрес:"


def test_персона_и_правила_попадают_в_промпт(script):
    block = persona_block(script)
    assert "Дарья" in block
    assert "Вектор" in block
    assert "ты и есть менеджер" in block


def test_профиль_разделяет_роли(script):
    block = profile_block(script, {"caller_name": "Ольга"})
    assert "Ольга" in block
    assert "звонящий" in block
    assert "будущий курсант" in block
    assert "student_name" in block


def test_известное_не_попадает_в_список_вопросов(script):
    block = profile_block(script, {"transmission": "механика"})
    строки = block.splitlines()
    известное = [s for s in строки if "transmission" in s and "механика" in s]
    assert известное
    assert "переспрашивать известное — ошибка" in block


def test_дословный_шаг_запрещает_пересказ(script):
    block = step_block(script.step("presentation"), {}, {})
    assert "дословно" in block
    assert "Не повторяй" in block


def test_обычный_шаг_разрешает_свои_слова(script):
    block = step_block(script.step("city"), {}, {})
    assert "своими словами" in block


def test_альтернативный_вопрос_передаётся_как_приём(script):
    block = step_block(script.step("transmission"), {}, {})
    assert "механика, автомат" in block


def test_цена_подставляется_в_дословный_шаг(script):
    block = step_block(script.step("price"), {}, {"price_line": "Стоимость — от 43900 рублей."})
    assert "от 43900 рублей" in block
    assert "{price_line}" not in block


def test_пустые_факты_не_засоряют_промпт():
    assert facts_block({}) == ""
    assert facts_block({"city": None, "branches": []}) == ""
    assert "Пермь" in facts_block({"city": {"name": "Пермь"}})


def test_перечень_справок_уходит_модели(script):
    block = aside_block(script, done=["medcheck"])
    assert "medcheck" in block
    assert "уже отвечали" in block
    assert "think" in block


def test_запрос_к_модели_содержит_системный_блок_и_хвост(script):
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
    assert len(messages) == 9
    assert messages[-1].content == "реплика 11"


def test_пустая_история_не_ломает_запрос(script):
    messages = build_turn_messages(
        script=script, step=None, profile={}, facts={}, history=[], asides_done=[]
    )
    assert len(messages) == 2
    assert "Все шаги скрипта закрыты" in messages[0].content


def test_города_отдаются_со_названиями():
    """По одному слагу город не опознать: kyrgan это Курган."""
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
    assert len(facts["branches"]) == 2
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
