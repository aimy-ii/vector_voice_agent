"""Тесты имени, контекста и саммари."""

from __future__ import annotations

from graph.context import (
    DYN_MISSING,
    DYN_NONE,
    DYN_READY,
    DYN_SEARCHING,
    DYN_WORKING,
    ConversationContext,
    _discount_lines,
    format_city_static,
    merge_static,
)
from graph.names import given_name
from graph.summary import build_summary


def test_статусы_динамики_константы_и_дефолт():
    assert DYN_NONE == "не требуется"
    assert DYN_READY == "готово"
    assert DYN_WORKING == "в работе"
    assert DYN_SEARCHING == "в поиске"
    assert DYN_MISSING == "не нашлось"
    ctx = ConversationContext()
    assert ctx.dynamic_status == DYN_NONE
    assert ctx.situation_slug is None
    assert ctx.filler_spoken is False
    assert ctx.dynamic_reply == ""
    assert ctx.last_agent_reply == ""
    assert ctx.dynamic_turn == 0
    assert ctx.last_reply_hash == ""
    assert ctx.dynamic_reply_hash == ""
    assert ctx.pending_fields == []
    assert ctx.render() == ""


def test_имя_три_случая():
    assert given_name("Андрей Андреевич") == "Андрей"
    assert given_name("Андрей Петров") == "Андрей Петров"
    assert given_name("Мария Ивановна") == "Мария"


def test_контекст_статика_один_раз_цена_фразой(fake_kb):
    city = {
        "slug": "perm",
        "name": "Пермь",
        "categories": [{"code": "B", "duration": "2,5 месяца"}],
        "vehicles": {"manual": ["Solaris"], "automatic": []},
        "theory_formats": ["очно"],
        "documents": ["паспорт"],
        "payment": {"installment": True},
        "messengers": ["Telegram"],
    }
    text = format_city_static(
        city_slug="perm",
        city_name="Пермь",
        city_meta=city,
        price_line="Стоимость обучения — от 43900 рублей.",
    )
    assert "Пермь" in text and "perm" in text
    assert "от 43900" in text
    assert "amount" not in text
    assert "Список филиалов и адреса в статику не входят" in text
    assert "есть рассрочка" in text.lower() or "рассроч" in text.lower()

    ctx = ConversationContext()
    ctx = merge_static(ctx, city_slug="perm", city_name="Пермь", city_meta=city, price_line="фраза")
    again = merge_static(ctx, city_slug="spb", city_name="Питер", city_meta=city)
    assert again.city_slug == "perm"
    assert again.city_name == "Пермь"
    assert again.static_text.count("Статика разговора") == 1
    assert again.static_text == ctx.static_text


def test_статика_города_в_системном_сообщении_один_раз(script):
    """Повторная подшивка merge_static не дублирует блок города в промпте."""
    from graph.prompts import build_turn_messages

    city_meta = {
        "categories": [{"code": "B", "duration": "2.5 мес"}],
        "vehicles": {"manual": ["Лада"], "automatic": []},
    }
    ctx = ConversationContext()
    ctx = merge_static(
        ctx,
        city_slug="perm",
        city_name="Пермь",
        city_meta=city_meta,
        price_line="Стоимость обучения — от 43900 рублей.",
    )
    ctx = merge_static(
        ctx,
        city_slug="perm",
        city_name="Пермь",
        city_meta=city_meta,
        price_line="Стоимость обучения — от 43900 рублей.",
    )
    assert ctx.render().count("Статика разговора") == 1
    assert ctx.render().count("Город: Пермь") == 1

    step = script.step("city")
    messages = build_turn_messages(
        script=script,
        history=[],
        profile={"caller_name": "Андрей", "city": "Пермь"},
        facts={},
        steps=[step],
        context_text=ctx.render(),
        asides_done=[],
    )
    system = str(messages[0].content)
    assert system.count("Статика разговора") == 1
    assert system.count("Город: Пермь") == 1


def test_format_city_static_без_рекламы_словарей_и_ключей():
    city = {
        "slug": "perm",
        "name": "Пермь",
        "categories": [{"code": "B", "duration": "2,5 месяца"}],
        "vehicles": {"manual": ["Solaris"], "automatic": []},
        "theory_formats": [
            {
                "name": "онлайн",
                "description": "Изучай теорию в удобное время! Отменяй занятие без штрафа!",
            },
            {"name": "очно", "description": "Приходи в класс!"},
        ],
        "documents": [
            {"name": "паспорт", "stage": "для старта"},
            {"name": "СНИЛС", "stage": "для договора"},
        ],
        "payment": {
            "installment_no_overpay": True,
            "matcap": True,
            "installment": False,
        },
        "messengers": ["Telegram"],
    }
    text = format_city_static(
        city_slug="perm",
        city_name="Пермь",
        city_meta=city,
        price_line="Стоимость обучения — от 43900 рублей.",
    )
    assert "онлайн" in text
    assert "очно" in text
    assert "Изучай" not in text
    assert "Отменяй" not in text
    assert "Приходи" not in text
    assert "паспорт" in text and "СНИЛС" in text
    assert "{'name'" not in text
    assert "stage" not in text
    assert "installment_no_overpay" not in text
    assert "без переплаты" in text.lower()
    assert "от 43900" in text


def test_format_city_static_число_филиалов_и_без_адресов():
    """При branches_count — служебная пометка; списка адресов нет."""
    with_count = format_city_static(
        city_slug="perm",
        city_name="Пермь",
        city_meta={"branches_count": 4, "categories": []},
    )
    assert "Филиалов в городе: 4" in with_count
    assert "служебно" in with_count
    assert "Чернышевского" not in with_count
    assert "ул." not in with_count
    assert "подбирает контекстер" in with_count

    without = format_city_static(
        city_slug="perm",
        city_name="Пермь",
        city_meta={"categories": []},
    )
    assert "Филиалов в городе" not in without
    assert "подбирает контекстер" in without


def _krasnoyarsk_card(**extra: object) -> dict:
    """Карточка Красноярска для тестов скидок в статике."""
    card: dict = {
        "name": "Красноярск",
        "branches_count": 15,
        "categories": [{"code": "B", "duration": "2 месяца", "start_frequency": "каждые 15 дней"}],
        "vehicles": {
            "manual": ["Lada Granta"],
            "automatic": ["Kia Rio"],
            "fleet_age": "не старше 2023 г",
        },
        "theory_formats": ["очно", "дистанционно"],
        "documents": [{"name": "Паспорт", "stage": "при договоре"}],
        "payment": {"installment_no_overpay": True},
        "messengers": ["Max"],
        "call_hours": "С 7:30 до 23:00",
    }
    card.update(extra)
    return card


def test_format_city_static_со_скидками():
    """Скидки печатаются одной строкой через «; » в исходном порядке."""
    discounts = [
        "студентам и школьникам скидка до 1000 рублей",
        "молодым мамам скидка до 1000 рублей",
        "именинникам скидка 1000 рублей на любой пакет обучения",
    ]
    text = format_city_static(
        city_slug="krasnoyarsk",
        city_name="Красноярск",
        city_meta=_krasnoyarsk_card(discounts=discounts),
        price_line="от 39 900",
    )
    expected = "Скидки и акции: " + "; ".join(discounts) + "."
    assert expected in text
    pos = text.index("Скидки и акции:")
    assert text.index(discounts[0], pos) < text.index(discounts[1], pos)
    assert text.index(discounts[1], pos) < text.index(discounts[2], pos)


def test_format_city_static_без_ключа_discounts():
    """Без ключа discounts — явная строка об отсутствии скидок."""
    text = format_city_static(
        city_slug="krasnoyarsk",
        city_name="Красноярск",
        city_meta=_krasnoyarsk_card(),
        price_line="от 39 900",
    )
    assert "Скидки и акции: сейчас нет, действующих скидок в городе не заявлено." in text


def test_format_city_static_с_пустым_списком_discounts():
    """Пустой список discounts — та же строка об отсутствии."""
    text = format_city_static(
        city_slug="krasnoyarsk",
        city_name="Красноярск",
        city_meta=_krasnoyarsk_card(discounts=[]),
        price_line="от 39 900",
    )
    assert "Скидки и акции: сейчас нет, действующих скидок в городе не заявлено." in text


def test_format_city_static_скидки_фильтруют_мусор():
    """Из списка скидок в строку уходят только непустые строки."""
    text = format_city_static(
        city_slug="krasnoyarsk",
        city_name="Красноярск",
        city_meta=_krasnoyarsk_card(discounts=["текст", None, 123, "  "]),
        price_line="от 39 900",
    )
    assert "Скидки и акции: текст." in text
    assert "None" not in text
    assert "123" not in text


def test_format_city_static_скидка_одной_строкой():
    """Одна строка discounts тоже печатается."""
    text = format_city_static(
        city_slug="krasnoyarsk",
        city_name="Красноярск",
        city_meta=_krasnoyarsk_card(discounts="студентам скидка 1000 рублей"),
        price_line="от 39 900",
    )
    assert "Скидки и акции: студентам скидка 1000 рублей." in text


def test_format_city_static_регресс_без_скидок_прежние_строки():
    """Без скидок прежние строки статики на месте, плюс строка об отсутствии."""
    text = format_city_static(
        city_slug="krasnoyarsk",
        city_name="Красноярск",
        city_meta=_krasnoyarsk_card(),
        price_line="от 39 900",
    )
    assert "Категории: B — 2 месяца, набор: каждые 15 дней." in text
    assert (
        "Автопарк: механика: Lada Granta; автомат: Kia Rio; возраст парка: не старше 2023 г."
    ) in text
    assert "Форматы теории: очно, дистанционно." in text
    # Этап рядом с названием: в справочнике он есть у каждого документа,
    # а в контекст уходили одни названия — и «права категории В, возраст
    # от 18 лет» читались как требование ко всем, а не к переобучению.
    assert "Документы: Паспорт (при договоре)." in text
    assert "Мессенджеры: Max." in text
    assert "Оплата: рассрочка без переплаты." in text
    assert "Часы колл-центра: С 7:30 до 23:00." in text
    assert "Цена (готовая фраза, произносить только так): от 39 900" in text
    assert "Скидки и акции: сейчас нет, действующих скидок в городе не заявлено." in text


def test_discount_lines_чистая_функция():
    """_discount_lines нормализует список, строку, None и мусор."""
    assert _discount_lines(["а", "б"]) == ["а", "б"]
    assert _discount_lines([]) == []
    assert _discount_lines(None) == []
    assert _discount_lines("одна фраза") == ["одна фраза"]
    assert _discount_lines(["текст", None, 123, "  "]) == ["текст"]


def test_мета_филиала_после_выбора():
    ctx = ConversationContext(city_slug="perm", city_name="Пермь", static_text="город")
    assert ctx.branch_slug is None
    ctx = merge_static(
        ctx,
        branch_slug="perm_chernyshevskogo",
        branch_meta={"address": "ул. Чернышевского, 28", "place_type": "учебный офис"},
    )
    assert ctx.branch_slug == "perm_chernyshevskogo"
    assert "Чернышевского" in ctx.static_text
    assert ctx.frozen is True


def test_саммари_город_слагом_и_названием(script):
    summary = build_summary(
        script=script,
        step_status={"name": "closed", "city": "closed"},
        profile={"caller_name": "Андрей", "city": "Пермь"},
        city_slug="perm",
        city_name="Пермь",
        branch_slug=None,
    )
    assert summary["city"] == {"slug": "perm", "name": "Пермь"}
    assert summary["steps"]["name"]["caller_name"] == "Андрей"
    assert "в нашу пользу" not in summary
