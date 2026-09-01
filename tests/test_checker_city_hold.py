def test_шаг_филиала_не_закрывается_без_адреса() -> None:
    """Обещание подобрать филиал шаг не закрывает.

    На разборе звонка человек назвал ориентир, распознавание переврало
    название станции, филиалы не подобрались. Бот сказал «сейчас подберу
    ближайший филиал», судья счёл вопрос отвеченным и закрыл шаг за один
    ход. Адрес так и не прозвучал: обещал и не назвал.
    """
    from graph.context import ConversationContext, branch_picked

    assert not branch_picked(ConversationContext())
    assert branch_picked(ConversationContext(branch_slug="sankt-peterburg_kolomyazhskiy"))
    assert branch_picked(
        ConversationContext(branch_cards=[{"slug": "a", "address": "улица Садовая, дом 38"}])
    )
