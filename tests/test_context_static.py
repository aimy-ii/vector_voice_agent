def test_возраст_подаётся_отдельно_от_документов() -> None:
    """В общем перечне бот выбирал между условиями наугад.

    Рядом стояли «возраст от 18 лет (для переобучения с В на С)» и
    «возраст от 16 лет (для начала обучения)». На «сыну семнадцать» бот
    ответил «учиться можно с семнадцати» — подстроил цифру под
    собеседника вместо того, чтобы взять её из данных.
    """
    from graph.context import format_city_static

    text = format_city_static(
        city_slug="ekaterinburg",
        city_name="Екатеринбург",
        city_meta={
            "documents": {
                "items": [
                    {"name": "паспорт", "stage": "для старта"},
                    {"name": "возраст от 16 лет", "stage": "для начала обучения"},
                    {"name": "возраст 18 лет", "stage": "для сдачи экзамена в ГИБДД"},
                ]
            },
        },
    )

    documents = next(line for line in text.splitlines() if line.startswith("Документы:"))
    ages = next(line for line in text.splitlines() if line.startswith("Возраст:"))
    assert "возраст" not in documents.lower()
    assert "паспорт" in documents
    assert "возраст от 16 лет (для начала обучения)" in ages
    assert "возраст 18 лет (для сдачи экзамена в ГИБДД)" in ages
