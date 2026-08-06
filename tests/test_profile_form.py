"""Тесты явной формы профиля."""

from __future__ import annotations

from graph.profile_form import (
    PROFILE_FORM,
    REWRITABLE_MARK,
    field_pairs,
    form_keys,
    rewritable_keys,
)

#: Ожидаемые ключи формы в порядке объявления.
_EXPECTED_KEYS = [
    "caller_name",
    "city",
    "student_is_caller",
    "student_name",
    "student_age",
    "location_hint",
    "experience",
    "transmission",
    "theory_format",
    "branch",
    "discount_category",
    "tariff_choice",
    "payment_pref",
    "appointment_time",
    "outcome",
    "second_category",
    "messenger",
    "urgency",
]


def test_ключи_уникальны_и_полей_восемнадцать():
    keys = [field.key for field in PROFILE_FORM]
    assert len(keys) == 18
    assert len(set(keys)) == 18


def test_перечень_ключей_точно_совпадает():
    assert [field.key for field in PROFILE_FORM] == _EXPECTED_KEYS


def test_rewritable_keys_только_location_hint():
    assert rewritable_keys() == frozenset({"location_hint"})


def test_field_pairs_помечает_уточняемое():
    pairs = dict(field_pairs())
    assert REWRITABLE_MARK in pairs["location_hint"]
    assert REWRITABLE_MARK not in pairs["caller_name"]


def test_form_keys_совпадает_с_формой():
    assert form_keys() == frozenset(field.key for field in PROFILE_FORM)
