"""Шаблоны ситуативных заглушек вне скрипта.

Шаблоны с подстановкой предмета и универсальный ``default``. Вызовов модели
в рантайме ноль: выбор случайный, мимо уже звучавших в звонке.
"""

from __future__ import annotations

import json
import random
from functools import lru_cache
from pathlib import Path
from typing import Mapping, Sequence

#: Имя ключа с универсальными фразами без предмета.
DEFAULT_SLUG = "default"
#: Имя ключа с шаблонами, куда подставляется предмет.
TEMPLATES_KEY = "templates"

_DATA_PATH = Path(__file__).resolve().parent / "data" / "situations.json"


@lru_cache(maxsize=1)
def load_situations() -> Mapping[str, tuple[str, ...]]:
    """Загружает шаблоны и фразы по умолчанию из JSON один раз.

    Returns:
        Словарь с обязательными ключами ``templates`` и ``default``.

    Raises:
        RuntimeError: если в файле нет обязательного ключа или он пуст.
    """
    raw = json.loads(_DATA_PATH.read_text(encoding="utf-8"))
    result: dict[str, tuple[str, ...]] = {}
    for key in (TEMPLATES_KEY, DEFAULT_SLUG):
        phrases = raw.get(key)
        if not isinstance(phrases, list):
            raise RuntimeError(f"В {_DATA_PATH} нет обязательного ключа «{key}»")
        cleaned = tuple(p.strip() for p in phrases if isinstance(p, str) and p.strip())
        if not cleaned:
            raise RuntimeError(f"В {_DATA_PATH} ключ «{key}» пуст")
        result[key] = cleaned
    return result


def pick_filler(subject: str | None, *, spoken: Sequence[str] = ()) -> str:
    """Готовая фраза-заглушка по предмету вопроса.

    Непустой предмет → случайный шаблон из ``templates`` с подстановкой
    ``{subject}``. Пустой → случайная фраза из ``default``. Молчания не бывает.
    Выбор мимо уже звучавших в этом звонке. Вызовов модели ноль.

    Args:
        subject: предмет вопроса одним-двумя словами или пусто.
        spoken: фразы, уже звучавшие в этом звонке.

    Returns:
        Непустая готовая фраза.
    """
    catalog = load_situations()
    text = (subject or "").strip()
    if text:
        pool = [tpl.format(subject=text) for tpl in catalog[TEMPLATES_KEY]]
    else:
        pool = list(catalog[DEFAULT_SLUG])
    used = set(spoken or [])
    fresh = [p for p in pool if p not in used]
    choice = random.choice(fresh or pool)
    if not choice:
        return catalog[DEFAULT_SLUG][0]
    return choice
