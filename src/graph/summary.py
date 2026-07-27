"""Саммари звонка: плоская структура из состояния, без модели.

Город хранится и слагом, и читаемым названием. Флаг «в нашу пользу» здесь
не считается — он на разборе после звонка.
"""

from __future__ import annotations

from typing import Any, Mapping

from script.build import CompiledScript
from script.models import Step
from script.planner import is_closed


def build_summary(
    *,
    script: CompiledScript,
    step_status: Mapping[str, str],
    profile: Mapping[str, str],
    city_slug: str | None = None,
    city_name: str | None = None,
    branch_slug: str | None = None,
) -> dict[str, Any]:
    """Собирает саммари: шаг → значение из диалога.

    Args:
        script: скомпилированный скрипт.
        step_status: статусы шагов.
        profile: собранный профиль.
        city_slug: слаг города.
        city_name: читаемое название города.
        branch_slug: слаг филиала.

    Returns:
        Плоский словарь саммари; доступен в любой момент разговора.
    """
    steps: dict[str, Any] = {}
    for step_id in script.step_order:
        if not is_closed(step_status.get(step_id)):
            continue
        step = script.step(step_id)
        if isinstance(step, Step) and step.fills:
            values = {
                key: profile[key]
                for key in step.fills
                if key in profile and str(profile[key]).strip()
            }
            steps[step_id] = values if values else {"closed": True}
        else:
            steps[step_id] = {"closed": True}

    city: dict[str, str] = {}
    if city_slug:
        city["slug"] = city_slug
    if city_name:
        city["name"] = city_name
    elif profile.get("city") and not city_slug:
        city["name"] = profile["city"]

    return {
        "steps": steps,
        "city": city,
        "branch_slug": branch_slug,
        "profile": dict(profile),
        "outcome": profile.get("outcome"),
    }
