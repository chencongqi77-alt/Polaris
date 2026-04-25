from __future__ import annotations

from typing import Any, Dict
from typing import Literal

from app.graph.state import MacpState


def route_after_eva(state: MacpState) -> Literal["approved", "reflect", "stop"]:
    """
    Decide whether graph finishes or loops back to SG.
    """

    if state.approved:
        return "approved"

    if state.reflection_count >= state.max_reflections:
        return "stop"

    state.reflection_count += 1
    return "reflect"


def route_after_eva_dict(state: Dict[str, Any]) -> Literal["approved", "reflect", "stop"]:
    typed_state = MacpState.model_validate(state)
    decision = route_after_eva(typed_state)
    state.update(typed_state.model_dump())
    return decision
