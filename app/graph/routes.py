from __future__ import annotations

from typing import Any, Dict, Literal


def route_after_tm(state: Any) -> Literal["HumanReview", "End"]:
    """Decide whether to go to human review or end."""
    # If state is a dict (LangGraph serialization), extract fields
    if isinstance(state, dict):
        subtasks = state.get("subtasks", [])
        errors = state.get("errors", [])
    else:
        subtasks = getattr(state, "subtasks", [])
        errors = getattr(state, "errors", [])

    if errors:
        return "End"
    if subtasks:
        return "HumanReview"
    return "End"


def route_after_human_review(state: Any) -> Literal["TM", "SG", "End"]:
    """After human review, decide next step."""
    if isinstance(state, dict):
        approved = state.get("approved", False)
        status = state.get("status", "")
        reflection_count = state.get("reflection_count", 0)
        max_reflections = state.get("max_reflections", 3)
    else:
        approved = getattr(state, "approved", False)
        status = getattr(state, "status", "")
        reflection_count = getattr(state, "reflection_count", 0)
        max_reflections = getattr(state, "max_reflections", 3)

    if status == "completed":
        return "End"

    # If not approved and under reflection limit, go back to TM to re-plan
    if not approved:
        if reflection_count < max_reflections:
            return "TM"
        return "End"

    # Approved - proceed to SG
    return "SG"


def route_after_sg(state: Any) -> Literal["Eva", "End"]:
    """After SG, decide whether to evaluate or end."""
    if isinstance(state, dict):
        candidates = state.get("candidates", [])
        errors = state.get("errors", [])
    else:
        candidates = getattr(state, "candidates", [])
        errors = getattr(state, "errors", [])

    if errors and not candidates:
        return "End"
    return "Eva"


def route_after_eva(state: Any) -> Literal["SG", "Memory"]:
    """After Eva, decide whether to loop back to SG or proceed to Memory.

    This is the reflection loop: if evaluation didn't pass and we haven't
    exceeded max reflections, loop back to SG for re-generation.
    """
    if isinstance(state, dict):
        approved = state.get("approved", False)
        reflection_count = state.get("reflection_count", 0)
        max_reflections = state.get("max_reflections", 3)
    else:
        approved = getattr(state, "approved", False)
        reflection_count = getattr(state, "reflection_count", 0)
        max_reflections = getattr(state, "max_reflections", 3)

    if approved:
        return "Memory"

    if reflection_count < max_reflections:
        return "SG"

    # Max reflections reached - force pass to Memory with best candidate
    return "Memory"


def route_after_eva_dict(state: Dict[str, Any]) -> Literal["approved", "reflect", "stop"]:
    """Legacy route helper for dict-based state."""
    from app.graph.state import MacpState
    typed_state = MacpState.model_validate(state)
    decision = route_after_eva(typed_state)
    state.update(typed_state.model_dump())
    return decision