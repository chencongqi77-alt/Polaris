from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel, Field, ValidationError, conlist, confloat

from app.graph.state import MacpState


class AgentInput(BaseModel):
    agent_name: str
    state: MacpState


class AgentOutput(BaseModel):
    agent_name: str
    state: MacpState
    success: bool = True


class TMPlanPayload(BaseModel):
    subtasks: conlist(str, min_length=3, max_length=3)  # type: ignore[valid-type]
    constraints: list[str] = Field(default_factory=list)


class SGCandidatePayload(BaseModel):
    content: str = Field(min_length=10)
    rationale: str = ""
    checklist: list[str] = Field(default_factory=list)


class EvaReviewPayload(BaseModel):
    score: confloat(ge=0.0, le=1.0)  # type: ignore[valid-type]
    feedback: str = Field(min_length=5)
    strengths: list[str] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)


def parse_json_payload(raw: str, schema: type[BaseModel]) -> BaseModel:
    """
    Strict JSON parser for agent LLM outputs.
    Raises ValueError when payload cannot be extracted or validated.
    """
    text = raw.strip()
    if not text:
        raise ValueError("Empty LLM output")
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end <= start:
            raise ValueError("No JSON object in LLM output") from None
        data = json.loads(text[start : end + 1])
    try:
        return schema.model_validate(data)
    except ValidationError as exc:
        raise ValueError(f"JSON payload validation failed: {exc}") from exc


def render_schema_hint(schema: type[BaseModel]) -> str:
    schema_doc: dict[str, Any] = schema.model_json_schema()
    return json.dumps(schema_doc, ensure_ascii=True)
