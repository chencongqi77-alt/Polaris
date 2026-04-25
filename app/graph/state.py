from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field

# SG候选方案
class Candidate(BaseModel):
    """Single SG candidate artifact."""

    id: str
    content: str
    modality: Literal["text", "code", "image"] = "text"
    metadata: Dict[str, Any] = Field(default_factory=dict)

# Eva反馈
class EvaluationResult(BaseModel):
    """Eva score result for one candidate."""

    candidate_id: str
    score: float
    passed: bool
    feedback: str

# 系统的“数据字典”
class MacpState(BaseModel):
    """
    Global shared state contract for all graph nodes.
    Person A owns this schema and versioning.
    """

    request_id: str = "local-dev"
    user_request: str
    constraints: List[str] = Field(default_factory=list)
    subtasks: List[str] = Field(default_factory=list)

    candidates: List[Candidate] = Field(default_factory=list)
    evaluations: List[EvaluationResult] = Field(default_factory=list)
    selected_candidate_id: Optional[str] = None

    approved: bool = False
    reflection_count: int = 0
    max_reflections: int = 3

    checkpoint_id: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    human_approved: bool = False
