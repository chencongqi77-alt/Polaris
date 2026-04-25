from __future__ import annotations

from pydantic import BaseModel

from app.graph.state import MacpState


class AgentInput(BaseModel):
    agent_name: str
    state: MacpState


class AgentOutput(BaseModel):
    agent_name: str
    state: MacpState
    success: bool = True
