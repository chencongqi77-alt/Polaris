from __future__ import annotations

from typing import Protocol

from app.graph.state import MacpState


class AgentNode(Protocol):
    """
    Shared node contract.
    Any concrete TM/SG/Eva class can be plugged into graph execution.
    """

    name: str

    def run(self, state: MacpState) -> MacpState:
        """Consume and mutate state, then return it."""
        ...
