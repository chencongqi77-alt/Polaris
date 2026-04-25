from __future__ import annotations

from typing import Any, Dict

from app.graph.state import MacpState
from app.interfaces.memory import MemoryStore


class MemoryService:
    def __init__(self, store: MemoryStore) -> None:
        self.store = store

    def retrieve_for_agent(self, state: MacpState, agent_name: str, top_k: int = 3) -> None:
        query = f"{state.user_request} | {';'.join(state.constraints)}"
        hits = self.store.search(query=query, top_k=top_k, scope="global")
        state.metadata.setdefault("memory_hits", {})[agent_name] = hits

    def persist_approved_artifact(self, state: MacpState) -> None:
        if not state.approved or not state.selected_candidate_id:
            return
        selected = next((c for c in state.candidates if c.id == state.selected_candidate_id), None)
        if selected is None:
            return
        score = max((ev.score for ev in state.evaluations if ev.candidate_id == selected.id), default=0.0)
        self.store.upsert(
            {
                "request_id": state.request_id,
                "scope": "global",
                "agent": "system",
                "memory_type": "artifact",
                "quality_score": score,
                "content": selected.content,
                "candidate_id": selected.id,
            }
        )
