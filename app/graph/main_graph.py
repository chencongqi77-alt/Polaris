from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from langgraph.graph import END, StateGraph
from langgraph.types import interrupt

from app.agents.eva import EvaAgent
from app.agents.sg import SolutionGeneratorAgent as SGAgent
from app.agents.tm import TaskManagerAgent as TmAgent
from app.graph.checkpoint import CheckpointManager
from app.graph.routes import (
    route_after_eva,
    route_after_human_review,
    route_after_sg,
    route_after_tm,
)
from app.graph.state import MacpState
from app.interfaces.mcp import MCPClient
from app.memory import MemoryService, create_memory_store

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
log = logging.getLogger("macp.graph")

# ---------------------------------------------------------------------------
# Node functions – one per graph step
# ---------------------------------------------------------------------------


def node_tm(state: MacpState) -> MacpState:
    """Task Manager node: decompose user request into subtasks + constraints."""
    log.info("[TM] Starting task management for request=%s", state.request_id)
    try:
        mcp = MCPClient()
        agent = TmAgent(mcp_client=mcp)
        agent.run(state)
        mcp.close()
    except Exception as exc:
        log.error("[TM] Error: %s", exc)
        state.errors.append(f"TM error: {exc}")
    return state


def node_human_review(state: MacpState) -> MacpState:
    """LangGraph interrupt-based human review."""
    log.info("[HumanReview] Interrupting for human review")
    state.status = "awaiting_human_review"

    payload = {
        "request_id": state.request_id,
        "message": (
            "The TM agent has finished parsing. Please review subtasks and constraints.\n"
            "You may revise both before the SG agents start working."
        ),
        "subtasks": state.subtasks,
        "constraints": state.constraints,
    }
    if state.feedback:
        payload["feedback"] = state.feedback

    resume = interrupt(payload)

    if not isinstance(resume, dict):
        state.approved = False
        state.status = "completed"
        return state

    approved = resume.get("approved", False)
    if approved:
        state.approved = True
        if "subtasks" in resume:
            state.subtasks = [str(s).strip() for s in resume["subtasks"] if str(s).strip()]
        if "constraints" in resume:
            state.constraints = [str(c).strip() for c in resume["constraints"] if str(c).strip()]
    else:
        state.approved = False
        if "subtasks" in resume:
            state.subtasks = [str(s).strip() for s in resume["subtasks"] if str(s).strip()]
        if "constraints" in resume:
            state.constraints = [str(c).strip() for c in resume["constraints"] if str(c).strip()]
        if "feedback" in resume:
            state.feedback = resume["feedback"]
        if "notes" in resume:
            notes = str(resume["notes"]).strip()
            if notes:
                state.constraints.append(f"[Human notes] {notes}")
        state.reflection_count += 1

    state.status = "human_reviewed"
    return state


def node_sg(state: MacpState) -> MacpState:
    """Solution Generator node: generate candidate solutions."""
    log.info("[SG] Starting candidate generation for subtask")
    try:
        mcp = MCPClient()
        agent = SGAgent(mcp_client=mcp)
        agent.run(state)
        mcp.close()
    except Exception as exc:
        log.error("[SG] Error: %s", exc)
        state.errors.append(f"SG error: {exc}")
    return state


def node_eva(state: MacpState) -> MacpState:
    """Evaluator node: score and select candidates."""
    log.info("[Eva] Starting evaluation for %d candidates", len(state.candidates))
    try:
        mcp = MCPClient()
        agent = EvaAgent(mcp_client=mcp)
        agent.run(state)
        mcp.close()
    except Exception as exc:
        log.error("[Eva] Error: %s", exc)
        state.errors.append(f"Eva error: {exc}")
    return state


def node_memory(state: MacpState) -> MacpState:
    """Persist approved artifacts to memory store."""
    log.info("[Memory] Running persistence")
    try:
        store = create_memory_store(mode="qdrant")
        if store is not None:
            service = MemoryService(store)
            service.retrieve_for_agent(state, agent_name="system")
            persisted = service.persist_approved_artifact(state)
            log.info("[Memory] Persisted: %s", persisted)
            store.close()
        else:
            log.info("[Memory] Store unavailable, skipping")
    except Exception as exc:
        log.warning("[Memory] Error (continuing): %s", exc)
    return state


def node_end(state: MacpState) -> MacpState:
    """Terminal node: mark state as completed."""
    state.status = "completed"
    return state


# ---------------------------------------------------------------------------
# Graph assembly
# ---------------------------------------------------------------------------

def build_graph(checkpoint_path: Optional[str] = "app_data/checkpoints/macp.sqlite"):
    builder = StateGraph(MacpState)

    builder.add_node("TM", node_tm)
    builder.add_node("HumanReview", node_human_review)
    builder.add_node("SG", node_sg)
    builder.add_node("Eva", node_eva)
    builder.add_node("Memory", node_memory)
    builder.add_node("End", node_end)

    builder.set_entry_point("TM")
    builder.add_conditional_edges("TM", route_after_tm, {
        "HumanReview": "HumanReview",
        "End": "End",
    })
    builder.add_conditional_edges("HumanReview", route_after_human_review, {
        "TM": "TM",
        "SG": "SG",
        "End": "End",
    })
    builder.add_conditional_edges("SG", route_after_sg, {
        "Eva": "Eva",
        "End": "End",
    })
    builder.add_conditional_edges("Eva", route_after_eva, {
        "SG": "SG",
        "Memory": "Memory",
    })
    builder.add_edge("Memory", "End")
    builder.add_edge("End", END)

    memory_saver = None
    if checkpoint_path:
        try:
            mgr = CheckpointManager(checkpoint_path)
            memory_saver = mgr.saver
            log.info("Checkpoint enabled: %s", checkpoint_path)
        except Exception as exc:
            log.warning("Checkpoint init failed (%s), running without persistence", exc)

    compile_kwargs: Dict[str, Any] = {}
    if memory_saver is not None:
        compile_kwargs["checkpointer"] = memory_saver

    return builder.compile(**compile_kwargs)


# ---------------------------------------------------------------------------
# Runner helpers
# ---------------------------------------------------------------------------

class MacpGraphRunner:
    def __init__(
        self,
        memory_mode: str = "qdrant",
        mcp_mode: str = "direct",
        checkpoint_path: Optional[str] = "app_data/checkpoints/macp.sqlite",
    ) -> None:
        self.memory_mode = memory_mode
        self.mcp_mode = mcp_mode
        self.graph = build_graph(checkpoint_path)

    def run(
        self,
        state: MacpState,
        thread_id: str = "local-thread",
        auto_approve_human_review: bool = True,
    ) -> MacpState:
        config = {"configurable": {"thread_id": thread_id}}
        result = self.graph.invoke(state, config=config)
        return result if isinstance(result, MacpState) else MacpState(**result)

    def run_until_human_review(
        self,
        state: MacpState,
        thread_id: str = "local-thread",
    ) -> Dict[str, Any]:
        config = {"configurable": {"thread_id": thread_id}}
        try:
            self.graph.invoke(state, config=config)
        except Exception as exc:
            if "Interrupt" in type(exc).__name__ or "interrupt" in str(exc).lower():
                log.info("Graph interrupted for human review")
            else:
                raise
        snapshot = self.graph.get_state(config)
        return snapshot.values if hasattr(snapshot, "values") else {}

    def resume_after_human_review(
        self,
        review: Dict[str, Any],
        thread_id: str = "local-thread",
    ) -> MacpState:
        config = {"configurable": {"thread_id": thread_id}}
        result = self.graph.invoke(review, config=config)
        return result if isinstance(result, MacpState) else MacpState(**result)

    def close(self) -> None:
        pass