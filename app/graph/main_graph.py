from __future__ import annotations

from typing import Any, Dict

from app.agents.eva import EvaluatorAgent
from app.agents.sg import SolutionGeneratorAgent
from app.agents.tm import TaskManagerAgent
from app.graph.checkpoint import CheckpointStore
from app.graph.routes import route_after_eva_dict
from app.graph.state import MacpState
from app.interfaces.memory import MockMemoryStore, MemoryStore
from app.interfaces.mcp import MCPClient, create_mcp_client
from app.interfaces.llm import LLMClient, OpenAILLMClient
from app.memory import create_memory_store
from app.memory.service import MemoryService

try:
    from langgraph.graph import END, START, StateGraph
    from langgraph.types import Command, interrupt
except ImportError:  # pragma: no cover
    END = "__end__"
    START = "__start__"
    StateGraph = None  # type: ignore[assignment]
    Command = None  # type: ignore[assignment]
    interrupt = None  # type: ignore[assignment]


MacpStateDict = Dict[str, Any]


def _node_tm(state: MacpStateDict, tm: TaskManagerAgent) -> MacpStateDict:
    typed = MacpState.model_validate(state)
    updated = tm.execute(typed)
    return updated.model_dump()


def _node_sg(state: MacpStateDict, sg: SolutionGeneratorAgent) -> MacpStateDict:
    typed = MacpState.model_validate(state)
    updated = sg.execute(typed)
    return updated.model_dump()


def _node_eva(state: MacpStateDict, eva: EvaluatorAgent) -> MacpStateDict:
    typed = MacpState.model_validate(state)
    updated = eva.execute(typed)
    return updated.model_dump()


def _node_human_review(state: MacpStateDict) -> MacpStateDict:
    if interrupt is None:
        raise ImportError("langgraph interrupt API is unavailable.")

    typed = MacpState.model_validate(state)
    if typed.human_approved:
        return typed.model_dump()

    review_payload = {
        "request_id": typed.request_id,
        "message": "Review TM plan, then approve or revise.",
        "subtasks": typed.subtasks,
        "constraints": typed.constraints,
    }
    review = interrupt(review_payload)

    if isinstance(review, dict):
        typed.human_approved = bool(review.get("approved", False))
        if isinstance(review.get("subtasks"), list):
            typed.subtasks = [str(item) for item in review["subtasks"]]
        if isinstance(review.get("constraints"), list):
            typed.constraints = [str(item) for item in review["constraints"]]
        typed.metadata["human_feedback"] = review
    else:
        typed.human_approved = bool(review)
        typed.metadata["human_feedback"] = {"approved": typed.human_approved}

    return typed.model_dump()


def _node_memory(state: MacpStateDict, memory_service: MemoryService) -> MacpStateDict:
    typed = MacpState.model_validate(state)
    persisted = memory_service.persist_approved_artifact(typed)
    typed.metadata["memory_persisted"] = persisted
    return typed.model_dump()


class MacpGraphRunner:
    """LangGraph orchestrator with HITL interrupt and checkpoint."""

    def __init__(
        self,
        tm: TaskManagerAgent | None = None,
        sg: SolutionGeneratorAgent | None = None,
        eva: EvaluatorAgent | None = None,
        memory_store: MemoryStore | None = None,
        memory_mode: str = "mock",
        mcp_client: MCPClient | None = None,
        mcp_mode: str = "direct",
        llm_client: LLMClient | None = None,
        checkpoint_path: str | None = "app_data/checkpoints/macp.sqlite",
    ) -> None:
        """Initialize the MACP graph runner.

        Args:
            tm: Task Manager agent (optional, will be created if not provided).
            sg: Solution Generator agent (optional, will be created if not provided).
            eva: Evaluator agent (optional, will be created if not provided).
            memory_store: Memory store for persistence (optional).
            memory_mode: Memory store mode if memory_store not provided ("mock", "qdrant").
            mcp_client: MCP client for tool calls (optional).
            mcp_mode: MCP client mode if mcp_client not provided ("mock", "direct", "stdio", "http").
            llm_client: LLM client for AI calls (optional, defaults to OpenAILLMClient from env).
            checkpoint_path: Path for checkpoint storage.
        """
        shared_memory = memory_store or create_memory_store(mode=memory_mode)
        shared_mcp = mcp_client or create_mcp_client(mode=mcp_mode)
        shared_llm = llm_client or OpenAILLMClient()
        
        # Initialize agents with shared LLM and MCP client
        self.tm = tm or TaskManagerAgent(
            llm_client=shared_llm,
            memory_store=shared_memory,
            mcp_client=shared_mcp,
        )
        self.sg = sg or SolutionGeneratorAgent(
            llm_client=shared_llm,
            memory_store=shared_memory,
            mcp_client=shared_mcp,
        )
        self.eva = eva or EvaluatorAgent(
            llm_client=shared_llm,
            memory_store=shared_memory,
            mcp_client=shared_mcp,
        )
        self.memory_service = MemoryService(shared_memory)
        self.mcp_client = shared_mcp
        self.checkpoint_store = CheckpointStore(checkpoint_path)
        self.graph = self._build_graph()

    def _build_graph(self):  # type: ignore[no-untyped-def]
        if StateGraph is None:
            raise ImportError(
                "langgraph is not installed. Install it with `pip install langgraph`."
            )

        builder = StateGraph(MacpStateDict)
        builder.add_node("tm", lambda state: _node_tm(state, self.tm))
        builder.add_node("human_review", _node_human_review)
        builder.add_node("sg", lambda state: _node_sg(state, self.sg))
        builder.add_node("eva", lambda state: _node_eva(state, self.eva))
        builder.add_node("memory", lambda state: _node_memory(state, self.memory_service))

        builder.add_edge(START, "tm")
        builder.add_edge("tm", "human_review")
        builder.add_edge("human_review", "sg")
        builder.add_edge("sg", "eva")

        # 关键：条件边 - 实现反射循环
        builder.add_conditional_edges(
            "eva",
            route_after_eva_dict,       # 路由函数
            {"approved": "memory", "reflect": "sg", "stop": END},
        )
        builder.add_edge("memory", END)
        return builder.compile(checkpointer=self.checkpoint_store.create())

    def run(
        self,
        state: MacpState,
        thread_id: str = "local-thread",
        auto_approve_human_review: bool = True,
    ) -> MacpState:
        config = {"configurable": {"thread_id": thread_id}}
        result = self.graph.invoke(state.model_dump(), config=config)
        if "__interrupt__" in result:
            if not auto_approve_human_review:
                raise RuntimeError("Human review required. Use resume_after_human_review().")
            result = self.graph.invoke(
                Command(resume={"approved": True}),
                config=config,
            )
        return MacpState.model_validate(result)

    def run_until_human_review(
        self, state: MacpState, thread_id: str = "local-thread"
    ) -> Dict[str, Any]:
        config = {"configurable": {"thread_id": thread_id}}
        return self.graph.invoke(state.model_dump(), config=config)

    def resume_after_human_review(
        self, review: Dict[str, Any], thread_id: str = "local-thread"
    ) -> MacpState:
        if Command is None:
            raise ImportError("langgraph Command API is unavailable.")
        config = {"configurable": {"thread_id": thread_id}}
        result = self.graph.invoke(Command(resume=review), config=config)
        return MacpState.model_validate(result)

    def close(self) -> None:
        self.checkpoint_store.close()
        disconnect = getattr(self.mcp_client, "disconnect", None)
        if callable(disconnect):
            try:
                disconnect()
            except Exception:  # noqa: BLE001
                # Best-effort shutdown; runner close shouldn't crash callers.
                pass
