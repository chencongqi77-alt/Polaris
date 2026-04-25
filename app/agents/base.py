from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List

from app.agents.io_models import AgentInput, AgentOutput
from app.graph.state import MacpState
from app.interfaces.llm import ChatMessage, LLMClient, MockLLMClient
from app.interfaces.memory import MemoryStore, MockMemoryStore
from app.interfaces.mcp import MCPClient, MockMCPClient
from app.prompts.registry import PromptRegistry


class AgentExecutionError(RuntimeError):
    pass


class BaseAgent(ABC):
    """
    Foundation class for all MACP agents.
    Provides:
    1) LLM invocation
    2) MCP tool invocation
    3) Prompt template rendering
    4) Error handling and retry
    5) Unified input/output envelope
    6) Agent-scoped state management
    7) Context awareness from global state
    """

    def __init__(
        self,
        name: str,
        config: Dict[str, Any] | None = None,
        llm_client: LLMClient | None = None,
        mcp_client: MCPClient | None = None,
        memory_store: MemoryStore | None = None,
        prompt_registry: PromptRegistry | None = None,
    ) -> None:
        self.name = name
        self.config = config or {}
        self.llm_client = llm_client or MockLLMClient()
        self.mcp_client = mcp_client or MockMCPClient()
        self.memory_store = memory_store or MockMemoryStore()
        self.prompt_registry = prompt_registry or PromptRegistry()


    @abstractmethod
    def run(self, state: MacpState) -> MacpState:
        raise NotImplementedError

    def execute(self, state: MacpState) -> MacpState:
        """Retry wrapper around run() with unified error recording."""
        envelope = AgentInput(agent_name=self.name, state=state)
        self._mark_trace(envelope.state)
        retries = int(self.config.get("max_retries", 2))
        errors: List[str] = []
        for attempt in range(retries + 1):
            try:
                next_state = self.run(envelope.state)
                return AgentOutput(agent_name=self.name, state=next_state, success=True).state
            except Exception as exc:  # noqa: BLE001
                errors.append(f"attempt={attempt} error={exc}")
                if attempt >= retries:
                    self._record_errors(envelope.state, errors)
                    raise AgentExecutionError(
                        f"{self.name} failed after {retries + 1} attempts"
                    ) from exc
        return envelope.state

    def _mark_trace(self, state: MacpState) -> None:
        trace = state.metadata.setdefault("trace", [])
        trace.append(self.name)

    def build_context(self, state: MacpState) -> Dict[str, Any]:
        return {
            "request_id": state.request_id,
            "user_request": state.user_request,
            "constraints": state.constraints,
            "subtasks": state.subtasks,
            "reflection_count": state.reflection_count,
            "previous_evaluations": [item.model_dump() for item in state.evaluations],
        }

    def render_prompt(self, name: str, variables: Dict[str, Any]) -> str:
        return self.prompt_registry.render(name, variables)

    def call_llm(self, prompt: str, temperature: float = 0.2) -> str:
        messages = [
            ChatMessage(role="system", content=f"You are MACP agent {self.name}."),
            ChatMessage(role="user", content=prompt),
        ]
        return self.llm_client.generate(messages=messages, temperature=temperature)

    def call_mcp_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        return self.mcp_client.call_tool(tool_name=tool_name, arguments=arguments)

    def get_agent_state(self, state: MacpState) -> Dict[str, Any]:
        agent_state = state.metadata.setdefault("agent_state", {})
        return agent_state.setdefault(self.name, {})

    def retrieve_memories(self, state: MacpState, top_k: int = 3) -> List[Dict[str, Any]]:
        query = f"{state.user_request} {' '.join(state.constraints)}"
        memories = self.memory_store.search(query=query, top_k=top_k, scope="global")
        state.metadata.setdefault("memory_hits", {})[self.name] = memories
        return memories

    def _record_errors(self, state: MacpState, errors: List[str]) -> None:
        history = state.metadata.setdefault("errors", [])
        history.extend(errors)
