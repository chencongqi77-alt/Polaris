from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from app.agents.io_models import AgentInput, AgentOutput
from app.graph.state import MacpState
from app.interfaces.llm import ChatMessage, LLMClient
from app.interfaces.memory import MemoryStore
from app.interfaces.mcp import MCPClient, McpClient
from app.prompts.registry import PromptRegistry


class AgentExecutionError(RuntimeError):
    pass


class BaseAgent(ABC):
    """
    Foundation class for all MACP agents.
    Provides:
    1) LLM invocation via self.llm (LLMAdapter) and self.call_llm()
    2) MCP tool invocation via self.mcp and self.call_mcp_tool()
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
        mcp: Any | None = None,
        memory_store: MemoryStore | None = None,
        prompt_registry: PromptRegistry | None = None,
    ) -> None:
        self.name = name
        self.config = config or {}
        self.prompt_registry = prompt_registry or PromptRegistry()

        # LLM adapter – try to create one if not provided
        if llm_client is not None:
            self.llm = llm_client
            self.llm_client = llm_client
        else:
            try:
                from app.interfaces.llm import LLMAdapter
                self.llm = LLMAdapter()
                self.llm_client = self.llm
            except Exception:
                self.llm = None  # type: ignore[assignment]
                self.llm_client = None

        # MCP client – agents pass this as 'mcp' positional/kwarg
        if mcp is not None:
            self.mcp = mcp
            self.mcp_client = mcp
        elif mcp_client is not None:
            self.mcp = mcp_client
            self.mcp_client = mcp_client
        else:
            try:
                self.mcp = McpClient()
                self.mcp_client = self.mcp
            except Exception:
                self.mcp = None  # type: ignore[assignment]
                self.mcp_client = None

        # Memory is optional – gracefully degrade
        self.memory_store = memory_store

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
        trace = state.extra_metadata.setdefault("trace", [])
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
        """Convenience: send a single prompt via the LLM adapter."""
        messages = [
            ChatMessage(role="system", content=f"You are MACP agent {self.name}."),
            ChatMessage(role="user", content=prompt),
        ]
        if self.llm_client is not None:
            return self.llm_client.generate(messages=messages, temperature=temperature)
        # Fallback: use self.llm.chat if available (agents use this pattern)
        if hasattr(self.llm, "chat"):
            resp = self.llm.chat(user_msg=prompt, temperature=temperature)  # type: ignore[union-attr]
            return resp.content if hasattr(resp, "content") else str(resp)
        raise RuntimeError("No LLM client available")

    def call_mcp_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Convenience: call an MCP tool via the standard interface."""
        if self.mcp_client is not None:
            result = self.mcp_client.call_tool(tool_name=tool_name, arguments=arguments)
            # Normalize MCPResult to dict
            if hasattr(result, "model_dump"):
                return result.model_dump()
            if isinstance(result, dict):
                return result
            return {"success": True, "content": str(result)}
        # Fallback: use self.mcp.call_tool_directly if available
        if hasattr(self.mcp, "call_tool_directly"):
            raw = self.mcp.call_tool_directly(tool_name, arguments)  # type: ignore[union-attr]
            if hasattr(raw, "model_dump"):
                return raw.model_dump()
            if isinstance(raw, dict):
                return raw
            return {"success": True, "content": str(raw)}
        raise RuntimeError("No MCP client available")

    def _call_mcp(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        """Convenience: call an MCP tool and return the content string.

        Many agents expect a plain string result from MCP calls.
        This method normalizes the result into a string.
        """
        result = self.call_mcp_tool(tool_name, arguments)
        if isinstance(result, dict):
            # Try common keys
            for key in ("content", "result", "output", "text"):
                if key in result:
                    val = result[key]
                    return val if isinstance(val, str) else str(val)
            if result.get("error"):
                return f"Error: {result['error']}"
            return str(result)
        return str(result)

    def _call_mcp_raw(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Convenience: call an MCP tool and return the raw dict result.

        Same as call_mcp_tool but with explicit dict return type.
        """
        return self.call_mcp_tool(tool_name, arguments)

    def get_agent_state(self, state: MacpState) -> Dict[str, Any]:
        agent_state = state.extra_metadata.setdefault("agent_state", {})
        return agent_state.setdefault(self.name, {})

    def retrieve_memories(self, state: MacpState, top_k: int = 3) -> List[Dict[str, Any]]:
        if self.memory_store is None:
            return []
        query = f"{state.user_request} {' '.join(state.constraints)}"
        memories = self.memory_store.search(query=query, top_k=top_k, scope="global")
        state.extra_metadata.setdefault("memory_hits", {})[self.name] = memories
        return memories

    def _record_errors(self, state: MacpState, errors: List[str]) -> None:
        history = state.extra_metadata.setdefault("errors", [])
        history.extend(errors)