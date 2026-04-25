from __future__ import annotations

from app.agents.base import BaseAgent
from app.graph.state import MacpState
from app.interfaces.llm import LLMClient
from app.interfaces.memory import MemoryStore
from app.interfaces.mcp import MCPClient
from app.prompts.registry import PromptRegistry


class TaskManagerAgent(BaseAgent):
    """
    TM: convert user request into explicit subtasks.
    """

    def __init__(
        self,
        llm_client: LLMClient | None = None,
        mcp_client: MCPClient | None = None,
        memory_store: MemoryStore | None = None,
    ) -> None:
        prompts = PromptRegistry(
            {
                "tm_plan": (
                    "User request: ${user_request}\n"
                    "Constraints: ${constraints}\n"
                    "Generate 3 concise subtasks for SG and Eva."
                )
            }
        )
        super().__init__(
            name="TM",
            config={"max_retries": 2},
            llm_client=llm_client,
            mcp_client=mcp_client,
            memory_store=memory_store,
            prompt_registry=prompts,
        )

    def run(self, state: MacpState) -> MacpState:
        memories = self.retrieve_memories(state, top_k=2)
        context = self.build_context(state)
        prompt = self.render_prompt(
            "tm_plan",
            {
                "user_request": context["user_request"],
                "constraints": ", ".join(context["constraints"]) or "none",
            },
        )
        if memories:
            prompt += f"\nReference memories: {memories}"
        draft = self.call_llm(prompt=prompt)

        if not state.subtasks:
            state.subtasks = [line.strip("- ").strip() for line in draft.splitlines() if line.strip()][:3]
            if not state.subtasks:
                state.subtasks = [
                    "Extract teaching goals from user request",
                    "Generate multiple candidate solutions",
                    "Evaluate candidates with rubric",
                ]
        if not state.constraints:
            state.constraints = ["clear output", "testable output"]

        tm_state = self.get_agent_state(state)
        tm_state["last_plan"] = state.subtasks
        return state
