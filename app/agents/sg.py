from __future__ import annotations

from app.agents.base import BaseAgent
from app.graph.state import Candidate, MacpState
from app.interfaces.llm import LLMClient
from app.interfaces.memory import MemoryStore
from app.interfaces.mcp import MCPClient
from app.prompts.registry import PromptRegistry


class SolutionGeneratorAgent(BaseAgent):
    """
    SG: produce K candidates for Eva.
    """

    def __init__(
        self,
        k: int = 3,
        llm_client: LLMClient | None = None,
        mcp_client: MCPClient | None = None,
        memory_store: MemoryStore | None = None,
    ) -> None:
        prompts = PromptRegistry(
            {
                "sg_generate": (
                    "Task: ${user_request}\n"
                    "Subtasks: ${subtasks}\n"
                    "Constraints: ${constraints}\n"
                    "Generate one concise candidate draft."
                )
            }
        )
        super().__init__(
            name="SG",
            config={"k": k, "max_retries": 2},
            llm_client=llm_client,
            mcp_client=mcp_client,
            memory_store=memory_store,
            prompt_registry=prompts,
        )

    def run(self, state: MacpState) -> MacpState:
        k = int(self.config["k"])
        memories = self.retrieve_memories(state, top_k=3)
        context = self.build_context(state)
        rendered_subtasks = "; ".join(context["subtasks"]) or "none"
        rendered_constraints = "; ".join(context["constraints"]) or "none"

        candidates = []
        for idx in range(k):
            prompt = self.render_prompt(
                "sg_generate",
                {
                    "user_request": context["user_request"],
                    "subtasks": rendered_subtasks,
                    "constraints": rendered_constraints,
                },
            )
            if memories:
                prompt += f"\nUseful prior artifacts: {memories}"
            content = self.call_llm(prompt=prompt, temperature=0.6)
            artifact = self.call_mcp_tool(
                "artifact.prepare",
                {"candidate_id": f"cand-{idx + 1}", "preview": content[:120]},
            )
            candidates.append(
                Candidate(
                    id=f"cand-{idx + 1}",
                    content=content,
                    modality="text",
                    metadata={"artifact": artifact},
                )
            )

        state.candidates = candidates
        sg_state = self.get_agent_state(state)
        sg_state["last_candidate_count"] = len(state.candidates)
        return state
