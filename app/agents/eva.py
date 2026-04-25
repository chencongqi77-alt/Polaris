from __future__ import annotations

from app.agents.base import BaseAgent
from app.graph.state import EvaluationResult, MacpState
from app.interfaces.llm import LLMClient
from app.interfaces.memory import MemoryStore
from app.interfaces.mcp import MCPClient
from app.prompts.registry import PromptRegistry


class EvaluatorAgent(BaseAgent):
    """
    Eva: score all candidates and choose a winner.
    """

    def __init__(
        self,
        pass_score: float = 0.75,
        llm_client: LLMClient | None = None,
        mcp_client: MCPClient | None = None,
        memory_store: MemoryStore | None = None,
    ) -> None:
        prompts = PromptRegistry(
            {
                "eva_review": (
                    "Evaluate this candidate with rubric.\n"
                    "Constraints: ${constraints}\n"
                    "Candidate: ${candidate}\n"
                    "Return concise critique."
                )
            }
        )
        super().__init__(
            name="Eva",
            config={"pass_score": pass_score, "max_retries": 1},
            llm_client=llm_client,
            mcp_client=mcp_client,
            memory_store=memory_store,
            prompt_registry=prompts,
        )

    def run(self, state: MacpState) -> MacpState:
        if not state.candidates:
            state.approved = False
            return state

        memories = self.retrieve_memories(state, top_k=2)
        pass_score = float(self.config["pass_score"])
        state.evaluations = []
        constraints = "; ".join(state.constraints) or "none"

        for i, candidate in enumerate(state.candidates):
            score = 0.70 + (i * 0.10)       # 当前的score函数完全是假的，需要你手动修改
            passed = score >= pass_score
            feedback = self.call_llm(
                prompt=self.render_prompt(
                    "eva_review",
                    {"constraints": constraints, "candidate": candidate.content[:400]},
                ),
                temperature=0.1,
            )
            if memories:
                feedback = f"{feedback}\nReference memories: {memories}"
            state.evaluations.append(
                EvaluationResult(
                    candidate_id=candidate.id,
                    score=round(score, 2),
                    passed=passed,
                    feedback=feedback,
                )
            )

        best = max(state.evaluations, key=lambda item: item.score)
        state.selected_candidate_id = best.candidate_id
        state.approved = best.passed
        eva_state = self.get_agent_state(state)
        eva_state["last_selected"] = state.selected_candidate_id
        return state
