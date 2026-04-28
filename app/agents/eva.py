from __future__ import annotations

import re

from app.agents.base import BaseAgent
from app.agents.io_models import EvaReviewPayload, parse_json_payload, render_schema_hint
from app.graph.state import Candidate, EvaluationResult, MacpState
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
                    "Return ONLY valid JSON matching this schema: ${json_schema}."
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

        for candidate in state.candidates:
            score, feedback = self._evaluate_candidate(candidate, constraints, pass_score)
            passed = score >= pass_score
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
        if not state.approved:
            state.reflection_count += 1
        eva_state = self.get_agent_state(state)
        eva_state["last_selected"] = state.selected_candidate_id
        return state

    def _evaluate_candidate(self, candidate: Candidate, constraints: str, pass_score: float) -> tuple[float, str]:
        prompt = self.render_prompt(
            "eva_review",
            {
                "constraints": constraints,
                "candidate": candidate.content[:1000],
                "json_schema": render_schema_hint(EvaReviewPayload),
            },
        )
        response = self.call_llm(prompt=prompt, temperature=0.1)
        parse_ok = True
        parse_error = ""
        try:
            payload = parse_json_payload(response, EvaReviewPayload)
            model_score = float(payload.score)
            feedback = payload.feedback
            if payload.strengths:
                feedback += f"\nStrengths: {', '.join(payload.strengths)}"
            if payload.risks:
                feedback += f"\nRisks: {', '.join(payload.risks)}"
        except ValueError as exc:
            parse_ok = False
            parse_error = str(exc)
            model_score = self._extract_score(response)
            feedback = response
        heuristic_score = self._heuristic_score(candidate.content, constraints)
        score = round(0.7 * model_score + 0.3 * heuristic_score, 4)
        feedback = (
            f"{feedback}\n"
            f"[eva-debug] model_score={model_score:.2f}, heuristic_score={heuristic_score:.2f}, "
            f"pass_score={pass_score:.2f}, json_parse_ok={parse_ok}"
        )
        if parse_error:
            feedback += f"\n[eva-json-error] {parse_error}"
        return score, feedback

    def _extract_score(self, response: str) -> float:
        score_patterns = [
            r"【分数】\s*([0-9]*\.?[0-9]+)",
            r"\bscore\s*[:=]\s*([0-9]*\.?[0-9]+)",
            r"(\d+\.?\d*)\s*/\s*100",
            r"\b(0\.\d+|1(?:\.0+)?)\b",
        ]
        for idx, pattern in enumerate(score_patterns):
            match = re.search(pattern, response, flags=re.IGNORECASE)
            if not match:
                continue
            raw = float(match.group(1))
            if idx == 2:
                raw = raw / 100.0
            return max(0.0, min(1.0, raw))
        return 0.65

    def _heuristic_score(self, content: str, constraints: str) -> float:
        score = 0.45
        words = content.split()
        if len(words) >= 80:
            score += 0.2
        elif len(words) >= 40:
            score += 0.1
        constraint_tokens = [token.strip().lower() for token in constraints.split(";") if token.strip()]
        lowered = content.lower()
        if any(token in lowered for token in constraint_tokens):
            score += 0.2
        if any(marker in content for marker in ("1.", "2.", "-", "步骤", "Step")):
            score += 0.1
        if any(term in lowered for term in ("example", "案例", "举例")):
            score += 0.1
        return min(score, 0.95)
