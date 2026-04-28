from __future__ import annotations

from app.agents.base import BaseAgent
from app.agents.io_models import SGCandidatePayload, parse_json_payload, render_schema_hint
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
                    "Return ONLY valid JSON matching this schema: ${json_schema}\n"
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
        subtasks = context["subtasks"] or self._fallback_subtasks(state.user_request)
        rendered_subtasks = "; ".join(subtasks) or "none"
        rendered_constraints = "; ".join(context["constraints"]) or "none"
        styles = ["structured lesson", "storytelling", "step-by-step practice"]
        feedback_instructions = self._build_revision_instructions(state)

        candidates = []
        for idx in range(k):
            prompt = self.render_prompt(
                "sg_generate",
                {
                    "user_request": context["user_request"],
                    "subtasks": rendered_subtasks,
                    "constraints": rendered_constraints,
                    "json_schema": render_schema_hint(SGCandidatePayload),
                },
            )
            prompt += f"\nGeneration style: {styles[idx % len(styles)]}."
            if feedback_instructions:
                prompt += f"\nRevision guidance: {feedback_instructions}"
            if memories:
                prompt += f"\nUseful prior artifacts: {memories}"
            raw = self.call_llm(prompt=prompt, temperature=0.6)
            parse_ok = True
            parse_error = ""
            try:
                payload = parse_json_payload(raw, SGCandidatePayload)
                content = payload.content.strip()
                rationale = payload.rationale
                checklist = payload.checklist
            except ValueError as exc:
                parse_ok = False
                parse_error = str(exc)
                content = raw
                rationale = ""
                checklist = []
            artifact = self.call_mcp_tool(
                "artifact.prepare",
                {"candidate_id": f"cand-{idx + 1}", "preview": content[:120]},
            )
            candidates.append(
                Candidate(
                    id=f"cand-{idx + 1}",
                    content=content,
                    modality="text",
                    metadata={
                        "artifact": artifact,
                        "json_parse_ok": parse_ok,
                        "json_parse_error": parse_error,
                        "rationale": rationale,
                        "checklist": checklist,
                    },
                )
            )

        state.candidates = candidates
        sg_state = self.get_agent_state(state)
        sg_state["last_candidate_count"] = len(state.candidates)
        sg_state["styles_used"] = styles[: len(state.candidates)]
        sg_state["json_schema_enabled"] = True
        sg_state["revision_guidance"] = feedback_instructions
        return state

    def _fallback_subtasks(self, user_request: str) -> list[str]:
        return [
            f"Clarify objective for: {user_request}",
            "Draft at least two alternative solutions",
            "Prepare final version for evaluator scoring",
        ]

    def _build_revision_instructions(self, state: MacpState) -> str:
        guidance: list[str] = []
        human_feedback = state.metadata.get("human_feedback", {})
        if isinstance(human_feedback, dict):
            if human_feedback.get("approved") is False:
                notes = str(human_feedback.get("notes", "")).strip()
                if notes:
                    guidance.append(f"Human reviewer requested changes: {notes}")
                if human_feedback.get("subtasks"):
                    guidance.append("Follow the revised subtasks from human review.")
                if human_feedback.get("constraints"):
                    guidance.append("Respect the revised constraints from human review.")

        if state.reflection_count > 0 and state.evaluations:
            failed_feedback = [item.feedback.strip() for item in state.evaluations if not item.passed and item.feedback.strip()]
            if failed_feedback:
                guidance.append(f"Address Eva feedback: {' | '.join(failed_feedback[:2])}")

        return " ".join(guidance).strip()
