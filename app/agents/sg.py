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

        # Research phase: gather external context via web search tools
        research_context = self._run_research_phase(state)

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
            if research_context:
                prompt += f"\nExternal research findings:\n{research_context}"
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

    def _run_research_phase(self, state: MacpState) -> str:
        """Run a research phase using web search tools to gather external context.

        Extracts key topics from the user request and subtasks, searches GitHub
        and the web for relevant information, and returns a summary string.

        Args:
            state: Current pipeline state.

        Returns:
            A string with research findings to inject into the prompt, or empty string.
        """
        import json as _json

        research_items: list[str] = []

        # Build search queries from user request and subtasks
        queries: list[str] = []
        user_q = state.user_request.strip()
        if user_q:
            # Use the first 80 chars as a search query
            queries.append(user_q[:80])

        # Add one query derived from subtasks if available
        if state.subtasks:
            subtask_q = " ".join(state.subtasks[:3])[:80]
            if subtask_q and subtask_q not in queries:
                queries.append(subtask_q)

        if not queries:
            return ""

        # Search GitHub for relevant repositories
        for q in queries[:2]:  # limit to 2 queries to avoid too many calls
            try:
                gh_result = self.call_mcp_tool(
                    "search.github",
                    {"query": q, "per_page": 3, "sort": "stars"},
                )
                if isinstance(gh_result, dict) and gh_result.get("status") == "ok":
                    for repo in gh_result.get("results", [])[:3]:
                        name = repo.get("name", "")
                        desc = repo.get("description", "")[:120]
                        url = repo.get("url", "")
                        stars = repo.get("stars", 0)
                        research_items.append(
                            f"[GitHub] {name} ({stars}★): {desc} {url}"
                        )
            except Exception:
                pass  # Non-fatal: skip on network/tool errors

        # Web search for general context
        for q in queries[:2]:
            try:
                web_result = self.call_mcp_tool(
                    "search.web",
                    {"query": q, "max_results": 3},
                )
                if isinstance(web_result, dict) and web_result.get("status") == "ok":
                    for item in web_result.get("results", [])[:3]:
                        title = item.get("title", "")[:80]
                        snippet = item.get("snippet", "")[:120]
                        url = item.get("url", "")
                        research_items.append(
                            f"[Web] {title}: {snippet} {url}"
                        )
            except Exception:
                pass  # Non-fatal

        if research_items:
            # Store research in state metadata for traceability
            agent_state = self.get_agent_state(state)
            agent_state["research_items"] = research_items
            return "\n".join(research_items)

        return ""

    def _build_revision_instructions(self, state: MacpState) -> str:
        guidance: list[str] = []
        human_feedback = state.extra_metadata.get("human_feedback", {})
        if isinstance(human_feedback, dict):
            notes = str(human_feedback.get("notes", "")).strip()
            if human_feedback.get("approved") is False:
                if notes:
                    guidance.append(f"Human reviewer requested changes: {notes}")
                if human_feedback.get("subtasks"):
                    guidance.append("Follow the revised subtasks from human review.")
                if human_feedback.get("constraints"):
                    guidance.append("Respect the revised constraints from human review.")
            elif notes:
                guidance.append(f"Human reviewer's additional request: {notes}")

            # Handle per-item feedback from feedback mode (works in both approved/reject flows)
            fb = human_feedback.get("feedback", {})
            if isinstance(fb, dict):
                subtask_fbs = fb.get("subtask_feedback", [])
                if isinstance(subtask_fbs, list):
                    for i, item_fb in enumerate(subtask_fbs):
                        item_fb = str(item_fb).strip()
                        if item_fb:
                            subtask_text = ""
                            if state.subtasks and i < len(state.subtasks):
                                subtask_text = f" (subtask: {state.subtasks[i][:60]})"
                            guidance.append(f"Human feedback on subtask {i + 1}{subtask_text}: {item_fb}")
                constraint_fbs = fb.get("constraint_feedback", [])
                if isinstance(constraint_fbs, list):
                    for i, item_fb in enumerate(constraint_fbs):
                        item_fb = str(item_fb).strip()
                        if item_fb:
                            constraint_text = ""
                            if state.constraints and i < len(state.constraints):
                                constraint_text = f" (constraint: {state.constraints[i][:60]})"
                            guidance.append(f"Human feedback on constraint {i + 1}{constraint_text}: {item_fb}")

        if state.reflection_count > 0 and state.evaluations:
            failed_feedback = [item.feedback.strip() for item in state.evaluations if not item.passed and item.feedback.strip()]
            if failed_feedback:
                guidance.append(f"Address Eva feedback: {' | '.join(failed_feedback[:2])}")

        return " ".join(guidance).strip()


# Alias for backward compatibility
SGAgent = SolutionGeneratorAgent
