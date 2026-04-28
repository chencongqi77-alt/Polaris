from __future__ import annotations

import re

from app.agents.base import BaseAgent
from app.agents.io_models import TMPlanPayload, parse_json_payload, render_schema_hint
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
                    "Return ONLY valid JSON matching this schema: ${json_schema}\n"
                    "Generate exactly 3 concise subtasks for SG and Eva."
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
                "json_schema": render_schema_hint(TMPlanPayload),
            },
        )
        if memories:
            prompt += f"\nReference memories: {memories}"
        draft = self.call_llm(prompt=prompt)

        tm_payload: TMPlanPayload | None = None
        parse_error = ""
        try:
            tm_payload = parse_json_payload(draft, TMPlanPayload)
        except ValueError as exc:
            parse_error = str(exc)

        if tm_payload is not None:
            state.subtasks = [item.strip() for item in tm_payload.subtasks]
            if not state.constraints:
                state.constraints = [item.strip() for item in tm_payload.constraints if item.strip()]
        elif not state.subtasks:
            parsed = self._parse_subtasks(draft)
            state.subtasks = parsed or self._default_subtasks()
            if not state.constraints:
                state.constraints = self._derive_constraints(state.user_request)
        elif not state.constraints:
            state.constraints = self._derive_constraints(state.user_request)

        tm_state = self.get_agent_state(state)
        tm_state["last_plan"] = state.subtasks
        tm_state["llm_plan_preview"] = draft[:200]
        tm_state["json_schema_enabled"] = True
        tm_state["json_parse_ok"] = tm_payload is not None
        if parse_error:
            tm_state["json_parse_error"] = parse_error
        return state

    def _parse_subtasks(self, text: str) -> list[str]:
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        parsed: list[str] = []
        for line in lines:
            cleaned = re.sub(r"^(\d+[\.\)]|[-*])\s*", "", line).strip()
            if not cleaned:
                continue
            if len(cleaned) < 6:
                continue
            parsed.append(cleaned)
            if len(parsed) >= 3:
                break
        return parsed

    def _derive_constraints(self, user_request: str) -> list[str]:
        request = user_request.lower()
        constraints = ["clear output", "testable output"]
        if any(keyword in request for keyword in ("beginner", "零基础", "新手", "文科")):
            constraints.append("beginner-friendly language")
        if any(keyword in request for keyword in ("python", "代码", "program", "开发")):
            constraints.append("include runnable examples")
        return constraints

    def _default_subtasks(self) -> list[str]:
        return [
            "Identify concrete user learning goal and output shape",
            "Generate multiple candidate answers with different styles",
            "Evaluate candidates against constraints and select one",
        ]
