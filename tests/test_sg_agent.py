"""专项测试：SG Agent。"""

from app.interfaces.llm import ChatMessage
from app.agents.sg import SolutionGeneratorAgent
from app.graph.state import EvaluationResult, MacpState
from app.interfaces.llm import MockLLMClient
from app.interfaces.mcp import MockMCPClient
from app.interfaces.memory import MockMemoryStore


def test_sg_generates_k_candidates_and_calls_mcp() -> None:
    sg = SolutionGeneratorAgent(
        k=2,
        llm_client=MockLLMClient(),
        mcp_client=MockMCPClient(),
        memory_store=MockMemoryStore(),
    )
    state = MacpState(
        user_request="解释监督学习",
        subtasks=["用生活例子说明", "补一个简单案例"],
        constraints=["面向新手"],
    )

    result = sg.execute(state)

    assert len(result.candidates) == 2
    assert all(c.metadata.get("artifact", {}).get("status") == "ok" for c in result.candidates)
    assert "SG" in result.metadata.get("trace", [])
    assert len(result.metadata["agent_state"]["SG"].get("styles_used", [])) == 2


class JsonSGClient:
    def __init__(self) -> None:
        self._count = 0

    def generate(self, messages: list[ChatMessage], temperature: float = 0.2) -> str:
        self._count += 1
        return (
            '{"content":"Candidate %d: use practical mini examples for newcomers.",'
            '"rationale":"Different style","checklist":["clear","practical"]}'
        ) % self._count


def test_sg_parses_json_candidate_payload() -> None:
    sg = SolutionGeneratorAgent(k=2, llm_client=JsonSGClient(), mcp_client=MockMCPClient(), memory_store=MockMemoryStore())
    state = MacpState(user_request="Explain supervised learning", subtasks=["a", "b", "c"], constraints=["clear"])

    result = sg.execute(state)

    assert result.candidates[0].content.startswith("Candidate 1")
    assert all(c.metadata.get("json_parse_ok") is True for c in result.candidates)


class CapturePromptSGClient:
    def __init__(self) -> None:
        self.prompts: list[str] = []

    def generate(self, messages: list[ChatMessage], temperature: float = 0.2) -> str:
        self.prompts.append(messages[-1].content)
        return '{"content":"Revised candidate","rationale":"Uses review feedback","checklist":["revised"]}'


def test_sg_includes_human_and_eva_feedback_in_prompt() -> None:
    client = CapturePromptSGClient()
    sg = SolutionGeneratorAgent(
        k=1,
        llm_client=client,
        mcp_client=MockMCPClient(),
        memory_store=MockMemoryStore(),
    )
    state = MacpState(
        user_request="Explain supervised learning for liberal arts students",
        subtasks=["Rewrite with analogy"],
        constraints=["beginner-friendly"],
        reflection_count=1,
        evaluations=[
            EvaluationResult(
                candidate_id="cand-1",
                score=0.42,
                passed=False,
                feedback="Too abstract, add a concrete analogy.",
            )
        ],
        metadata={
            "human_feedback": {
                "approved": False,
                "notes": "Use a friend-recommendation analogy instead of technical jargon.",
                "subtasks": ["Rewrite with analogy"],
                "constraints": ["beginner-friendly"],
            }
        },
    )

    result = sg.execute(state)

    assert len(result.candidates) == 1
    prompt = client.prompts[0]
    assert "Human reviewer requested changes" in prompt
    assert "friend-recommendation analogy" in prompt
    assert "Address Eva feedback" in prompt
