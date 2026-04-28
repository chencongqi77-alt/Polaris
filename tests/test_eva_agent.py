"""专项测试：Eva Agent。"""

from app.interfaces.llm import ChatMessage
from app.agents.eva import EvaluatorAgent
from app.graph.state import Candidate, MacpState
from app.interfaces.llm import MockLLMClient
from app.interfaces.mcp import MockMCPClient
from app.interfaces.memory import MockMemoryStore


def test_eva_scores_and_selects_best_candidate() -> None:
    eva = EvaluatorAgent(
        pass_score=0.75,
        llm_client=MockLLMClient(),
        mcp_client=MockMCPClient(),
        memory_store=MockMemoryStore(),
    )
    state = MacpState(
        user_request="介绍神经网络",
        constraints=["通俗表达"],
        candidates=[
            Candidate(id="cand-1", content="候选方案1"),
            Candidate(id="cand-2", content="候选方案2"),
        ],
    )

    result = eva.execute(state)

    assert len(result.evaluations) == 2
    assert result.selected_candidate_id is not None
    assert isinstance(result.approved, bool)
    assert "Eva" in result.metadata.get("trace", [])
    assert all("[eva-debug]" in item.feedback for item in result.evaluations)


class JsonEvaClient:
    def __init__(self) -> None:
        self._score = 0.82

    def generate(self, messages: list[ChatMessage], temperature: float = 0.2) -> str:
        payload = (
            '{"score": %.2f, "feedback": "Good structure and clarity.",'
            '"strengths": ["clear steps"], "risks": ["needs deeper examples"]}'
        ) % self._score
        self._score -= 0.1
        return payload


def test_eva_parses_json_review_payload() -> None:
    eva = EvaluatorAgent(pass_score=0.7, llm_client=JsonEvaClient(), mcp_client=MockMCPClient(), memory_store=MockMemoryStore())
    state = MacpState(
        user_request="Introduce neural networks",
        constraints=["beginner-friendly"],
        candidates=[
            Candidate(id="cand-1", content="Step 1. Explain neurons with daily-life examples."),
            Candidate(id="cand-2", content="Brief overview only."),
        ],
    )

    result = eva.execute(state)

    assert result.selected_candidate_id == "cand-1"
    assert "json_parse_ok=True" in result.evaluations[0].feedback
