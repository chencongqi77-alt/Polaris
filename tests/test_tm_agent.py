"""专项测试：TM Agent。"""

from app.interfaces.llm import ChatMessage
from app.agents.tm import TaskManagerAgent
from app.graph.state import MacpState
from app.interfaces.llm import MockLLMClient
from app.interfaces.mcp import MockMCPClient
from app.interfaces.memory import MockMemoryStore


def test_tm_generates_subtasks_and_constraints() -> None:
    tm = TaskManagerAgent(
        llm_client=MockLLMClient(),
        mcp_client=MockMCPClient(),
        memory_store=MockMemoryStore(),
    )
    state = MacpState(
        user_request="给文科生解释机器学习",
        constraints=["不能使用公式"],
    )

    result = tm.execute(state)

    assert len(result.subtasks) > 0
    assert len(result.subtasks) == 3
    assert "TM" in result.metadata.get("trace", [])
    assert "TM" in result.metadata.get("agent_state", {})
    assert result.metadata["agent_state"]["TM"].get("llm_plan_preview")


class JsonTMClient:
    def generate(self, messages: list[ChatMessage], temperature: float = 0.2) -> str:
        return (
            '{"subtasks":["Define lesson objective","Draft examples","Prepare evaluation checklist"],'
            '"constraints":["beginner-friendly","testable output"]}'
        )


def test_tm_parses_json_payload_when_valid() -> None:
    tm = TaskManagerAgent(llm_client=JsonTMClient(), mcp_client=MockMCPClient(), memory_store=MockMemoryStore())
    state = MacpState(user_request="Teach Python basics", constraints=[])

    result = tm.execute(state)

    assert result.subtasks[0] == "Define lesson objective"
    assert "beginner-friendly" in result.constraints
    assert result.metadata["agent_state"]["TM"]["json_parse_ok"] is True
