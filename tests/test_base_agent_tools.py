"""专项测试：BaseAgent 的工具调用能力。"""

from app.agents.base import BaseAgent
from app.graph.state import MacpState
from app.interfaces.llm import MockLLMClient
from app.interfaces.mcp import MockMCPClient
from app.interfaces.memory import MockMemoryStore


class ToolProbeAgent(BaseAgent):
    """最小探针 Agent，用于验证 Base 能力是否可用。"""

    def __init__(self) -> None:
        super().__init__(
            name="ToolProbe",
            llm_client=MockLLMClient(),
            mcp_client=MockMCPClient(),
            memory_store=MockMemoryStore(),
            config={"max_retries": 1},
        )

    def run(self, state: MacpState) -> MacpState:
        llm_text = self.call_llm("请返回一句测试文本")
        mcp_result = self.call_mcp_tool("artifact.prepare", {"candidate_id": "probe"})
        memories = self.retrieve_memories(state, top_k=1)

        probe_state = self.get_agent_state(state)
        probe_state["llm_text"] = llm_text
        probe_state["mcp_status"] = mcp_result.get("status")
        probe_state["memory_hits_count"] = len(memories)
        return state


def test_base_agent_can_call_llm_mcp_and_memory() -> None:
    agent = ToolProbeAgent()
    state = MacpState(user_request="测试 BaseAgent 能力")

    result = agent.execute(state)
    agent_state = result.metadata.get("agent_state", {}).get("ToolProbe", {})

    assert agent_state.get("mcp_status") == "ok"
    assert isinstance(agent_state.get("llm_text"), str)
    assert "ToolProbe" in result.metadata.get("trace", [])
