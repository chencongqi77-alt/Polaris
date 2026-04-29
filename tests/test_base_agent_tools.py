"""专项测试：BaseAgent 的工具调用能力（真实调用）。"""

import os
import pytest
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

from app.agents.base import BaseAgent
from app.graph.state import MacpState
from app.interfaces.llm import OpenAILLMClient
from app.interfaces.mcp import DirectMCPClient
from app.memory import create_memory_store


class ToolProbeAgent(BaseAgent):
    """最小探针 Agent，用于验证 Base 能力是否可用。"""

    def __init__(self) -> None:
        api_key = os.environ.get("OPENAI_API_KEY", "").strip()
        if not api_key or api_key == "sk-你的阿里云百炼APIKey":
            pytest.skip("OPENAI_API_KEY 未设置或为默认值")

        try:
            memory_store = create_memory_store(mode="qdrant")
        except RuntimeError:
            pytest.skip("Qdrant 服务不可用")

        super().__init__(
            name="ToolProbe",
            llm_client=OpenAILLMClient(),
            mcp_client=DirectMCPClient(),
            memory_store=memory_store,
            config={"max_retries": 1},
        )

    def run(self, state: MacpState) -> MacpState:
        llm_text = self.call_llm("请返回一句测试文本，不超过20个字")
        mcp_result = self.call_mcp_tool("artifact.prepare", {"candidate_id": "probe", "preview": "probe content"})
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
    agent_state = result.extra_metadata.get("agent_state", {}).get("ToolProbe", {})

    # 验证 MCP 直接调用正常（artifact.prepare 返回 "prepared"）
    assert agent_state.get("mcp_status") == "prepared"

    # 验证 LLM 真实返回了文本
    assert isinstance(agent_state.get("llm_text"), str)
    assert len(agent_state["llm_text"]) > 0

    # 验证 Trace 记录
    assert "ToolProbe" in result.extra_metadata.get("trace", [])
