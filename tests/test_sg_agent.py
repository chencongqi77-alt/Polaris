"""专项测试：SG Agent（真实调用）。"""

import os
import pytest
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

from app.agents.sg import SolutionGeneratorAgent
from app.graph.state import MacpState
from app.interfaces.llm import OpenAILLMClient
from app.interfaces.mcp import DirectMCPClient
from app.memory import create_memory_store


@pytest.fixture
def sg_agent_real():
    """SG Agent with real LLM + MCP + Qdrant"""
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key or api_key == "sk-你的阿里云百炼APIKey":
        pytest.skip("OPENAI_API_KEY 未设置或为默认值")

    try:
        memory_store = create_memory_store(mode="qdrant")
    except RuntimeError:
        pytest.skip("Qdrant 服务不可用")

    return SolutionGeneratorAgent(
        k=2,
        llm_client=OpenAILLMClient(),
        mcp_client=DirectMCPClient(),
        memory_store=memory_store,
    )


def test_sg_generates_k_candidates_and_calls_mcp(sg_agent_real: SolutionGeneratorAgent) -> None:
    """测试 SG Agent 真实生成 k 个候选方案并调用 MCP"""
    state = MacpState(
        user_request="解释监督学习",
        subtasks=["用生活例子说明", "补一个简单案例"],
        constraints=["面向新手"],
    )

    result = sg_agent_real.execute(state)

    # 应该生成 2 个候选方案
    assert len(result.candidates) == 2

    # 每个候选方案的 MCP artifact 状态应该正常
    for c in result.candidates:
        artifact = c.metadata.get("artifact", {})
        assert artifact.get("status") in ("ok", "prepared"), f"候选 {c.id} 的 artifact 状态异常: {artifact}"

    # Trace 记录
    assert "SG" in result.extra_metadata.get("trace", [])
    assert len(result.extra_metadata["agent_state"]["SG"].get("styles_used", [])) == 2


def test_sg_with_reflection(sg_agent_real: SolutionGeneratorAgent) -> None:
    """测试 SG Agent 在反思模式下生成候选方案"""
    from app.graph.state import EvaluationResult

    state = MacpState(
        user_request="解释监督学习",
        subtasks=["用生活例子说明"],
        constraints=["面向新手"],
        reflection_count=1,
        evaluations=[
            EvaluationResult(
                candidate_id="cand-0",
                score=0.4,
                passed=False,
                feedback="太抽象了，需要更具体的例子",
            )
        ],
    )

    result = sg_agent_real.execute(state)

    # 应该生成新的候选方案
    assert len(result.candidates) == 2
    # 内容应该非空
    for c in result.candidates:
        assert len(c.content) > 0