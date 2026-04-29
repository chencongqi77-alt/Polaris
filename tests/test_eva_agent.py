"""专项测试：Eva Agent（真实调用）。"""

import os
import pytest
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

from app.agents.eva import EvaluatorAgent
from app.graph.state import Candidate, MacpState
from app.interfaces.llm import OpenAILLMClient
from app.interfaces.mcp import DirectMCPClient
from app.memory import create_memory_store


@pytest.fixture
def eva_agent_real():
    """Eva Agent with real LLM + MCP + Qdrant"""
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key or api_key == "sk-你的阿里云百炼APIKey":
        pytest.skip("OPENAI_API_KEY 未设置或为默认值")

    try:
        memory_store = create_memory_store(mode="qdrant")
    except RuntimeError:
        pytest.skip("Qdrant 服务不可用")

    return EvaluatorAgent(
        pass_score=0.75,
        llm_client=OpenAILLMClient(),
        mcp_client=DirectMCPClient(),
        memory_store=memory_store,
    )


def test_eva_scores_and_selects_best_candidate(eva_agent_real: EvaluatorAgent) -> None:
    """测试 Eva Agent 真实评分并选择最佳候选"""
    state = MacpState(
        user_request="介绍神经网络",
        constraints=["通俗表达"],
        candidates=[
            Candidate(id="cand-1", content="神经网络是一种模仿人脑结构的计算模型，由多层相互连接的节点组成。"),
            Candidate(id="cand-2", content="神经网络是人工智能的核心技术之一，广泛应用于图像识别和自然语言处理。"),
        ],
    )

    result = eva_agent_real.execute(state)

    # 应该对每个候选方案进行评估
    assert len(result.evaluations) == 2

    # 应该选出了最佳候选
    assert result.selected_candidate_id is not None
    assert result.selected_candidate_id in ("cand-1", "cand-2")

    # 评估分数应该是合理范围
    for eval_result in result.evaluations:
        assert 0.0 <= eval_result.score <= 1.0
        assert len(eval_result.feedback) > 0

    # approved 状态应该与评分一致
    best_eval = next(e for e in result.evaluations if e.candidate_id == result.selected_candidate_id)
    assert result.approved == best_eval.passed

    # Trace 记录
    assert "Eva" in result.extra_metadata.get("trace", [])
