"""专项测试：TM Agent（真实调用）。"""

import os
import pytest
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

from app.agents.tm import TaskManagerAgent
from app.graph.state import MacpState
from app.interfaces.llm import OpenAILLMClient
from app.interfaces.mcp import DirectMCPClient
from app.memory import create_memory_store


@pytest.fixture
def tm_agent_real():
    """TM Agent with real LLM + MCP + Qdrant"""
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key or api_key == "sk-你的阿里云百炼APIKey":
        pytest.skip("OPENAI_API_KEY 未设置或为默认值")

    try:
        memory_store = create_memory_store(mode="qdrant")
    except RuntimeError:
        pytest.skip("Qdrant 服务不可用")

    return TaskManagerAgent(
        llm_client=OpenAILLMClient(),
        mcp_client=DirectMCPClient(),
        memory_store=memory_store,
    )


def test_tm_generates_subtasks_and_constraints(tm_agent_real: TaskManagerAgent) -> None:
    """测试 TM Agent 真实调用 LLM 生成子任务和约束"""
    state = MacpState(
        user_request="给文科生解释机器学习",
        constraints=["不能使用公式"],
    )

    result = tm_agent_real.execute(state)

    # 真实 LLM 应该生成子任务
    assert len(result.subtasks) > 0
    # 约束应该保留原有内容
    assert "不能使用公式" in result.constraints

    # Trace 记录
    assert "TM" in result.extra_metadata.get("trace", [])
    assert "TM" in result.extra_metadata.get("agent_state", {})

    # LLM 应该返回了预览内容
    agent_state = result.extra_metadata["agent_state"]["TM"]
    assert agent_state.get("llm_plan_preview"), "TM 应该记录 LLM 的规划预览"


def test_tm_with_complex_request(tm_agent_real: TaskManagerAgent) -> None:
    """测试 TM Agent 处理复杂请求"""
    state = MacpState(
        user_request="设计一个为期8周的Python入门课程，面向零基础学生",
        constraints=["循序渐进", "每节课有动手练习", "不用专业术语"],
    )

    result = tm_agent_real.execute(state)

    # 应该生成合理的子任务数量
    assert len(result.subtasks) >= 2
    assert len(result.subtasks) <= 10

    # 原有约束应该保留
    for c in ["循序渐进", "每节课有动手练习", "不用专业术语"]:
        assert c in result.constraints