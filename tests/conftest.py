"""
Pytest fixtures shared by MACP tests.
=====================================
这个文件定义了测试共享的 fixtures。

主要测试文件是 test_for_humans.py，它提供了：
- 体验模式（无 API Key）：用 MockLLMClient 模拟流程
- 真枪实弹模式（有 API Key）：真实调用 LLM API

运行方法：
    pytest tests/test_for_humans.py -v -s
"""

from __future__ import annotations

import os
from typing import Any

import pytest
from dotenv import load_dotenv

# 自动加载项目根目录下的 .env 文件
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

from app.agents.eva import EvaluatorAgent
from app.agents.sg import SolutionGeneratorAgent
from app.agents.tm import TaskManagerAgent
from app.graph.main_graph import MacpGraphRunner
from app.graph.state import MacpState
from app.interfaces.llm import MockLLMClient, OpenAILLMClient
from app.interfaces.mcp import MockMCPClient
from app.interfaces.memory import MockMemoryStore


# ═══════════════════════════════════════════════════════════════════════════════
# 环境检测
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.fixture
def has_api_key() -> bool:
    """检查是否设置了 OPENAI_API_KEY"""
    key = os.environ.get("OPENAI_API_KEY", "").strip()
    return bool(key)


# ═══════════════════════════════════════════════════════════════════════════════
# LLM 客户端 fixtures
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.fixture
def mock_llm() -> MockLLMClient:
    """基础 Mock LLM，用于快速单元测试"""
    return MockLLMClient()


@pytest.fixture
def real_llm(has_api_key: bool) -> Any:
    """
    真实 LLM 客户端，需要 OPENAI_API_KEY。
    如果没有设置，会跳过测试。
    """
    if not has_api_key:
        pytest.skip("OPENAI_API_KEY 未设置，跳过真实 LLM 测试")
    return OpenAILLMClient()


# ═══════════════════════════════════════════════════════════════════════════════
# Agent fixtures
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.fixture
def mock_mcp() -> MockMCPClient:
    """Mock MCP 客户端"""
    return MockMCPClient()


@pytest.fixture
def mock_memory() -> MockMemoryStore:
    """Mock Memory 存储"""
    return MockMemoryStore()


@pytest.fixture
def tm_agent(mock_llm: MockLLMClient, mock_mcp: MockMCPClient) -> TaskManagerAgent:
    """TM Agent（使用 Mock LLM）"""
    return TaskManagerAgent(llm_client=mock_llm, mcp_client=mock_mcp)


@pytest.fixture
def sg_agent(mock_llm: MockLLMClient, mock_mcp: MockMCPClient) -> SolutionGeneratorAgent:
    """SG Agent（使用 Mock LLM，生成 2 个候选）"""
    return SolutionGeneratorAgent(k=2, llm_client=mock_llm, mcp_client=mock_mcp)


@pytest.fixture
def eva_agent(mock_llm: MockLLMClient, mock_mcp: MockMCPClient) -> EvaluatorAgent:
    """Eva Agent（使用 Mock LLM，及格分 0.75）"""
    return EvaluatorAgent(pass_score=0.75, llm_client=mock_llm, mcp_client=mock_mcp)


# ═══════════════════════════════════════════════════════════════════════════════
# Runner fixtures
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.fixture
def mock_runner(
    mock_llm: MockLLMClient,
    mock_mcp: MockMCPClient,
    mock_memory: MockMemoryStore,
) -> MacpGraphRunner:
    """Runner with mock LLM and MCP（快速测试，无真实 API 调用）"""
    tm = TaskManagerAgent(llm_client=mock_llm, mcp_client=mock_mcp, memory_store=mock_memory)
    sg = SolutionGeneratorAgent(k=2, llm_client=mock_llm, mcp_client=mock_mcp, memory_store=mock_memory)
    eva = EvaluatorAgent(pass_score=0.75, llm_client=mock_llm, mcp_client=mock_mcp, memory_store=mock_memory)
    return MacpGraphRunner(tm=tm, sg=sg, eva=eva, checkpoint_path=None)


@pytest.fixture
def live_runner(has_api_key: bool) -> MacpGraphRunner:
    """
    Runner with real LLM（需要 OPENAI_API_KEY）。
    如果没有 API Key，会使用 Mock LLM 作为后备。
    """
    if has_api_key:
        llm = OpenAILLMClient()
    else:
        llm = MockLLMClient()

    mcp = MockMCPClient()
    memory = MockMemoryStore()

    tm = TaskManagerAgent(llm_client=llm, mcp_client=mcp, memory_store=memory)
    sg = SolutionGeneratorAgent(k=2, llm_client=llm, mcp_client=mcp, memory_store=memory)
    eva = EvaluatorAgent(pass_score=0.75, llm_client=llm, mcp_client=mcp, memory_store=memory)

    return MacpGraphRunner(tm=tm, sg=sg, eva=eva, checkpoint_path=None)


# ═══════════════════════════════════════════════════════════════════════════════
# State fixtures
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.fixture
def base_state() -> MacpState:
    """基础测试状态"""
    return MacpState(user_request="Build a machine learning intro class")


@pytest.fixture
def complex_state() -> MacpState:
    """带子任务和约束的状态"""
    return MacpState(
        user_request="Create a beginner AI syllabus",
        subtasks=["outline learning objectives", "generate examples", "write final lesson"],
        constraints=["no formulas", "story-driven", "beginner-friendly"],
    )