"""
Pytest fixtures shared by MACP tests.
=====================================
所有 fixture 使用真实实现，不使用 Mock。

前置条件：
- Qdrant 服务运行在 localhost:6333
- .env 中配置了 OPENAI_API_KEY、OPENAI_BASE_URL、OPENAI_MODEL

运行方法：
    pytest tests/ -v -s
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
from app.interfaces.llm import OpenAILLMClient
from app.mcp_server.direct_client import DirectMCPClient
from app.interfaces.mcp import create_mcp_client
from app.memory import create_memory_store


# ═══════════════════════════════════════════════════════════════════════════════
# 环境检测
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.fixture
def has_api_key() -> bool:
    """检查是否设置了 OPENAI_API_KEY"""
    key = os.environ.get("OPENAI_API_KEY", "").strip()
    return bool(key) and key != "sk-你的阿里云百炼APIKey"


# ═══════════════════════════════════════════════════════════════════════════════
# LLM 客户端 fixtures（真实调用）
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.fixture
def real_llm(has_api_key: bool) -> OpenAILLMClient:
    """
    真实 LLM 客户端，使用阿里云百炼 API。
    如果没有设置 API Key，会跳过测试。
    """
    if not has_api_key:
        pytest.skip("OPENAI_API_KEY 未设置或为默认值，跳过 LLM 测试")
    return OpenAILLMClient()


# ═══════════════════════════════════════════════════════════════════════════════
# MCP 客户端 fixtures（本地直接调用）
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.fixture
def direct_mcp() -> DirectMCPClient:
    """MCP 客户端，直接调用本地工具函数（无需外部服务）"""
    return DirectMCPClient()


# ═══════════════════════════════════════════════════════════════════════════════
# Memory 存储 fixtures（真实 Qdrant）
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.fixture
def qdrant_memory():
    """真实 Qdrant 向量存储"""
    store = create_memory_store(mode="qdrant")
    if store is None:
        pytest.skip("Qdrant 服务不可用，跳过 Memory 测试")
    return store


# ═══════════════════════════════════════════════════════════════════════════════
# Agent fixtures（真实 LLM + 真实 MCP）
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.fixture
def tm_agent(real_llm: OpenAILLMClient, direct_mcp: DirectMCPClient, qdrant_memory) -> TaskManagerAgent:
    """TM Agent（真实 LLM + 真实 MCP + 真实 Qdrant）"""
    return TaskManagerAgent(llm_client=real_llm, mcp_client=direct_mcp, memory_store=qdrant_memory)


@pytest.fixture
def sg_agent(real_llm: OpenAILLMClient, direct_mcp: DirectMCPClient, qdrant_memory) -> SolutionGeneratorAgent:
    """SG Agent（真实 LLM + 真实 MCP + 真实 Qdrant，生成 2 个候选）"""
    return SolutionGeneratorAgent(k=2, llm_client=real_llm, mcp_client=direct_mcp, memory_store=qdrant_memory)


@pytest.fixture
def eva_agent(real_llm: OpenAILLMClient, direct_mcp: DirectMCPClient, qdrant_memory) -> EvaluatorAgent:
    """Eva Agent（真实 LLM + 真实 MCP + 真实 Qdrant，及格分 0.75）"""
    return EvaluatorAgent(pass_score=0.75, llm_client=real_llm, mcp_client=direct_mcp, memory_store=qdrant_memory)


# ═══════════════════════════════════════════════════════════════════════════════
# Runner fixtures（真实全链路）
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.fixture
def live_runner(has_api_key: bool, qdrant_memory) -> MacpGraphRunner:
    """
    真实全链路 Runner（需要 OPENAI_API_KEY + Qdrant）。
    如果没有 API Key，会跳过测试。
    """
    if not has_api_key:
        pytest.skip("OPENAI_API_KEY 未设置，跳过集成测试")

    llm = OpenAILLMClient()
    mcp = DirectMCPClient()

    tm = TaskManagerAgent(llm_client=llm, mcp_client=mcp, memory_store=qdrant_memory)
    sg = SolutionGeneratorAgent(k=2, llm_client=llm, mcp_client=mcp, memory_store=qdrant_memory)
    eva = EvaluatorAgent(pass_score=0.75, llm_client=llm, mcp_client=mcp, memory_store=qdrant_memory)

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