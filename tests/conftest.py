"""
Pytest fixtures shared by all MACP tests.
"""

from __future__ import annotations

import os
from typing import Dict, Any

import pytest
from dotenv import load_dotenv

# 自动加载项目根目录下的 .env 文件
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

from app.agents.eva import EvaluatorAgent
from app.agents.io_models import AgentInput, AgentOutput
from app.agents.sg import SolutionGeneratorAgent
from app.agents.tm import TaskManagerAgent
from app.graph.main_graph import MacpGraphRunner
from app.graph.state import Candidate, EvaluationResult, MacpState
from app.interfaces.llm import ChatMessage, LLMClient, MockLLMClient
from app.interfaces.mcp import MCPClient, MockMCPClient
from app.prompts.registry import PromptRegistry


# ---------------------------------------------------------------------------
# 基础夹具
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_llm() -> MockLLMClient:
    """A deterministic mock LLM that echoes the last prompt."""
    return MockLLMClient()


@pytest.fixture
def mock_mcp() -> MockMCPClient:
    """A deterministic mock MCP client."""
    return MockMCPClient()


@pytest.fixture
def prompt_registry() -> PromptRegistry:
    """Default prompt registry with canned templates."""
    return PromptRegistry()


# ---------------------------------------------------------------------------
# 状态夹具
# ---------------------------------------------------------------------------

@pytest.fixture
def base_state() -> MacpState:
    """Minimal state for quick unit-test assertion."""
    return MacpState(user_request="Build a machine learning intro class")


@pytest.fixture
def complex_state() -> MacpState:
    """State with subtasks and constraints pre-set."""
    return MacpState(
        user_request="Create a beginner AI syllabus",
        subtasks=["outline learning objectives", "generate examples", "write final lesson"],
        constraints=["no formulas", "story-driven", "beginner-friendly"],
    )


@pytest.fixture
def approved_state() -> MacpState:
    """State that has already passed Eva."""
    c1 = Candidate(id="cand-1", content="Simple AI intro", modality="text")
    return MacpState(
        user_request="Test approved path",
        candidates=[c1],
        evaluations=[EvaluationResult(candidate_id="cand-1", score=0.95, passed=True, feedback="ok")],
        selected_candidate_id="cand-1",
        approved=True,
    )


@pytest.fixture
def max_reflection_state() -> MacpState:
    """State at max reflections – should route to stop."""
    c1 = Candidate(id="cand-1", content="Draft content", modality="text")
    return MacpState(
        user_request="Test max-reflections path",
        candidates=[c1],
        evaluations=[EvaluationResult(candidate_id="cand-1", score=0.50, passed=False, feedback="needs work")],
        approved=False,
        reflection_count=2,
        max_reflections=2,
    )


# ---------------------------------------------------------------------------
# Agent 夹具
# ---------------------------------------------------------------------------

@pytest.fixture
def tm_agent() -> TaskManagerAgent:
    return TaskManagerAgent()


@pytest.fixture
def sg_agent() -> SolutionGeneratorAgent:
    return SolutionGeneratorAgent(k=2)


@pytest.fixture
def eva_agent() -> EvaluatorAgent:
    return EvaluatorAgent(pass_score=0.75)


# ---------------------------------------------------------------------------
# 运行器夹具（Mock LLM / 真实 LLM 二选一）
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_runner(base_state: MacpState) -> MacpGraphRunner:
    """Runner with deterministic mock LLM and MCP, no checkpoint file."""
    llm = MockLLMClient()
    mcp = MockMCPClient()
    prompts = PromptRegistry({
        "tm_plan": "User request: ${user_request}\nConstraints: ${constraints}\nGenerate 3 concise subtasks.",
        "sg_generate": "Task: ${user_request}\nGenerate one candidate.",
        "eva_review": "Constraints: ${constraints}\nCandidate: ${candidate}\nCritique.",
    })
    tm = TaskManagerAgent(llm_client=llm, mcp_client=mcp, prompt_registry=prompts)
    sg = SolutionGeneratorAgent(k=2, llm_client=llm, mcp_client=mcp, prompt_registry=prompts)
    eva = EvaluatorAgent(pass_score=0.75, llm_client=llm, mcp_client=mcp, prompt_registry=prompts)
    runner = MacpGraphRunner(tm=tm, sg=sg, eva=eva, checkpoint_path=None)
    return runner
