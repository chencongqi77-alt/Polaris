"""
测试 2：端到端真实流程测试
-------------------------
使用真实的 LLM（需要 API Key）跑通 TM → Human Review → SG → Eva 的完整流程，
验证整个图是否能正常流转、Agent 之间数据能正确传递。

    运行方式：
        # 需要先设置 API Key
        set OPENAI_API_KEY=sk-xxx  （Windows）
        pytest tests/test_real_flow.py -v -s

    注意：
        这些测试会实际消耗 token，建议用小模型如 gpt-4o-mini
"""

from __future__ import annotations

import os

import pytest

from app.graph.state import MacpState
from app.graph.main_graph import MacpGraphRunner
from app.agents.tm import TaskManagerAgent
from app.agents.sg import SolutionGeneratorAgent
from app.agents.eva import EvaluatorAgent
from app.interfaces.llm import OpenAILLMClient
from app.interfaces.mcp import MockMCPClient
from app.prompts.registry import PromptRegistry


# ============================================================
# 测试前的环境检查
# ============================================================

def _require_api_key() -> str:
    """如果环境变量没有 API Key，跳过测试。"""
    key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not key:
        pytest.skip("OPENAI_API_KEY 未设置 — 跳过真实 LLM 测试")
    return key


# ============================================================
# 测试 1：最小的端到端流程
# ============================================================

class TestMinimalEndToEnd:

    @pytest.fixture
    def real_runner(self):
        """
        构造一个使用真实 LLM + Mock MCP 的 MacpGraphRunner。
        Mock MCP 让测试不依赖外部文件系统，专注验证 LLM 连通和 Graph 流转。
        """
        _require_api_key()

        llm = OpenAILLMClient()
        mcp = MockMCPClient()

        tm = TaskManagerAgent(llm_client=llm, mcp_client=mcp)
        sg = SolutionGeneratorAgent(k=2, llm_client=llm, mcp_client=mcp)
        eva = EvaluatorAgent(pass_score=0.75, llm_client=llm, mcp_client=mcp)
        return MacpGraphRunner(tm=tm, sg=sg, eva=eva, checkpoint_path=None)

    def test_tm_generates_subtasks(self, real_runner):
        """
        TM 节点：测试它能否从用户请求中生成合理的子任务。
        只走到 TM 结束，不跑完整图。
        """
        state = MacpState(user_request="为文科大学生设计一门 ML 科普课，不能有数学公式")
        tm = real_runner.tm
        result = tm.execute(state)

        assert len(result.subtasks) > 0, "TM 未能生成子任务"
        print(f"\n[TM] 生成的子任务: {result.subtasks}")

    def test_sg_generates_candidates(self, real_runner):
        """
        SG 节点：直接测试 SG 能否根据任务生成候选方案。
        绕过 TM，直接造一个带 subtasks 的状态传给 SG。
        """
        state = MacpState(
            user_request="解释什么是有监督学习",
            subtasks=["用生活类比解释监督学习", "给一个简单例子", "说明应用场景"],
            constraints=["不能使用公式", "面向编程新手"],
        )
        sg = real_runner.sg
        result = sg.execute(state)

        assert len(result.candidates) > 0, "SG 未能生成候选方案"
        print(f"\n[SG] 生成的候选数: {len(result.candidates)}")
        for c in result.candidates:
            print(f"  - {c.id}: {c.content[:100]}...")

    def test_eva_scores_candidates(self, real_runner):
        """
        Eva 节点：测试 Eva 能否对 SG 生成的候选方案进行评分。
        """
        state = MacpState(
            user_request="解释神经网络",
            subtasks=["基本概念", "工作原理", "应用举例"],
            constraints=["通俗易懂", "拒绝公式"],
        )
        # 先让 SG 生成候选
        sg = real_runner.sg
        state = sg.execute(state)

        # 再让 Eva 评估
        eva = real_runner.eva
        result = eva.execute(state)

        assert len(result.evaluations) > 0, "Eva 未能生成评估结果"
        print(f"\n[Eva] 评估结果:")
        for e in result.evaluations:
            print(f"  - {e.candidate_id}: score={e.score}, passed={e.passed}")
        assert result.selected_candidate_id is not None, "Eva 未选择最佳候选"
        print(f"  -> 最佳候选: {result.selected_candidate_id}")

    def test_full_graph_run(self, real_runner):
        """
        完整流程测试：从 TM → Human Review(自动通过) → SG → Eva。
        这是最重要的测试 — 它验证整个图能跑通。
        """
        state = MacpState(user_request="为我设计一节 30 分钟的 Python 入门课")
        result = real_runner.run(state)

        # 验证关键字段被正确填充
        assert len(result.subtasks) > 0, "完整流程未生成 subtasks"
        assert len(result.candidates) > 0, "完整流程未生成 candidates"
        assert len(result.evaluations) > 0, "完整流程未生成 evaluations"
        assert result.selected_candidate_id is not None, "完整流程未选择候选"
        assert isinstance(result.approved, bool), "approved 字段缺失"

        # 打印摘要
        print(f"\n[完整流程]")
        print(f"  User Request: {result.user_request}")
        print(f"  Subtasks ({len(result.subtasks)}): {result.subtasks}")
        print(f"  Candidates ({len(result.candidates)}): {[c.id for c in result.candidates]}")
        print(f"  Evaluations: {[(e.candidate_id, e.score, e.passed) for e in result.evaluations]}")
        print(f"  Selected: {result.selected_candidate_id}")
        print(f"  Approved: {result.approved}")
        print(f"  Trace: {result.metadata.get('trace', [])}")


# ============================================================
# 测试 2：Human-in-the-Loop 流程
# ============================================================

class TestHumanInTheLoop:

    @pytest.fixture
    def real_runner(self):
        _require_api_key()
        llm = OpenAILLMClient()
        mcp = MockMCPClient()
        tm = TaskManagerAgent(llm_client=llm, mcp_client=mcp)
        sg = SolutionGeneratorAgent(k=2, llm_client=llm, mcp_client=mcp)
        eva = EvaluatorAgent(pass_score=0.75, llm_client=llm, mcp_client=mcp)
        return MacpGraphRunner(tm=tm, sg=sg, eva=eva, checkpoint_path=None)

    def test_run_until_human_review_then_resume(self, real_runner):
        """
        测试 HITL 模式：
        1. 先跑到 human_review 节点停下
        2. 人工审批通过，继续跑完 SG → Eva
        3. 验证最终状态正确
        """
        thread_id = "test-real-hitl"
        state = MacpState(user_request="设计一个 Docker 入门教程")

        # 第一步：跑到 human_review 停止
        result_until_review = real_runner.run_until_human_review(state, thread_id=thread_id)

        # 验证确实停在了 human_review（即结果中包含 interrupt）
        assert "__interrupt__" in result_until_review, (
            "预期停在 human_review 节点，但没有触发 interrupt"
        )
        interrupt_val = result_until_review["__interrupt__"][0].value
        print(f"\n[HITL] 人工审批内容: {interrupt_val}")

        # 第二步：模拟人类审批，加上额外的约束
        resume_review = {
            "approved": True,
            "constraints": ["step-by-step", "适合初学者", "包含实际命令示例"],
            "subtasks": [
                "Docker 基本概念",
                "安装和配置",
                "第一个容器化应用",
            ],
        }
        final_state = real_runner.resume_after_human_review(resume_review, thread_id=thread_id)

        # 验证最终状态
        assert final_state.human_approved is True
        assert "step-by-step" in final_state.constraints
        assert "Docker 基本概念" in final_state.subtasks

        # 验证后续流程走完了
        assert len(final_state.candidates) > 0, "HITL 后 SG 未生成候选"
        assert len(final_state.evaluations) > 0, "HITL 后 Eva 未评估"
        assert final_state.selected_candidate_id is not None, "HITL 后未选择最佳候选"

        print(f"\n[HITL 完整结果]")
        print(f"  Approved: {final_state.approved}")
        print(f"  Best candidate: {final_state.selected_candidate_id}")
        print(f"  Trace: {final_state.metadata.get('trace', [])}")


# ============================================================
# 测试 3：自定义配置 + 国内代理测试
# ============================================================

class TestCustomConfiguration:

    def test_with_custom_base_url(self):
        """
        你可以用这个测试来验证国内代理（如 DeepSeek / 通义千问 / 硅基流动）的连通性。
        设置方式：
            set OPENAI_BASE_URL=https://api.deepseek.com/v1
            set OPENAI_API_KEY=sk-deepseek-key
            pytest tests/test_real_flow.py::TestCustomConfiguration -v
        """
        api_key = _require_api_key()
        base_url = os.environ.get("OPENAI_BASE_URL", "").strip()

        llm = OpenAILLMClient(api_key=api_key, base_url=base_url or None)
        mcp = MockMCPClient()

        tm = TaskManagerAgent(llm_client=llm, mcp_client=mcp)
        sg = SolutionGeneratorAgent(k=2, llm_client=llm, mcp_client=mcp)
        eva = EvaluatorAgent(pass_score=0.75, llm_client=llm, mcp_client=mcp)
        runner = MacpGraphRunner(tm=tm, sg=sg, eva=eva, checkpoint_path=None)

        state = MacpState(user_request="简单测试连通性")
        result = runner.run(state)

        assert result.subtasks, "自定义 base_url 下 TM 未生成子任务"
        print(f"\n[自定义 Base URL] 子任务: {result.subtasks}")
        print(f"  Trace: {result.metadata.get('trace', [])}")
