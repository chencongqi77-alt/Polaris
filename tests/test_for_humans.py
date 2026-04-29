"""
给同学看的总入口测试（真实调用版）。

目标：验证 MCP + Qdrant + LLM 全链路是否能跑通。

前置条件：
- .env 中配置了 OPENAI_API_KEY、OPENAI_BASE_URL
- Qdrant 服务运行在 localhost:6333
- docker run -d --name qdrant -p 6333:6333 qdrant/qdrant

运行方法：
    pytest tests/test_for_humans.py -v -s
"""

import os
import pytest
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

from app.graph.main_graph import MacpGraphRunner
from app.graph.state import MacpState


@pytest.fixture
def full_runner():
    """真实全链路 Runner（LLM + MCP + Qdrant）"""
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key or api_key == "sk-你的阿里云百炼APIKey":
        pytest.skip("OPENAI_API_KEY 未设置或为默认值，跳过集成测试")

    # 检查 Qdrant 是否可用
    from app.memory import create_memory_store
    store = create_memory_store(mode="qdrant")
    if store is None:
        pytest.skip("Qdrant 服务不可用，跳过集成测试")

    # 使用 MacpGraphRunner 的内置构造，它会自动创建 agents
    return MacpGraphRunner(
        memory_mode="qdrant",
        mcp_mode="direct",
        checkpoint_path=None,
    )


def test_end_to_end_pipeline_smoke(full_runner: MacpGraphRunner) -> None:
    """
    真实全链路冒烟测试：TM → SG → Eva → Memory
    验证整个 pipeline 是否能跑通。
    """
    state = MacpState(user_request="给零基础同学准备一节 Python 入门课")

    result = full_runner.run(state)

    # TM 应该生成了子任务
    assert len(result.subtasks) > 0, f"TM 没有生成子任务: subtasks={result.subtasks}"

    # SG 应该生成了候选方案
    assert len(result.candidates) > 0, f"SG 没有生成候选方案: candidates={result.candidates}"

    # Eva 应该评估了候选方案
    assert len(result.evaluations) > 0, f"Eva 没有评估结果: evaluations={result.evaluations}"

    # 应该选出了最佳候选
    assert result.selected_candidate_id is not None, "没有选出最佳候选"

    # 如果通过了评估，Memory 应该持久化
    if result.approved:
        assert result.extra_metadata.get("memory_persisted") is True, "通过评估但 Memory 未持久化"
    else:
        # 如果没通过，应该是达到了反思上限
        assert result.reflection_count >= result.max_reflections

    # Trace 应该以 TM 开头
    assert result.extra_metadata.get("trace", [])[0] == "TM"
