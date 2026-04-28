"""
给同学看的总入口测试（简单版）。

目标：5 秒内看懂这个项目主流程能不能跑通。
"""

from app.graph.main_graph import MacpGraphRunner
from app.graph.state import MacpState


def test_end_to_end_pipeline_smoke() -> None:
    runner = MacpGraphRunner(checkpoint_path=None)
    state = MacpState(user_request="给零基础同学准备一节 Python 入门课")

    result = runner.run(state)

    assert result.human_approved is True
    assert len(result.subtasks) > 0
    assert len(result.candidates) > 0
    assert len(result.evaluations) > 0
    assert result.selected_candidate_id is not None
    if result.approved:
        assert result.metadata.get("memory_persisted") is True
    else:
        assert result.reflection_count >= result.max_reflections
    assert result.metadata.get("trace", [])[0] == "TM"