from app.graph.main_graph import MacpGraphRunner
from app.graph.state import MacpState


def test_graph_runner_reaches_terminal_state() -> None:
    state = MacpState(user_request="Build a machine learning intro class")
    runner = MacpGraphRunner(checkpoint_path=None)

    result = runner.run(state)

    assert result.selected_candidate_id is not None
    assert len(result.evaluations) > 0
    assert isinstance(result.approved, bool)
    assert result.metadata["trace"][0] == "TM"
    assert result.metadata.get("memory_persisted") is True
    assert "memory_hits" in result.metadata


def test_human_review_resume_path() -> None:
    runner = MacpGraphRunner(checkpoint_path=None)
    thread_id = "test-hitl"
    initial = MacpState(user_request="Create a beginner AI syllabus")

    paused = runner.run_until_human_review(initial, thread_id=thread_id)
    assert "__interrupt__" in paused

    resumed = runner.resume_after_human_review(
        {
            "approved": True,
            "constraints": ["no formulas", "story-driven examples"],
            "subtasks": ["outline", "examples", "final lesson"],
        },
        thread_id=thread_id,
    )
    assert resumed.human_approved is True
    assert "no formulas" in resumed.constraints
