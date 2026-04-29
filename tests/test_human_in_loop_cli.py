"""Tests for CLI human-in-the-loop helpers."""

from __future__ import annotations

import json

import pytest

from app.main import _collect_interactive_review, _load_review_payload


def test_load_review_payload_from_inline_json() -> None:
    payload = _load_review_payload(
        review_json='{"approved": true, "subtasks": ["revise outline"]}',
        review_file=None,
    )

    assert payload == {"approved": True, "subtasks": ["revise outline"]}


def test_load_review_payload_from_file(tmp_path) -> None:
    review_file = tmp_path / "review.json"
    review_file.write_text(
        json.dumps({"approved": False, "constraints": ["no formulas"]}),
        encoding="utf-8",
    )

    payload = _load_review_payload(review_json=None, review_file=str(review_file))

    assert payload == {"approved": False, "constraints": ["no formulas"]}


def test_load_review_payload_requires_approved_field() -> None:
    with pytest.raises(SystemExit, match="must include an 'approved' field"):
        _load_review_payload(review_json='{"subtasks": ["missing approval"]}', review_file=None)


def test_load_review_payload_rejects_multiple_sources() -> None:
    with pytest.raises(SystemExit, match="Use only one"):
        _load_review_payload(
            review_json='{"approved": true}',
            review_file="review.json",
        )


def test_collect_interactive_review_keeps_existing_items(monkeypatch) -> None:
    answers = iter(["a", ""])
    monkeypatch.setattr("builtins.input", lambda _: next(answers))

    payload = {
        "message": "Review TM plan",
        "subtasks": ["task a", "task b"],
        "constraints": ["clear output"],
    }
    review = _collect_interactive_review(payload)

    assert review == {
        "approved": True,
        "subtasks": ["task a", "task b"],
        "constraints": ["clear output"],
    }


def test_collect_interactive_review_prints_plan_before_prompt(monkeypatch, capsys) -> None:
    answers = iter(["a", ""])
    monkeypatch.setattr("builtins.input", lambda _: next(answers))

    _collect_interactive_review(
        {
            "request_id": "demo-1",
            "message": "Review TM plan",
            "subtasks": ["task a", "task b"],
            "constraints": ["clear output"],
        }
    )

    output = capsys.readouterr().out
    assert "request_id: demo-1" in output
    assert "TM subtasks:" in output
    assert "1. task a" in output
    assert "Constraints:" in output
    assert "1. clear output" in output


def test_collect_interactive_review_allows_rewriting_lists(monkeypatch) -> None:
    answers = iter([
        "r",        # revise action
        "r",        # rewrite mode
        "new task 1",
        "new task 2",
        "",
        "constraint 1",
        "",
        "Need one more revision",
    ])
    monkeypatch.setattr("builtins.input", lambda _: next(answers))

    review = _collect_interactive_review(
        {"subtasks": ["old task"], "constraints": ["old constraint"]}
    )

    assert review == {
        "approved": False,
        "subtasks": ["new task 1", "new task 2"],
        "constraints": ["constraint 1"],
        "notes": "Need one more revision",
    }


def test_collect_interactive_review_feedback_mode(monkeypatch) -> None:
    """Feedback mode collects per-item notes without changing the list."""
    answers = iter([
        "r",            # revise action
        "",             # feedback mode (default, press Enter)
        "too vague",    # feedback for subtask 1
        "",             # skip feedback for constraint 1
        "Please be more specific",
    ])
    monkeypatch.setattr("builtins.input", lambda _: next(answers))

    review = _collect_interactive_review(
        {"subtasks": ["old task"], "constraints": ["old constraint"]}
    )

    assert review == {
        "approved": False,
        "subtasks": ["old task"],
        "constraints": ["old constraint"],
        "feedback": {
            "subtask_feedback": ["too vague"],
            "constraint_feedback": [""],
        },
        "notes": "Please be more specific",
    }


def test_collect_interactive_review_can_quit(monkeypatch) -> None:
    answers = iter(["q"])
    monkeypatch.setattr("builtins.input", lambda _: next(answers))

    with pytest.raises(SystemExit) as exc:
        _collect_interactive_review({"subtasks": ["old task"], "constraints": ["old constraint"]})

    assert exc.value.code == 2
