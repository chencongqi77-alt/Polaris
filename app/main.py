from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional

from dotenv import load_dotenv

from app.graph.main_graph import MacpGraphRunner
from app.graph.state import MacpState


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="macp",
        description="Run the MACP pipeline (TM -> SG -> Eva -> Memory).",
    )
    p.add_argument(
        "user_request",
        nargs="?",
        help="The user's request/prompt. If omitted, reads from stdin.",
    )
    p.add_argument("--thread-id", default="local-thread", help="LangGraph thread_id.")
    p.add_argument(
        "--checkpoint-path",
        default="app_data/checkpoints/macp.sqlite",
        help="SQLite checkpoint path. Use 'none' to disable persistence.",
    )
    p.add_argument(
        "--memory-mode",
        default="qdrant",
        choices=["qdrant"],
        help="Memory backend.",
    )
    p.add_argument(
        "--mcp-mode",
        default="direct",
        choices=["direct", "stdio", "http"],
        help="MCP client mode for tool calls.",
    )
    p.add_argument(
        "--no-auto-approve",
        action="store_true",
        help="Require manual human review (will stop at interrupt).",
    )
    p.add_argument(
        "--resume-review-json",
        help=(
            "Resume a paused human review using a JSON payload, for example "
            '\'{"approved": true, "subtasks": ["..."], "constraints": ["..."]}\''
        ),
    )
    p.add_argument(
        "--resume-review-file",
        help="Resume a paused human review using a JSON file.",
    )
    p.add_argument(
        "--json",
        action="store_true",
        help="Print full final state as JSON.",
    )
    p.add_argument(
        "--max-reflections",
        type=int,
        default=3,
        help="Max reflection loops.",
    )
    return p


def _read_user_request(arg_value: Optional[str]) -> str:
    if arg_value and arg_value.strip():
        return arg_value.strip()
    text = sys.stdin.read().strip()
    if not text:
        raise SystemExit("Missing user_request. Provide as argument or via stdin.")
    return text


def _load_review_payload(review_json: Optional[str], review_file: Optional[str]) -> Optional[Dict[str, Any]]:
    if review_json and review_file:
        raise SystemExit("Use only one of --resume-review-json or --resume-review-file.")

    raw_payload: Optional[str] = None
    if review_json:
        raw_payload = review_json
    elif review_file:
        raw_payload = Path(review_file).read_text(encoding="utf-8")
    else:
        return None

    try:
        payload = json.loads(raw_payload)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Invalid human review JSON: {exc}") from exc

    if not isinstance(payload, dict):
        raise SystemExit("Human review payload must be a JSON object.")
    if "approved" not in payload:
        raise SystemExit("Human review payload must include an 'approved' field.")
    return payload


def _prompt_edit_mode() -> str:
    """Ask user to choose between feedback mode (default) or rewrite mode."""
    while True:
        mode = input(
            "\nEdit mode: [f]eedback (default) / [r]ewrite all: "
        ).strip().lower()
        if mode in {"f", "feedback", ""}:
            return "feedback"
        if mode in {"r", "rewrite"}:
            return "rewrite"
        print("Please enter 'f' for feedback or 'r' for rewrite.")


def _prompt_feedback_for_items(label: str, items: list[str]) -> tuple[list[str], list[str]]:
    """Collect per-item feedback while keeping the original list unchanged.

    Returns (original_items, feedback_per_item).
    """
    print(f"\nCurrent {label}:")
    if items:
        for idx, item in enumerate(items, start=1):
            print(f"  {idx}. {item}")
    else:
        print("  (empty)")
        return items, []

    print(f"\nEnter feedback for each {label[:-1]} (press Enter to skip, type feedback to add):")
    feedbacks: list[str] = []
    for idx, item in enumerate(items, start=1):
        short = (item[:60] + "...") if len(item) > 60 else item
        fb = input(f"  {idx}. {short}\n     [skip/your feedback]: ").strip()
        feedbacks.append(fb)
    return items, feedbacks


def _prompt_rewrite_items(label: str, items: Any) -> list[str]:
    """Completely replace items via user input (original behavior)."""
    normalized = [str(item).strip() for item in items if str(item).strip()] if isinstance(items, list) else []
    print(f"\nCurrent {label}:")
    if normalized:
        for idx, item in enumerate(normalized, start=1):
            print(f"  {idx}. {item}")
    else:
        print("  (empty)")

    print(f"\n⚠️  Enter ALL {label} to replace the current list.")
    print(f"Enter one {label[:-1]} per line. Submit an empty line to finish.")
    updated: list[str] = []
    while True:
        line = input("> ").strip()
        if not line:
            break
        updated.append(line)
    return updated if updated else normalized


def _prompt_review_action() -> str:
    while True:
        action = input("\nChoose action: [a]pprove / [r]evise / [q]uit: ").strip().lower()
        if action in {"a", "approve"}:
            return "approve"
        if action in {"r", "revise"}:
            return "revise"
        if action in {"q", "quit"}:
            return "quit"
        print("Please enter 'a', 'r', or 'q'.")


def _print_review_payload(payload: Dict[str, Any]) -> None:
    request_id = payload.get("request_id")
    if request_id:
        print(f"request_id: {request_id}")

    message = payload.get("message")
    if message:
        print(str(message))

    subtasks = payload.get("subtasks", [])
    print("\nTM subtasks:")
    if isinstance(subtasks, list) and subtasks:
        for idx, item in enumerate(subtasks, start=1):
            print(f"  {idx}. {item}")
    else:
        print("  (empty)")

    constraints = payload.get("constraints", [])
    print("\nConstraints:")
    if isinstance(constraints, list) and constraints:
        for idx, item in enumerate(constraints, start=1):
            print(f"  {idx}. {item}")
    else:
        print("  (empty)")


def _collect_interactive_review(payload: Dict[str, Any]) -> Dict[str, Any]:
    print("=== HUMAN REVIEW REQUIRED ===")
    _print_review_payload(payload)
    action = _prompt_review_action()
    if action == "quit":
        raise SystemExit(2)

    subtasks = [str(item).strip() for item in payload.get("subtasks", []) if str(item).strip()]
    constraints = [str(item).strip() for item in payload.get("constraints", []) if str(item).strip()]
    approved = action == "approve"

    if not approved:
        # Ask user which edit mode they prefer (feedback is the default)
        mode = _prompt_edit_mode()

        if mode == "feedback":
            # Feedback mode: keep original subtasks/constraints, collect per-item notes
            subtasks, subtask_feedbacks = _prompt_feedback_for_items("subtasks", subtasks)
            constraints, constraint_feedbacks = _prompt_feedback_for_items("constraints", constraints)

            notes = input("\nAdditional notes for SG/TM (press Enter to skip): ").strip()

            review: Dict[str, Any] = {
                "approved": False,
                "subtasks": subtasks,
                "constraints": constraints,
                "feedback": {
                    "subtask_feedback": subtask_feedbacks,
                    "constraint_feedback": constraint_feedbacks,
                },
            }
            if notes:
                review["notes"] = notes
            return review
        else:
            # Rewrite mode: replace subtasks/constraints entirely
            subtasks = _prompt_rewrite_items("subtasks", subtasks)
            constraints = _prompt_rewrite_items("constraints", constraints)

    note_prompt = "Optional approval note (press Enter to skip): " if approved else "Revision note for SG/TM: "
    notes = input(note_prompt).strip()

    review = {"approved": approved, "subtasks": subtasks, "constraints": constraints}
    if notes:
        review["notes"] = notes
    return review


def _print_human_interrupt(payload: Dict[str, Any]) -> None:
    print("=== HUMAN REVIEW REQUIRED ===")
    # Filter out non-serializable internal keys (e.g., __interrupt__)
    safe_payload = {
        k: v for k, v in payload.items()
        if not k.startswith("__") and isinstance(v, (str, int, float, bool, list, dict, type(None)))
    }
    print(json.dumps(safe_payload, ensure_ascii=False, indent=2))
    print(
        "\nTo continue, re-run with the same --thread-id and one of:\n"
        "  --resume-review-json '{\"approved\": true}'\n"
        "  --resume-review-file path/to/review.json"
    )


def _print_summary(state: MacpState) -> None:
    print("=== MACP RESULT ===")
    print(f"request_id: {state.request_id}")
    print(f"approved: {state.approved}")
    print(f"reflection_count: {state.reflection_count}/{state.max_reflections}")
    print(f"subtasks: {len(state.subtasks)}")
    for i, task in enumerate(state.subtasks, start=1):
        print(f"  {i}. {task}")
    print(f"candidates: {len(state.candidates)}")
    print(f"evaluations: {len(state.evaluations)}")
    print(f"selected_candidate_id: {state.selected_candidate_id}")

    if state.selected_candidate_id:
        selected = next((c for c in state.candidates if c.id == state.selected_candidate_id), None)
        if selected is not None:
            print("\n--- SELECTED CONTENT ---")
            print(selected.content)


def main(argv: Optional[list[str]] = None) -> int:
    # Load environment variables from .env file
    load_dotenv()

    # Best-effort UTF-8 output for Windows consoles.
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            try:
                reconfigure(encoding="utf-8")
            except Exception:  # noqa: BLE001
                pass

    args = _build_parser().parse_args(argv)
    checkpoint_path = None if str(args.checkpoint_path).lower() in {"none", "null", ""} else args.checkpoint_path
    review_payload = _load_review_payload(args.resume_review_json, args.resume_review_file)

    runner = MacpGraphRunner(
        memory_mode=args.memory_mode,
        mcp_mode=args.mcp_mode,
        checkpoint_path=checkpoint_path,
    )
    try:
        if review_payload is not None:
            result = runner.resume_after_human_review(
                review=review_payload,
                thread_id=args.thread_id,
            )
        else:
            user_request = _read_user_request(args.user_request)
            state = MacpState(user_request=user_request, max_reflections=args.max_reflections)
            result = runner.run(
                state,
                thread_id=args.thread_id,
                auto_approve_human_review=not args.no_auto_approve,
            )

        if args.json:
            print(json.dumps(result.model_dump(), ensure_ascii=False, indent=2))
        else:
            _print_summary(result)
        return 0
    except RuntimeError as exc:
        # Human review required path (when --no-auto-approve and interrupt triggered)
        msg = str(exc)
        if "Human review required" in msg:
            user_request = _read_user_request(args.user_request)
            payload = runner.run_until_human_review(
                MacpState(user_request=user_request, max_reflections=args.max_reflections),
                thread_id=args.thread_id,
            )
            if args.no_auto_approve:
                review_payload = _collect_interactive_review(payload)
                result = runner.resume_after_human_review(
                    review=review_payload,
                    thread_id=args.thread_id,
                )
                if args.json:
                    print(json.dumps(result.model_dump(), ensure_ascii=False, indent=2))
                else:
                    _print_summary(result)
                return 0
            _print_human_interrupt(payload)
            return 2
        raise
    finally:
        runner.close()


if __name__ == "__main__":
    raise SystemExit(main())

