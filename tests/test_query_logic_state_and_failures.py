"""

Purpose:
This test module verifies state reset behavior and failure-category normalization helpers in
src/core/query_logic.py.

What This File Contains:
- Session-state reset assertions for New Question flow fields.
- Generation failure-code mapping checks.
- Execution failure-text mapping checks.

Test Guarantees:
- UI state reset logic clears only intended keys and preserves unrelated training state.
- Failure categories remain stable for telemetry aggregation.

Why This Matters:
Consistent state management and category mapping are essential for reliable UX and meaningful metrics.
"""

from src.core.query_logic import (
    classify_execution_failure,
    classify_generation_failure,
    reset_question_flow_state,
)


def test_reset_question_flow_state_resets_expected_keys() -> None:
    state = {
        "question": "Show revenue by month.",
        "generated_sql": "SELECT * FROM t",
        "generated_explanation": "old explanation",
        "explanation_status": "ready",
        "last_result_df": {"rows": [1, 2]},
        "last_result_elapsed": 1.23,
        "suggestions": ["q1"],
        "generation_notes": ["note"],
        "feedback_last_rating": "up",
        "feedback_last_question_hash": "abc123",
        "training_working_df": "keep_me",
    }

    reset_question_flow_state(state)

    assert state["question"] == ""
    assert state["generated_sql"] == ""
    assert state["generated_explanation"] == ""
    assert state["explanation_status"] == "idle"
    assert state["last_result_df"] is None
    assert state["last_result_elapsed"] is None
    assert state["suggestions"] == []
    assert state["generation_notes"] == []
    assert state["feedback_last_rating"] is None
    assert state["feedback_last_question_hash"] == "no_question"
    assert state["training_working_df"] == "keep_me"


def test_classify_generation_failure_mapping() -> None:
    assert classify_generation_failure("timeout") == "timeout"
    assert classify_generation_failure("no_match") == "no_match"
    assert classify_generation_failure("model_error") == "runtime_fail"
    assert classify_generation_failure("anything_else") == "runtime_fail"


def test_classify_execution_failure_mapping() -> None:
    assert classify_execution_failure("Query timeout after 15 seconds.") == "timeout"
    assert classify_execution_failure("Parser Error: syntax error at or near") == "compile_fail"
    assert classify_execution_failure("Binder Error: Referenced column not found") == "compile_fail"
    assert classify_execution_failure("IO Error: disk full") == "runtime_fail"


