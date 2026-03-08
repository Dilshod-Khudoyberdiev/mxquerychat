"""

Purpose:
This test module covers local pre-generation guardrails and deterministic SQL explanation helpers in
src/core/query_logic.py.

What This File Contains:
- Guardrail tests for empty prompts, short prompts, write intent, and off-topic prompts.
- Positive test confirming valid domain questions pass guardrail checks.
- Explanation helper test for expected interpretive clues.

Test Guarantees:
- Local blocking behavior remains stable before model invocation.
- Deterministic explanation text includes key structural hints when SQL contains filters,
  aggregations, grouping, and ordering.

Why This Matters:
These tests keep early-stage user guidance predictable and reduce unnecessary LLM calls.
"""

from src.core.query_logic import explain_sql_brief, get_local_guardrail_message


def test_guardrail_rejects_empty_question() -> None:
    message = get_local_guardrail_message("")
    assert "please enter" in message.lower()


def test_guardrail_rejects_too_short_question() -> None:
    message = get_local_guardrail_message("revenue")
    assert "fuller data question" in message.lower()


def test_guardrail_rejects_write_intent() -> None:
    message = get_local_guardrail_message("Please delete all rows")
    assert "read-only mode" in message.lower()


def test_guardrail_rejects_off_topic_prompt() -> None:
    message = get_local_guardrail_message("Write me a poem")
    assert "off-topic" in message.lower()


def test_guardrail_allows_dataset_question() -> None:
    message = get_local_guardrail_message("Show revenue by state for 2025")
    assert message == ""


def test_explain_sql_brief_has_expected_clues() -> None:
    sql = """
    SELECT tv.jahr, SUM(tv.umsatz_eur) AS umsatz_eur
    FROM ticket_verkaeufe tv
    WHERE tv.jahr = 2025
    GROUP BY tv.jahr
    ORDER BY tv.jahr
    """
    explanation = explain_sql_brief(sql)
    assert "ticket_verkaeufe" in explanation
    assert "filters" in explanation.lower()
    assert "sum" in explanation.lower()
    assert "grouped" in explanation.lower()
    assert "sorted" in explanation.lower()



