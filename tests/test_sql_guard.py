"""

Purpose:
This test module validates the read-only SQL guard behavior implemented in sql_guard.py. The tests
confirm that allowed SELECT/WITH patterns pass while unsafe statement forms are rejected.

What This File Contains:
- Parametrized positive and negative cases for basic read-only acceptance.
- Focused edge-case checks for semicolon misuse and embedded dangerous patterns.

Test Guarantees:
- Multi-statement and write-intent SQL does not pass validation.
- Benign text that incidentally contains similar substrings is not falsely blocked.

Why This Matters:
These tests preserve the foundational safety contract of mxQueryChat: no write or DDL execution paths
should proceed through the runtime pipeline.
"""

import pytest

from sql_guard import validate_read_only_sql


@pytest.mark.parametrize(
    ("sql", "expected_ok"),
    [
        ("SELECT 1", True),
        ("WITH cte AS (SELECT 1) SELECT * FROM cte", True),
        ("", False),
        ("SELECT 1; SELECT 2;", False),
        ("SELECT 1; SELECT 2", False),
        ("INSERT INTO x VALUES (1)", False),
        ("SELECT dropdown_value FROM t", True),
    ],
)
def test_validate_read_only_sql(sql: str, expected_ok: bool) -> None:
    is_ok, _ = validate_read_only_sql(sql)
    assert is_ok is expected_ok


def test_validate_read_only_sql_blocks_mid_query_semicolon() -> None:
    is_ok, message = validate_read_only_sql("SELECT 1; -- trailing comment then text")
    assert is_ok is False
    assert "semicolon" in message.lower()


def test_validate_read_only_sql_blocks_keyword_inside_select() -> None:
    is_ok, message = validate_read_only_sql("SELECT * FROM a; DROP TABLE a;")
    assert is_ok is False
    assert "multiple sql statements" in message.lower() or "drop" in message.lower()



