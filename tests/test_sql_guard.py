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

