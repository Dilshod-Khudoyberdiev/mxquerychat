from src.core.query_logic import run_query_if_read_only
from sql_guard import validate_read_only_sql


def test_run_query_if_read_only_blocks_unsafe_sql() -> None:
    called = {"count": 0}

    def run_fn(_sql: str):
        called["count"] += 1
        return "should not run"

    is_allowed, result, message = run_query_if_read_only(
        "INSERT INTO x VALUES (1)",
        validate_read_only_sql,
        run_fn,
    )

    assert is_allowed is False
    assert result is None
    assert called["count"] == 0
    assert "blocked" in message.lower() or "only select/with" in message.lower()


def test_run_query_if_read_only_executes_safe_sql() -> None:
    called = {"count": 0}

    def run_fn(sql: str):
        called["count"] += 1
        return {"sql": sql, "rows": 3}

    is_allowed, result, message = run_query_if_read_only(
        "SELECT 1",
        validate_read_only_sql,
        run_fn,
    )

    assert is_allowed is True
    assert called["count"] == 1
    assert result == {"sql": "SELECT 1", "rows": 3}
    assert message == "OK"

