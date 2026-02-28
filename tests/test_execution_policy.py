from pathlib import Path

import duckdb

from src.db.execution_policy import (
    ExecutionPolicy,
    apply_row_limit,
    extract_complexity_policy_details,
    run_query_with_timeout,
    validate_sql_complexity,
)


def test_validate_sql_complexity_ok() -> None:
    policy = ExecutionPolicy(max_sql_chars=200, max_joins=3, max_ctes=2)
    ok, message = validate_sql_complexity("SELECT 1", policy)
    assert ok is True
    assert message == "OK"


def test_validate_sql_complexity_blocks_too_many_joins() -> None:
    policy = ExecutionPolicy(max_joins=1)
    sql = "SELECT * FROM a JOIN b ON a.id = b.id JOIN c ON b.id = c.id"
    ok, message = validate_sql_complexity(sql, policy)
    assert ok is False
    assert "join count" in message.lower()


def test_apply_row_limit_wraps_query() -> None:
    limited = apply_row_limit("SELECT * FROM ticket_verkaeufe", 1000)
    assert "FROM (" in limited
    assert "LIMIT 1000" in limited


def test_extract_complexity_policy_details_parses_threshold() -> None:
    reason, threshold = extract_complexity_policy_details(
        "Query blocked: JOIN count (7) exceeds policy limit (6)."
    )
    assert reason == "max_joins"
    assert threshold == 6


def test_run_query_with_timeout_success(tmp_path: Path) -> None:
    db_path = tmp_path / "test.duckdb"
    con = duckdb.connect(str(db_path))
    con.execute("CREATE TABLE t(a INTEGER)")
    con.execute("INSERT INTO t VALUES (1), (2), (3)")
    con.close()

    df, elapsed, error = run_query_with_timeout(str(db_path), "SELECT * FROM t", 5)
    assert error == ""
    assert len(df) == 3
    assert elapsed >= 0


def test_run_query_with_timeout_query_error(tmp_path: Path) -> None:
    db_path = tmp_path / "test.duckdb"
    con = duckdb.connect(str(db_path))
    con.execute("CREATE TABLE t(a INTEGER)")
    con.close()

    df, elapsed, error = run_query_with_timeout(str(db_path), "SELECT * FROM missing_table", 5)
    assert len(df) == 0
    assert elapsed == 0
    assert "missing_table" in error.lower() or "not found" in error.lower()
