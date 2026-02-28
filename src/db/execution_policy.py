"""Execution policy checks and guarded DuckDB execution."""

from __future__ import annotations

import multiprocessing as mp
import re
import time
from dataclasses import dataclass
from queue import Empty

import duckdb
import pandas as pd


@dataclass(frozen=True)
class ExecutionPolicy:
    max_rows: int = 1000
    timeout_seconds: int = 15
    max_joins: int = 6
    max_ctes: int = 4
    max_sql_chars: int = 20000


def _estimate_cte_count(sql: str) -> int:
    cleaned = (sql or "").strip().rstrip(";")
    if not cleaned.lower().startswith("with "):
        return 0
    return len(re.findall(r"\b[a-zA-Z_][a-zA-Z0-9_]*\s+as\s*\(", cleaned, re.IGNORECASE))


def validate_sql_complexity(sql: str, policy: ExecutionPolicy) -> tuple[bool, str]:
    """Apply lightweight complexity limits before execution."""
    cleaned = (sql or "").strip()
    if not cleaned:
        return False, "SQL is empty."

    if len(cleaned) > policy.max_sql_chars:
        return (
            False,
            f"Query blocked: SQL length exceeds policy limit ({policy.max_sql_chars} chars).",
        )

    join_count = len(re.findall(r"\bjoin\b", cleaned, re.IGNORECASE))
    if join_count > policy.max_joins:
        return (
            False,
            f"Query blocked: JOIN count ({join_count}) exceeds policy limit ({policy.max_joins}).",
        )

    cte_count = _estimate_cte_count(cleaned)
    if cte_count > policy.max_ctes:
        return (
            False,
            f"Query blocked: CTE count ({cte_count}) exceeds policy limit ({policy.max_ctes}).",
        )

    return True, "OK"


def apply_row_limit(sql: str, max_rows: int) -> str:
    """Wrap query to enforce hard row cap."""
    cleaned = (sql or "").strip().rstrip(";")
    return (
        "SELECT *\n"
        "FROM (\n"
        f"{cleaned}\n"
        ") AS q\n"
        f"LIMIT {int(max_rows)}"
    )


def _query_worker(duckdb_path: str, sql: str, result_queue) -> None:
    con = duckdb.connect(duckdb_path, read_only=True)
    try:
        start = time.time()
        df = con.execute(sql).df()
        elapsed = time.time() - start
        result_queue.put({"ok": True, "df": df, "elapsed": elapsed})
    except Exception as exc:
        result_queue.put({"ok": False, "error": str(exc)})
    finally:
        con.close()


def run_query_with_timeout(
    duckdb_path: str, sql: str, timeout_seconds: int
) -> tuple[pd.DataFrame, float, str]:
    """
    Run query in a worker process and terminate on timeout.
    Returns (df, elapsed_seconds, error_message).
    """
    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    process = ctx.Process(target=_query_worker, args=(duckdb_path, sql, queue))
    process.start()
    process.join(timeout_seconds)

    if process.is_alive():
        process.terminate()
        process.join()
        return pd.DataFrame(), 0.0, f"Query timeout after {timeout_seconds} seconds."

    try:
        payload = queue.get(timeout=2)
    except Empty:
        return pd.DataFrame(), 0.0, "Query failed: no result returned by worker."

    if not payload.get("ok"):
        return pd.DataFrame(), 0.0, payload.get("error", "Unknown query error.")

    return payload["df"], float(payload["elapsed"]), ""

