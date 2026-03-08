"""

Purpose:
This test module validates metadata behavior for src/db/data_source.py when a DuckDB file path does
not exist.

What This File Contains:
- A focused assertion that engine name, existence flag, and path formatting are returned correctly.

Test Guarantees:
- Missing-file states are represented cleanly without crashing metadata calls.

Why This Matters:
Clear missing-source diagnostics improve supportability during setup and deployment checks.
"""

from src.db.data_source import get_active_source_info


def test_get_active_source_info_for_missing_file() -> None:
    info = get_active_source_info("does_not_exist.duckdb")
    assert info["engine"] == "DuckDB"
    assert info["exists"] == "no"
    assert "duckdb" in info["path"].lower()



