from src.db.data_source import get_active_source_info


def test_get_active_source_info_for_missing_file() -> None:
    info = get_active_source_info("does_not_exist.duckdb")
    assert info["engine"] == "DuckDB"
    assert info["exists"] == "no"
    assert "duckdb" in info["path"].lower()

