"""DuckDB data source helpers (MVP scope: DuckDB only)."""

from __future__ import annotations

from pathlib import Path

import streamlit as st


def get_active_source_info(duckdb_path: str) -> dict[str, str]:
    """Return small metadata for the active DuckDB source."""
    path = Path(duckdb_path)
    exists = path.exists()
    size_bytes = path.stat().st_size if exists else 0
    size_mb = round(size_bytes / (1024 * 1024), 2) if exists else 0.0
    return {
        "engine": "DuckDB",
        "path": str(path.resolve()) if exists else str(path),
        "exists": "yes" if exists else "no",
        "size_mb": f"{size_mb}",
    }


def refresh_schema_cache() -> None:
    """
    Refresh schema-related cache entries.
    Current implementation clears Streamlit cached data for a clean reload.
    """
    st.cache_data.clear()


def reload_dataset_cache() -> None:
    """
    Reload dataset-backed caches.
    Current implementation clears Streamlit cached data.
    """
    st.cache_data.clear()

