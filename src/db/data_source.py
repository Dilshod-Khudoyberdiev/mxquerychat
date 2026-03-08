"""

Purpose:
This module provides lightweight helpers for DuckDB source metadata and cache-refresh operations used
by the Streamlit sidebar controls.

What This File Contains:
- A metadata function that reports engine, path, existence, and size of the active DuckDB file.
- Cache refresh helpers that clear Streamlit cached data to force fresh schema/data reads.

Key Invariants and Safety Guarantees:
- Source metadata is read-only and does not mutate the database.
- Cache invalidation is explicit and predictable through dedicated helper calls.

How Other Modules Use This File:
app.py imports these helpers for the Data Source panel, including Reload Dataset and Refresh Schema
buttons that support operational control during demos and troubleshooting.
"""

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



