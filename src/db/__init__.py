"""

Purpose:
This package marker file describes src.db as the home for database-facing helpers, including source
metadata operations and guarded execution policy logic.

What This File Represents:
- Namespace for DuckDB execution controls.
- Separation between policy enforcement and generation logic.

Key Design Invariant:
All runtime database execution should pass through explicit policy checks and timeout boundaries.

How Other Modules Use This File:
app.py and tools import modules from src.db to enforce the same database behavior across workflows.
"""



