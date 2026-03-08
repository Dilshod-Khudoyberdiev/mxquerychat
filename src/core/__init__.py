"""

Purpose:
This package marker file explains that src.core contains the central decision logic for question
interpretation, SQL planning, retries, and failure normalization.

What This File Represents:
- Namespace boundary for query planning logic.
- Clear separation between orchestration logic and UI code.

Key Design Invariant:
Core decision paths remain import-safe and testable without Streamlit dependencies.

How Other Modules Use This File:
app.py, benchmark scripts, and evaluation scripts import src.core modules through this package.
"""



