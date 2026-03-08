"""

Purpose:
This package marker file documents src.llm as the location for optional language-model helpers that
enhance UX without changing execution safety behavior.

What This File Represents:
- Namespace for explanation-oriented LLM helpers.
- Clear modular boundary between SQL execution and explanatory text generation.

Key Design Invariant:
LLM explanation features remain optional and must not block query execution workflows.

How Other Modules Use This File:
app.py imports sql_explainer functionality through this package when explanation is explicitly requested.
"""



