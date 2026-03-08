"""

Purpose:
This package marker file documents the role of the src package as the modular home for reusable
application logic. It separates core behavior from the Streamlit UI shell.

What This File Represents:
- The top-level namespace for core, db, llm, and utils subpackages.
- A stable import anchor for internal modules.

Key Design Invariant:
Application behavior should be implemented in reusable src modules first, then orchestrated by app.py.

How Other Modules Use This File:
Python uses this file to treat src as an importable package in app runtime, test execution, and tooling.
"""



