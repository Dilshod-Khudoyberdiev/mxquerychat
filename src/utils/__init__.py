"""

Purpose:
This package marker file explains src.utils as the shared location for utility helpers such as logging
and telemetry support.

What This File Represents:
- Namespace for cross-cutting utilities.
- Separation of operational concerns from domain logic.

Key Design Invariant:
Utility helpers should remain lightweight, reusable, and independent from UI page concerns.

How Other Modules Use This File:
app.py and tests import telemetry support from this package for consistent event tracking.
"""



