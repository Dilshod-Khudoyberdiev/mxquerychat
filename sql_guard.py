"""Read-only SQL guard: rejects anything that is not a plain SELECT or WITH statement."""

import re

BLOCKED_KEYWORDS = [
    "insert",
    "update",
    "delete",
    "merge",
    "replace",
    "create",
    "alter",
    "drop",
    "truncate",
    "grant",
    "revoke",
    "attach",
    "detach",
    "copy",
    "export",
    "import",
    "call",
    "execute",
    "pragma",
    "recursive",  # prevents WITH RECURSIVE runaway CTEs
]


def validate_read_only_sql(sql: str) -> tuple[bool, str]:
    """
    Returns: (is_allowed, message)
    """
    if not sql or not sql.strip():
        return False, "SQL is empty."

    cleaned = sql.strip()

    # 1) Block multiple statements (very common injection pattern)
    # Allow a trailing semicolon but disallow multiple.
    if cleaned.count(";") > 1:
        return False, "Blocked: multiple SQL statements are not allowed."

    # If one semicolon exists, only allow it at the end
    if ";" in cleaned and not cleaned.rstrip().endswith(";"):
        return False, "Blocked: semicolon must be only at the end."

    # Remove trailing semicolon for checks
    cleaned_no_semicolon = cleaned.rstrip().rstrip(";").strip()

    # 2) Must start with SELECT or WITH
    first_word = cleaned_no_semicolon.split(None, 1)[0].lower()
    if first_word not in ("select", "with"):
        return False, "Blocked: only SELECT/WITH queries are allowed."

    # 3) Block dangerous keywords anywhere (simple substring match)
    lowered = cleaned_no_semicolon.lower()
    for kw in BLOCKED_KEYWORDS:
        # use word boundary so "dropdown" doesn't match "drop"
        if re.search(rf"\b{re.escape(kw)}\b", lowered):
            return (
                False,
                f"Blocked: keyword '{kw.upper()}' is not allowed (read-only mode).",
            )

    return True, "OK"



