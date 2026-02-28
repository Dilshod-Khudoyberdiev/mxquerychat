"""Import-safe query planning, fallback, and safety helpers."""

from __future__ import annotations

import re
from typing import Any, Callable, MutableMapping, Optional

OFF_TOPIC_PATTERNS = [
    r"\bpoem\b",
    r"\bsong\b",
    r"\blyrics\b",
    r"\bjoke\b",
    r"\bweather\b",
    r"\bhomework\b",
    r"\bessay\b",
]

WRITE_PATTERNS = [
    r"\binsert\b",
    r"\bupdate\b",
    r"\bdelete\b",
    r"\bdrop\b",
    r"\balter\b",
    r"\btruncate\b",
]


def extract_requested_years(question: str) -> list[int]:
    return [int(y) for y in re.findall(r"\b(19\d{2}|20\d{2})\b", question or "")]


def extract_requested_year(question: str) -> Optional[int]:
    years = extract_requested_years(question)
    if not years:
        return None
    return years[0]


def contains_any(text: str, terms: list[str]) -> bool:
    q = (text or "").lower()
    return any(term in q for term in terms)


def build_year_filter(question: str, alias: str = "tv") -> str:
    requested_year = extract_requested_year(question)
    if requested_year is None:
        return ""
    return f"WHERE {alias}.jahr = {requested_year}"


def build_template_sql(question: str) -> tuple[str, str]:
    """
    Deterministic planner for common analytics questions.
    Returns (sql, note). Empty sql means no template matched.
    """
    q = (question or "").lower()
    wants_revenue = contains_any(q, ["revenue", "umsatz", "sales"])
    wants_state = contains_any(q, ["state", "federal state", "bundesland", "states"])
    wants_month = contains_any(q, ["month", "monat", "monthly"])
    wants_ticket_type = contains_any(
        q, ["ticket type", "ticket types", "ticketart", "ticketarten", "ticket product"]
    )
    wants_tariff = contains_any(
        q,
        [
            "tariff",
            "tariff association",
            "tariff associations",
            "tarif",
            "tarifverbund",
            "tarifverbunde",
            "tarifverbuende",
            "verbund",
            "association",
        ],
    )
    wants_total = contains_any(q, ["total", "gesamt", "overall"])

    # Revenue by tariff association + state + month
    if wants_revenue and wants_tariff and wants_state and wants_month:
        sql = f"""
SELECT
    tv.jahr,
    tv.monat,
    t.name AS tarifverbund_name,
    rb.bundesland_name,
    SUM(tv.umsatz_eur) AS umsatz_eur
FROM ticket_verkaeufe tv
JOIN tarifverbuende t
    ON tv.tarifverbund_id = t.tarifverbund_id
JOIN postleitzahlen p
    ON CAST(tv.plz AS VARCHAR) = p.plz
JOIN regionen_bundesland rb
    ON p.bundesland_code2 = rb.bundesland_code2
{build_year_filter(question, alias="tv")}
GROUP BY tv.jahr, tv.monat, t.name, rb.bundesland_name
ORDER BY tv.jahr, tv.monat, t.name, rb.bundesland_name
""".strip()
        return (
            sql,
            "Template planner used: revenue by tariff association, state, and month "
            "(ticket_verkaeufe -> tarifverbuende + postleitzahlen -> regionen_bundesland).",
        )

    # Revenue by tariff association + state
    if wants_revenue and wants_tariff and wants_state:
        sql = f"""
SELECT
    tv.jahr,
    t.name AS tarifverbund_name,
    rb.bundesland_name,
    SUM(tv.umsatz_eur) AS umsatz_eur
FROM ticket_verkaeufe tv
JOIN tarifverbuende t
    ON tv.tarifverbund_id = t.tarifverbund_id
JOIN postleitzahlen p
    ON CAST(tv.plz AS VARCHAR) = p.plz
JOIN regionen_bundesland rb
    ON p.bundesland_code2 = rb.bundesland_code2
{build_year_filter(question, alias="tv")}
GROUP BY tv.jahr, t.name, rb.bundesland_name
ORDER BY tv.jahr, t.name, umsatz_eur DESC
""".strip()
        return (
            sql,
            "Template planner used: revenue by tariff association and state "
            "(ticket_verkaeufe -> tarifverbuende + postleitzahlen -> regionen_bundesland).",
        )

    # Revenue by tariff association + month
    if wants_revenue and wants_tariff and wants_month:
        sql = f"""
SELECT
    tv.jahr,
    tv.monat,
    t.name AS tarifverbund_name,
    SUM(tv.umsatz_eur) AS umsatz_eur
FROM ticket_verkaeufe tv
JOIN tarifverbuende t
    ON tv.tarifverbund_id = t.tarifverbund_id
{build_year_filter(question, alias="tv")}
GROUP BY tv.jahr, tv.monat, t.name
ORDER BY tv.jahr, tv.monat, umsatz_eur DESC
""".strip()
        return (
            sql,
            "Template planner used: revenue by tariff association and month "
            "(ticket_verkaeufe -> tarifverbuende).",
        )

    # Revenue by tariff association
    if wants_revenue and wants_tariff:
        sql = f"""
SELECT
    tv.jahr,
    t.name AS tarifverbund_name,
    SUM(tv.umsatz_eur) AS umsatz_eur
FROM ticket_verkaeufe tv
JOIN tarifverbuende t
    ON tv.tarifverbund_id = t.tarifverbund_id
{build_year_filter(question, alias="tv")}
GROUP BY tv.jahr, t.name
ORDER BY tv.jahr, umsatz_eur DESC
""".strip()
        return (
            sql,
            "Template planner used: revenue by tariff association "
            "(ticket_verkaeufe -> tarifverbuende).",
        )

    # Revenue by state + month
    if wants_revenue and wants_state and wants_month:
        sql = f"""
SELECT
    tv.jahr,
    tv.monat,
    rb.bundesland_name,
    SUM(tv.umsatz_eur) AS umsatz_eur
FROM ticket_verkaeufe tv
JOIN postleitzahlen p
    ON CAST(tv.plz AS VARCHAR) = p.plz
JOIN regionen_bundesland rb
    ON p.bundesland_code2 = rb.bundesland_code2
{build_year_filter(question, alias="tv")}
GROUP BY tv.jahr, tv.monat, rb.bundesland_name
ORDER BY tv.jahr, tv.monat, rb.bundesland_name
""".strip()
        return (
            sql,
            "Template planner used: revenue by state and month "
            "(ticket_verkaeufe -> postleitzahlen -> regionen_bundesland).",
        )

    # Revenue by state
    if wants_revenue and wants_state:
        sql = f"""
SELECT
    tv.jahr,
    rb.bundesland_name,
    SUM(tv.umsatz_eur) AS umsatz_eur
FROM ticket_verkaeufe tv
JOIN postleitzahlen p
    ON CAST(tv.plz AS VARCHAR) = p.plz
JOIN regionen_bundesland rb
    ON p.bundesland_code2 = rb.bundesland_code2
{build_year_filter(question, alias="tv")}
GROUP BY tv.jahr, rb.bundesland_name
ORDER BY tv.jahr, umsatz_eur DESC
""".strip()
        return (
            sql,
            "Template planner used: revenue by state "
            "(ticket_verkaeufe -> postleitzahlen -> regionen_bundesland).",
        )

    # Revenue by month
    if wants_revenue and wants_month:
        sql = f"""
SELECT
    tv.jahr,
    tv.monat,
    SUM(tv.umsatz_eur) AS umsatz_eur
FROM ticket_verkaeufe tv
{build_year_filter(question, alias="tv")}
GROUP BY tv.jahr, tv.monat
ORDER BY tv.jahr, tv.monat
""".strip()
        return (
            sql,
            "Template planner used: revenue by month (ticket_verkaeufe).",
        )

    # Revenue by ticket type
    if wants_revenue and wants_ticket_type:
        sql = f"""
SELECT
    tv.jahr,
    tp.ticket_name,
    SUM(tv.umsatz_eur) AS umsatz_eur
FROM ticket_verkaeufe tv
JOIN ticket_produkte tp
    ON tv.ticket_code = tp.ticket_code
{build_year_filter(question, alias="tv")}
GROUP BY tv.jahr, tp.ticket_name
ORDER BY tv.jahr, umsatz_eur DESC
""".strip()
        return (
            sql,
            "Template planner used: revenue by ticket type "
            "(ticket_verkaeufe -> ticket_produkte).",
        )

    # Total revenue
    if wants_revenue and wants_total:
        sql = f"""
SELECT
    SUM(tv.umsatz_eur) AS umsatz_eur
FROM ticket_verkaeufe tv
{build_year_filter(question, alias="tv")}
""".strip()
        return (sql, "Template planner used: total revenue (ticket_verkaeufe).")

    return "", ""


def extract_sql_only(raw_text: str) -> str:
    """
    Keep only SQL from model output.
    Returns empty string if SQL cannot be extracted.
    """
    if not raw_text:
        return ""

    text = raw_text.strip()
    code_block = re.search(r"```sql\s*(.*?)```", text, re.IGNORECASE | re.DOTALL)
    if code_block:
        return code_block.group(1).strip().rstrip(";")

    select_or_with = re.search(r"\b(select|with)\b[\s\S]*", text, re.IGNORECASE)
    if select_or_with:
        candidate = select_or_with.group(0).strip()
        candidate = re.sub(r"```$", "", candidate).strip()
        return candidate.rstrip(";")

    return ""


def extract_did_you_mean_candidates(error_text: str, schema_tree: dict) -> list[str]:
    if not error_text:
        return []
    quoted = re.findall(r'"([^"]+)"', error_text)
    schema_tables = set(schema_tree.keys())
    return [item for item in quoted if item in schema_tables]


def guess_relevant_tables(question: str, schema_tree: dict) -> list[str]:
    q_lower = (question or "").lower()
    candidates = []

    if any(token in q_lower for token in ["revenue", "umsatz", "sales", "verkauf"]):
        candidates.extend(["ticket_verkaeufe", "plan_umsatz", "sonstige_angebote"])
    if any(token in q_lower for token in ["state", "bundesland", "region"]):
        candidates.extend(["postleitzahlen", "regionen_bundesland"])
    if any(token in q_lower for token in ["plz", "postal", "postleitzahl"]):
        candidates.append("postleitzahlen")
    if any(token in q_lower for token in ["ticket", "type", "produkt"]):
        candidates.extend(["ticket_verkaeufe", "ticket_produkte"])
    if any(token in q_lower for token in ["tariff", "tarif", "association", "verbund"]):
        candidates.append("tarifverbuende")
    if any(token in q_lower for token in ["meldestelle", "reporting office"]):
        candidates.append("meldestellen")

    # Keep order, remove duplicates, only existing tables.
    result = []
    seen = set()
    for table in candidates:
        if table in schema_tree and table not in seen:
            seen.add(table)
            result.append(table)

    if not result:
        # fallback to core fact + dimension tables
        for table in [
            "ticket_verkaeufe",
            "postleitzahlen",
            "regionen_bundesland",
            "ticket_produkte",
            "tarifverbuende",
        ]:
            if table in schema_tree and table not in seen:
                seen.add(table)
                result.append(table)

    return result[:8]


def build_join_hints(question: str) -> str:
    q_lower = (question or "").lower()
    hints = []

    if any(token in q_lower for token in ["state", "bundesland", "region"]):
        hints.append(
            "For state-level analytics, join "
            "ticket_verkaeufe.plz = postleitzahlen.plz, then "
            "postleitzahlen.bundesland_code2 = regionen_bundesland.bundesland_code2."
        )
    if any(token in q_lower for token in ["ticket", "type", "produkt"]):
        hints.append(
            "For ticket type details, join "
            "ticket_verkaeufe.ticket_code = ticket_produkte.ticket_code."
        )
    if any(token in q_lower for token in ["tariff", "tarif", "association", "verbund"]):
        hints.append(
            "For tariff association details, join "
            "ticket_verkaeufe.tarifverbund_id = tarifverbuende.tarifverbund_id."
        )
    if any(token in q_lower for token in ["meldestelle", "reporting office"]):
        hints.append(
            "For reporting office details, join "
            "ticket_verkaeufe.meldestelle_code = meldestellen.meldestelle_code."
        )

    return "\n".join(f"- {hint}" for hint in hints)


def build_retry_prompt(
    question: str, bad_sql: str, compile_error: str, schema_tree: dict
) -> str:
    relevant_tables = guess_relevant_tables(question, schema_tree)
    did_you_mean_tables = extract_did_you_mean_candidates(compile_error, schema_tree)
    for table in did_you_mean_tables:
        if table not in relevant_tables:
            relevant_tables.append(table)

    table_context_lines = []
    for table in relevant_tables:
        cols = schema_tree.get(table, [])
        col_names = ", ".join(col for col, _ in cols[:18])
        table_context_lines.append(f"- {table}({col_names})")
    table_context = "\n".join(table_context_lines)

    join_hints = build_join_hints(question)

    return (
        "Generate DuckDB SQL for mxquerychat.\n"
        "Use ONLY the tables and columns listed below.\n"
        "Return ONLY SQL (no explanation, no markdown).\n"
        "Use SELECT or WITH only.\n\n"
        f"Original question:\n{question}\n\n"
        f"Previous SQL failed:\n{bad_sql}\n\n"
        f"DuckDB error:\n{compile_error}\n\n"
        "Available relevant tables and columns:\n"
        f"{table_context}\n\n"
        "Join hints:\n"
        f"{join_hints if join_hints else '- Use the most suitable joins based on common keys.'}\n"
    )


def build_first_pass_prompt(question: str, schema_tree: dict) -> str:
    """First model call prompt: schema-guided from the start for higher accuracy."""
    relevant_tables = guess_relevant_tables(question, schema_tree)
    table_context_lines = []
    for table in relevant_tables:
        cols = schema_tree.get(table, [])
        col_names = ", ".join(col for col, _ in cols[:20])
        table_context_lines.append(f"- {table}({col_names})")
    table_context = "\n".join(table_context_lines)
    join_hints = build_join_hints(question)

    return (
        "Generate DuckDB SQL for mxquerychat.\n"
        "Use ONLY the tables and columns listed below.\n"
        "Return ONLY SQL (no explanation, no markdown).\n"
        "Use SELECT or WITH only.\n\n"
        f"Question:\n{question}\n\n"
        "Available relevant tables and columns:\n"
        f"{table_context}\n\n"
        "Join hints:\n"
        f"{join_hints if join_hints else '- Use the most suitable joins based on common keys.'}\n"
    )


def build_final_retry_prompt(
    question: str, bad_sql: str, compile_error: str, schema_tree: dict
) -> str:
    """Final strict prompt: force SQL-only or explicit NO_MATCH token."""
    relevant_tables = guess_relevant_tables(question, schema_tree)
    table_context_lines = []
    for table in relevant_tables:
        cols = schema_tree.get(table, [])
        col_names = ", ".join(col for col, _ in cols[:20])
        table_context_lines.append(f"- {table}({col_names})")
    table_context = "\n".join(table_context_lines)
    join_hints = build_join_hints(question)

    return (
        "You are generating SQL for mxquerychat on DuckDB.\n"
        "Rules:\n"
        "1) Use ONLY the listed tables and columns.\n"
        "2) Output ONLY one SQL query using SELECT/WITH, no markdown.\n"
        "3) If the question cannot be answered from this schema, output exactly: NO_MATCH\n\n"
        f"Question:\n{question}\n\n"
        f"Previous failed SQL:\n{bad_sql if bad_sql else '-- no valid SQL produced --'}\n\n"
        f"Compilation/validation issue:\n{compile_error}\n\n"
        "Available tables and columns:\n"
        f"{table_context}\n\n"
        "Join hints:\n"
        f"{join_hints if join_hints else '- Use the most suitable joins based on shared keys.'}\n"
    )


def generate_sql_with_retry(
    generate_sql_fn: Callable[[str], str],
    question_text: str,
    schema_tree: dict,
    compile_sql_fn: Callable[[str], str],
    timeout_seconds: int,
    run_with_timeout_fn: Callable[[Callable[[], Any], int], tuple[Any, Any]],
) -> tuple[str, list[str], str]:
    """
    Returns: (sql, notes, error_code)
    sql is empty when no valid SQL could be generated.
    error_code: "", "timeout", "model_error", "no_match"
    """
    notes: list[str] = []

    def run_attempt(prompt_text: str, attempt_name: str) -> tuple[str, str, str]:
        """
        Returns: (sql, status, detail)
        status: "ok", "timeout", "model_error", "non_sql", "no_match", "compile_error"
        """
        raw_sql, call_error = run_with_timeout_fn(
            lambda: generate_sql_fn(prompt_text),
            timeout_seconds,
        )
        if call_error:
            call_error_text = str(call_error)
            if "timeout" in call_error_text.lower():
                notes.append(f"{attempt_name}: model timeout ({call_error_text}).")
                return "", "timeout", call_error_text
            notes.append(f"{attempt_name}: model error ({call_error_text}).")
            return "", "model_error", call_error_text

        raw_text = (raw_sql or "").strip()
        if raw_text.upper() == "NO_MATCH":
            notes.append(f"{attempt_name}: model returned NO_MATCH.")
            return "", "no_match", "NO_MATCH"

        sql = extract_sql_only(raw_text)
        if not sql:
            notes.append(f"{attempt_name}: model returned non-SQL output.")
            return "", "non_sql", raw_text[:120]

        try:
            compile_error = compile_sql_fn(sql)
        except Exception as exc:
            compile_error = str(exc)

        if compile_error:
            notes.append(f"{attempt_name}: SQL validation failed, trying stricter prompt.")
            return sql, "compile_error", compile_error

        notes.append(f"{attempt_name}: valid SQL generated.")
        return sql, "ok", ""

    first_prompt = build_first_pass_prompt(question_text, schema_tree)
    sql_1, status_1, detail_1 = run_attempt(first_prompt, "Attempt 1 (schema-guided)")
    if status_1 == "ok":
        return sql_1, notes, ""
    if status_1 == "timeout":
        return "", notes, "timeout"
    if status_1 == "model_error":
        return "", notes, "model_error"

    retry_prompt = build_retry_prompt(
        question_text,
        sql_1 or "-- no valid SQL produced --",
        detail_1 or "No compilable SQL in attempt 1.",
        schema_tree,
    )
    sql_2, status_2, detail_2 = run_attempt(
        retry_prompt, "Attempt 2 (schema-aware retry)"
    )
    if status_2 == "ok":
        notes.append("Resolved using related tables and join hints.")
        return sql_2, notes, ""
    if status_2 == "timeout":
        return "", notes, "timeout"
    if status_2 == "model_error":
        return "", notes, "model_error"

    final_prompt = build_final_retry_prompt(
        question_text,
        sql_2 or sql_1,
        detail_2 or detail_1 or "No compilable SQL in previous attempts.",
        schema_tree,
    )
    sql_3, status_3, _ = run_attempt(final_prompt, "Attempt 3 (final strict attempt)")
    if status_3 == "ok":
        notes.append("Resolved on final strict attempt.")
        return sql_3, notes, ""
    if status_3 == "timeout":
        return "", notes, "timeout"
    if status_3 == "model_error":
        return "", notes, "model_error"

    notes.append("All generation strategies failed: no reliable SQL from available schema.")
    return "", notes, "no_match"


def get_local_guardrail_message(question: str) -> str:
    """Fast local checks before calling the model."""
    if not question or not question.strip():
        return "Please enter a question."

    q = question.lower()
    if len(q.split()) < 2:
        return "Please write a fuller data question."

    if any(re.search(pattern, q) for pattern in WRITE_PATTERNS):
        return "Read-only mode: write operations are not allowed."

    if any(re.search(pattern, q) for pattern in OFF_TOPIC_PATTERNS):
        return "Off-topic request. Please ask about the mxquerychat dataset."

    return ""


def explain_sql_brief(sql: str) -> str:
    """Return a short deterministic explanation of the SQL query intent."""
    cleaned = (sql or "").strip()
    if not cleaned:
        return "No SQL to explain yet."

    lowered = cleaned.lower()
    table_match = re.search(r"\bfrom\s+([a-zA-Z0-9_\.]+)", lowered)
    table_text = table_match.group(1) if table_match else "the selected tables"

    aggregation = []
    if "sum(" in lowered:
        aggregation.append("SUM")
    if "count(" in lowered:
        aggregation.append("COUNT")
    if "avg(" in lowered:
        aggregation.append("AVG")
    if "min(" in lowered:
        aggregation.append("MIN")
    if "max(" in lowered:
        aggregation.append("MAX")

    parts = [f"This query reads data from `{table_text}`."]
    if "where " in lowered:
        parts.append("It applies filters before returning results.")
    if aggregation:
        parts.append(
            "It calculates aggregated values using "
            + ", ".join(aggregation)
            + "."
        )
    if "group by" in lowered:
        parts.append("Results are grouped by one or more dimensions.")
    if "order by" in lowered:
        parts.append("Output is sorted for easier reading.")

    return " ".join(parts)


def classify_generation_failure(error_code: str) -> str:
    """Map generation error codes to standardized failure categories."""
    normalized = (error_code or "").strip().lower()
    if normalized == "timeout":
        return "timeout"
    if normalized == "no_match":
        return "no_match"
    if normalized == "compile_fail":
        return "compile_fail"
    if normalized in {"model_error", "runtime_fail"}:
        return "runtime_fail"
    return "runtime_fail"


def classify_execution_failure(error_text: str) -> str:
    """Classify execution-stage failures into stable categories."""
    lowered = (error_text or "").lower()
    if "timeout" in lowered:
        return "timeout"

    compile_markers = [
        "parser error",
        "syntax error",
        "binder error",
        "catalog error",
        "does not exist",
        "no such table",
        "no such column",
    ]
    if any(token in lowered for token in compile_markers):
        return "compile_fail"

    return "runtime_fail"


def reset_question_flow_state(state: MutableMapping[str, Any]) -> None:
    """Reset New Question flow values without touching training data state."""
    state["question"] = ""
    state["generated_sql"] = ""
    state["last_result_df"] = None
    state["last_result_elapsed"] = None
    state["suggestions"] = []
    state["generation_notes"] = []
    state["feedback_last_rating"] = None
    state["feedback_last_question_hash"] = "no_question"


def run_query_if_read_only(
    sql: str,
    validate_fn: Callable[[str], tuple[bool, str]],
    run_fn: Callable[[str], Any],
) -> tuple[bool, Any, str]:
    """
    Execute only when SQL passes read-only validation.
    Returns: (is_allowed, run_result_or_none, message)
    """
    is_ok, message = validate_fn(sql)
    if not is_ok:
        return False, None, message
    return True, run_fn(sql), "OK"
