"""
Beginner-friendly Streamlit UI for mxquerychat.

Flow:
1) Ask a question
2) Generate SQL
3) Review / edit SQL
4) Validate read-only safety
5) Run query and view results
"""

import re
import time
import hashlib
import os
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from pathlib import Path
from typing import Optional

import duckdb
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from src.core import query_logic
from src.db.data_source import (
    get_active_source_info,
    refresh_schema_cache,
    reload_dataset_cache,
)
from src.db.execution_policy import (
    ExecutionPolicy,
    apply_row_limit,
    extract_complexity_policy_details,
    run_query_with_timeout,
    validate_sql_complexity,
)
from src.llm.sql_explainer import (
    build_explanation_cache_key,
    maybe_generate_explanation,
)
from src.utils.telemetry import configure_app_logging, record_metric_event
from sql_guard import validate_read_only_sql
from vannaagent import (
    get_vanna,
    get_exact_training_sql,
    load_training_examples,
    normalize_training_for_save,
    save_training_examples,
    train_from_examples,
)

DUCKDB_PATH = "mxquerychat.duckdb"
ICON_PATH = Path("docs/images/icon.png")
LOGO_PATH = Path("docs/images/logo.png")
MODEL_TIMEOUT_SECONDS = 65
OLLAMA_BASE_URL = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434")
EXPLANATION_MODEL = os.getenv("EXPLANATION_MODEL", os.getenv("OLLAMA_MODEL", "mistral"))
try:
    EXPLANATION_TIMEOUT_SECONDS = int(os.getenv("EXPLANATION_TIMEOUT_SECONDS", "35"))
except ValueError:
    EXPLANATION_TIMEOUT_SECONDS = 35
EXECUTION_POLICY = ExecutionPolicy()
configure_app_logging()

EXAMPLE_QUESTIONS = [
    "What is the total revenue in 2025?",
    "Show revenue per month for 2024.",
    "Which ticket types generate the most revenue?",
    "Show revenue by federal state for 2025.",
]


@st.cache_resource(show_spinner=False)
def get_vanna_cached():
    return get_vanna()


@st.cache_data(show_spinner=False)
def get_schema_tree() -> dict:
    """Return schema as: {table_name: [(column_name, data_type), ...]}."""
    con = duckdb.connect(DUCKDB_PATH, read_only=True)
    try:
        tables = con.execute(
            """
            SELECT table_name
            FROM information_schema.tables
            WHERE table_schema = 'main'
            ORDER BY table_name
            """
        ).fetchall()

        schema = {}
        for (table_name,) in tables:
            cols = con.execute(
                f"""
                SELECT column_name, data_type
                FROM information_schema.columns
                WHERE table_name = '{table_name}'
                ORDER BY ordinal_position
                """
            ).fetchall()
            schema[table_name] = cols
        return schema
    finally:
        con.close()


@st.cache_data(show_spinner=False)
def get_available_years() -> list[int]:
    """Read available years from core fact tables."""
    con = duckdb.connect(DUCKDB_PATH, read_only=True)
    try:
        rows = con.execute(
            """
            SELECT DISTINCT jahr
            FROM (
                SELECT jahr FROM ticket_verkaeufe
                UNION ALL
                SELECT jahr FROM plan_umsatz
                UNION ALL
                SELECT jahr FROM sonstige_angebote
            ) t
            WHERE jahr IS NOT NULL
            ORDER BY jahr
            """
        ).fetchall()
        return [int(row[0]) for row in rows]
    except Exception:
        return []
    finally:
        con.close()


def extract_requested_years(question: str) -> list[int]:
    return [int(y) for y in re.findall(r"\b(19\d{2}|20\d{2})\b", question or "")]


def nearest_year(target_year: int, available_years: list[int]) -> Optional[int]:
    if not available_years:
        return None
    return min(available_years, key=lambda value: abs(value - target_year))


def build_suggested_questions(question: str) -> list[str]:
    """Build 'Did you mean' prompts using available years."""
    suggestions: list[str] = []
    years = get_available_years()
    requested = extract_requested_years(question)
    q_lower = (question or "").lower()

    default_year = max(years) if years else 2025
    if requested and years:
        default_year = nearest_year(requested[0], years) or default_year

    # Keep suggestions relevant to the original intent.
    if "state" in q_lower or "bundesland" in q_lower:
        suggestions.append(f"Show revenue by federal state for {default_year}.")
    if (
        "tariff" in q_lower
        or "tarif" in q_lower
        or "association" in q_lower
        or "verbund" in q_lower
    ):
        suggestions.append(
            f"Show revenue by tariff association for {default_year}."
        )
        suggestions.append(
            f"Show revenue by tariff association and federal state for {default_year}."
        )
    if "month" in q_lower or "monat" in q_lower:
        suggestions.append(f"Show revenue per month for {default_year}.")
    if ("state" in q_lower or "bundesland" in q_lower) and (
        "month" in q_lower or "monat" in q_lower
    ):
        suggestions.append(
            f"Show revenue by federal state and month for {default_year}."
        )

    if requested and years:
        for year in requested:
            nearest = nearest_year(year, years)
            if nearest is not None:
                suggestions.append(f"Show revenue per month for {nearest}.")
        latest = max(years)
        suggestions.append(f"What is the total revenue in {latest}?")
    else:
        suggestions.extend(EXAMPLE_QUESTIONS[:3])

    # Keep unique order and max 4
    seen = set()
    result = []
    for item in suggestions:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result[:4]


def get_data_availability_message(question: str) -> tuple[str, list[str]]:
    """Return an availability warning + suggestions when data is clearly unavailable."""
    requested_years = extract_requested_years(question)
    available_years = get_available_years()

    if requested_years and available_years:
        missing = [y for y in requested_years if y not in available_years]
        if missing:
            return (
                "I could not find data for year(s): "
                + ", ".join(str(y) for y in missing)
                + f". Available years: {', '.join(str(y) for y in available_years)}.",
                build_suggested_questions(question),
            )
    return "", []


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


def validate_sql_compiles(sql: str) -> str:
    """
    Return empty string if SQL compiles against DuckDB schema, else error message.
    """
    if not sql:
        return "No SQL generated."
    con = duckdb.connect(DUCKDB_PATH, read_only=True)
    try:
        con.execute(f"EXPLAIN {sql}")
        return ""
    except Exception as exc:
        return str(exc)
    finally:
        con.close()


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


def generate_sql_with_retry(vn, question_text: str) -> tuple[str, list[str], str]:
    """
    Returns: (sql, notes, error_code)
    sql is empty when no valid SQL could be generated.
    error_code: "", "timeout", "model_error", "no_match"
    """
    notes: list[str] = []
    schema_tree = get_schema_tree()

    def run_attempt(prompt_text: str, attempt_name: str) -> tuple[str, str, str]:
        """
        Returns: (sql, status, detail)
        status: "ok", "timeout", "model_error", "non_sql", "no_match", "compile_error"
        """
        raw_sql, call_error = run_with_timeout(
            lambda: vn.generate_sql(prompt_text),
            MODEL_TIMEOUT_SECONDS,
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

        compile_error = validate_sql_compiles(sql)
        if compile_error:
            notes.append(
                f"{attempt_name}: SQL validation failed, trying stricter prompt."
            )
            return sql, "compile_error", compile_error

        notes.append(f"{attempt_name}: valid SQL generated.")
        return sql, "ok", ""

    first_prompt = build_first_pass_prompt(question_text, schema_tree)
    sql_1, status_1, detail_1 = run_attempt(
        first_prompt, "Attempt 1 (schema-guided)"
    )
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


def run_with_timeout(fn, timeout_seconds: int):
    """Run function with timeout so UI does not hang forever."""
    executor = ThreadPoolExecutor(max_workers=1)
    future = executor.submit(fn)
    try:
        return future.result(timeout=timeout_seconds), None
    except FutureTimeoutError:
        future.cancel()
        return None, f"Model timeout after {timeout_seconds} seconds."
    except Exception as exc:
        return None, str(exc)
    finally:
        executor.shutdown(wait=False, cancel_futures=True)


def get_local_guardrail_message(question: str) -> str:
    """Fast local checks before calling the model."""
    if not question or not question.strip():
        return "Please enter a question."

    q = question.lower()
    if len(q.split()) < 2:
        return "Please write a fuller data question."

    if any(re.search(p, q) for p in WRITE_PATTERNS):
        return "Read-only mode: write operations are not allowed."

    if any(re.search(p, q) for p in OFF_TOPIC_PATTERNS):
        return "Off-topic request. Please ask about the mxquerychat dataset."

    return ""


def run_read_only_query(sql: str) -> tuple[pd.DataFrame, float]:
    con = duckdb.connect(DUCKDB_PATH, read_only=True)
    try:
        start = time.time()
        df = con.execute(sql).df()
        elapsed = time.time() - start
        return df, elapsed
    finally:
        con.close()


def try_show_bar_chart(df: pd.DataFrame) -> None:
    """Show a simple chart when first column is x and second is numeric."""
    if df.shape[1] < 2:
        st.info("Chart skipped: result needs at least 2 columns.")
        return

    x_col = df.columns[0]
    y_col = df.columns[1]
    if not pd.api.types.is_numeric_dtype(df[y_col]):
        st.info("Chart skipped: second column is not numeric.")
        return

    st.bar_chart(df[[x_col, y_col]].set_index(x_col))


def init_state():
    defaults = {
        "question": "",
        "generated_sql": "",
        "last_result_df": None,
        "last_result_elapsed": None,
        "sql_cache": {},
        "suggestions": [],
        "generation_notes": [],
        "metrics_questions_total": 0,
        "metrics_generation_success": 0,
        "metrics_generation_failed": 0,
        "metrics_blocked_requests": 0,
        "metrics_cache_hits": 0,
        "metrics_query_success": 0,
        "metrics_query_failed": 0,
        "metrics_feedback_up": 0,
        "metrics_feedback_down": 0,
        "feedback_last_rating": None,
        "feedback_last_question_hash": "no_question",
        "generated_explanation": "",
        "explanation_status": "idle",
        "explanation_cache": {},
        "training_working_df": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def reset_question_flow():
    query_logic.reset_question_flow_state(st.session_state)


def build_question_hash(question: str) -> str:
    text = (question or "").strip()
    if not text:
        return "no_question"
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def inject_enter_to_send_js() -> None:
    """
    Keyboard behavior for question input:
    - Enter: send
    - Shift+Enter: new line
    """
    components.html(
        """
        <script>
        (function () {
          const doc = window.parent.document;

          function bindEnterToSend() {
            const textarea = doc.querySelector('textarea[aria-label="Question"]');
            if (!textarea) return false;
            if (textarea.dataset.mxBound === "1") return true;
            textarea.dataset.mxBound = "1";

            textarea.addEventListener("keydown", function (event) {
              if (event.key === "Enter" && !event.shiftKey && !event.isComposing) {
                event.preventDefault();
                // Push latest textarea value into Streamlit's frontend state.
                textarea.dispatchEvent(new Event("input", { bubbles: true }));
                textarea.dispatchEvent(new Event("change", { bubbles: true }));

                // Click Send after a short delay so state is synced.
                setTimeout(function () {
                  const sendButton = Array.from(doc.querySelectorAll("button"))
                    .find((btn) => btn.innerText.trim() === "Send");
                  if (sendButton) sendButton.click();
                }, 80);
              }
            });
            return true;
          }

          if (!bindEnterToSend()) {
            let tries = 0;
            const timer = setInterval(function () {
              tries += 1;
              if (bindEnterToSend() || tries > 40) clearInterval(timer);
            }, 100);
          }
        })();
        </script>
        """,
        height=0,
        width=0,
    )


def format_suggestion_button_text(text: str, max_len: int = 52) -> str:
    """Short label for suggestion buttons while keeping full text in tooltip."""
    cleaned = " ".join((text or "").split())
    if len(cleaned) <= max_len:
        return cleaned
    return cleaned[: max_len - 3].rstrip() + "..."


page_icon = str(ICON_PATH) if ICON_PATH.exists() else ":mag:"
st.set_page_config(page_title="mxquerychat", page_icon=page_icon, layout="wide")
init_state()

st.markdown(
    """
    <style>
      .block-container {padding-top: 1.5rem;}
      .mx-card {
        padding: 12px 14px;
        border: 1px solid #d9d9d9;
        border-radius: 10px;
        background: #fafafa;
      }
      .mx-prompt-card {
        padding: 14px;
        border: 1px solid #c9d6ea;
        border-radius: 12px;
        background: #f4f8ff;
        margin-top: 8px;
        margin-bottom: 10px;
      }
      div[data-testid="stTextArea"] textarea {
        border: 2px solid #a9bfdc;
        border-radius: 10px;
        background: #ffffff;
        min-height: 64px;
      }
      div[data-testid="stTextArea"] textarea:focus {
        border: 2px solid #4a78b2;
        box-shadow: 0 0 0 2px rgba(74, 120, 178, 0.2);
      }
    </style>
    """,
    unsafe_allow_html=True,
)

top_left, top_right = st.columns([0.8, 0.2])
with top_left:
    st.title("mxquerychat")
    st.caption("Ask questions in plain language, get SQL query and results. Built with Vanna agent and DuckDB")
with top_right:
    st.markdown("<div style='height: 34px;'></div>", unsafe_allow_html=True)
    if st.button("New Chat", width="stretch"):
        reset_question_flow()
        st.rerun()

if LOGO_PATH.exists():
    st.sidebar.image(str(LOGO_PATH), width="stretch")
st.sidebar.header("Navigation")
view = st.sidebar.radio("Choose page", ["New Question", "Training Data"])
st.sidebar.markdown("---")
st.sidebar.subheader("Session Metrics")
st.sidebar.metric("Questions", st.session_state.metrics_questions_total)
st.sidebar.metric("SQL Generated", st.session_state.metrics_generation_success)
st.sidebar.metric("Generation Failed", st.session_state.metrics_generation_failed)
st.sidebar.metric("Blocked Requests", st.session_state.metrics_blocked_requests)
st.sidebar.metric("Query Success", st.session_state.metrics_query_success)
st.sidebar.metric("Query Failed", st.session_state.metrics_query_failed)
st.sidebar.metric("Feedback Up", st.session_state.metrics_feedback_up)
st.sidebar.metric("Feedback Down", st.session_state.metrics_feedback_down)

if view == "New Question":
    st.sidebar.markdown("---")
    st.sidebar.subheader("Data Source (DuckDB)")
    source_info = get_active_source_info(DUCKDB_PATH)
    st.sidebar.caption(f"Engine: {source_info['engine']}")
    st.sidebar.caption(f"Path: `{source_info['path']}`")
    st.sidebar.caption(f"Exists: {source_info['exists']} | Size MB: {source_info['size_mb']}")

    ds_col1, ds_col2 = st.sidebar.columns(2)
    with ds_col1:
        if st.button("Reload Dataset", width="stretch"):
            reload_dataset_cache()
            st.session_state.sql_cache = {}
            st.session_state.suggestions = []
            st.session_state.last_result_df = None
            st.session_state.last_result_elapsed = None
            record_metric_event("dataset_reload", success=True, source="duckdb")
            st.rerun()
    with ds_col2:
        if st.button("Refresh Schema", width="stretch"):
            refresh_schema_cache()
            record_metric_event("schema_refresh", success=True, source="duckdb")
            st.rerun()

    with st.sidebar.expander("Schema Tree", expanded=False):
        try:
            schema_tree_sidebar = get_schema_tree()
            for table_name, cols in schema_tree_sidebar.items():
                st.markdown(f"**{table_name}**")
                for column_name, data_type in cols:
                    st.caption(f"{column_name} ({data_type})")
        except Exception as exc:
            st.error(f"Schema load failed: {exc}")

if view == "New Question":
    st.markdown("### Step 1: Ask a data question")
    st.markdown('<div class="mx-card">Try one of these examples first.</div>', unsafe_allow_html=True)

    ex_cols = st.columns(len(EXAMPLE_QUESTIONS))
    for idx, ex_question in enumerate(EXAMPLE_QUESTIONS):
        with ex_cols[idx]:
            if st.button(f"Example {idx + 1}", width="stretch"):
                st.session_state.question = ex_question

    st.markdown(
        '<div class="mx-prompt-card"><strong>Question</strong><br/>Type your question in plain language.</div>',
        unsafe_allow_html=True,
    )
    st.session_state.question = st.text_area(
        "Question",
        value=st.session_state.question,
        placeholder="Example: Show revenue by state for 2025.",
        label_visibility="visible",
        height=72,
    )
    inject_enter_to_send_js()
    if st.session_state.suggestions:
        st.caption("Suggested questions:")
        sug_cols = st.columns(len(st.session_state.suggestions))
        for idx, suggestion in enumerate(st.session_state.suggestions):
            with sug_cols[idx]:
                if st.button(
                    format_suggestion_button_text(suggestion),
                    key=f"suggestion_{idx}",
                    help=suggestion,
                    width="stretch",
                ):
                    st.session_state.question = suggestion
                    st.session_state.generated_sql = ""
                    st.session_state.generated_explanation = ""
                    st.session_state.explanation_status = "idle"
                    st.session_state.generation_notes = []
                    st.session_state.suggestions = []
                    st.rerun()

    if st.button("Send", type="primary"):
        generation_started_at = time.time()
        st.session_state.generated_explanation = ""
        st.session_state.explanation_status = "idle"
        st.session_state.metrics_questions_total += 1
        question_for_metrics = (st.session_state.question or "").strip()
        record_metric_event(
            "question_submitted",
            question_length=len(question_for_metrics),
        )
        st.session_state.generation_notes = []
        local_block = query_logic.get_local_guardrail_message(st.session_state.question)
        if local_block:
            st.session_state.generated_sql = ""
            st.session_state.suggestions = build_suggested_questions(st.session_state.question)
            st.session_state.generation_notes = [f"Stopped by local guardrail: {local_block}"]
            st.session_state.metrics_blocked_requests += 1
            st.session_state.metrics_generation_failed += 1
            local_failure_category = (
                "blocked_read_only"
                if "read-only" in local_block.lower()
                else "no_match"
            )
            record_metric_event(
                "sql_generation",
                success=False,
                path="local_guardrail",
                failure_reason=local_block,
                failure_category=local_failure_category,
                duration_ms=int((time.time() - generation_started_at) * 1000),
            )
            st.caption(f"Failure type: {local_failure_category}")
            st.warning(local_block)
        else:
            question_text = st.session_state.question.strip()
            cache_key = question_text.lower()
            handled_fast_path = False
            availability_message, availability_suggestions = get_data_availability_message(
                question_text
            )
            if availability_message:
                st.session_state.generated_sql = ""
                st.session_state.suggestions = availability_suggestions
                st.session_state.generation_notes = [availability_message]
                st.session_state.metrics_generation_failed += 1
                record_metric_event(
                    "sql_generation",
                    success=False,
                    path="data_availability",
                    failure_reason=availability_message,
                    failure_category="no_match",
                    duration_ms=int((time.time() - generation_started_at) * 1000),
                )
                st.caption("Failure type: no_match")
                st.warning(availability_message)
                handled_fast_path = True

            # 1) Session cache (fastest)
            if not handled_fast_path:
                cached_sql = st.session_state.sql_cache.get(cache_key)
                if cached_sql:
                    st.session_state.generated_sql = cached_sql
                    st.session_state.suggestions = []
                    st.session_state.generation_notes = ["Loaded from session cache."]
                    st.session_state.metrics_generation_success += 1
                    st.session_state.metrics_cache_hits += 1
                    record_metric_event(
                        "sql_generation",
                        success=True,
                        path="session_cache",
                        duration_ms=int((time.time() - generation_started_at) * 1000),
                    )
                    st.success("Used cached SQL (instant).")
                    handled_fast_path = True

            # 2) Exact match in training examples (fast, no LLM)
            if not handled_fast_path:
                examples_df = load_training_examples()
                direct_sql = get_exact_training_sql(question_text, examples_df)
                if direct_sql:
                    st.session_state.generated_sql = direct_sql
                    st.session_state.sql_cache[cache_key] = direct_sql
                    st.session_state.suggestions = []
                    st.session_state.generation_notes = [
                        "Matched directly from training examples."
                    ]
                    st.session_state.metrics_generation_success += 1
                    record_metric_event(
                        "sql_generation",
                        success=True,
                        path="training_exact_match",
                        duration_ms=int((time.time() - generation_started_at) * 1000),
                    )
                    st.success("Used training example SQL (instant).")
                    handled_fast_path = True

            # 3) Deterministic template planner (reliable joins for common intents)
            if not handled_fast_path:
                template_sql, template_note = query_logic.build_template_sql(question_text)
                if template_sql:
                    compile_error = validate_sql_compiles(template_sql)
                    if not compile_error:
                        st.session_state.generated_sql = template_sql
                        st.session_state.sql_cache[cache_key] = template_sql
                        st.session_state.suggestions = []
                        st.session_state.generation_notes = [template_note]
                        st.session_state.metrics_generation_success += 1
                        record_metric_event(
                            "sql_generation",
                            success=True,
                            path="template_planner",
                            duration_ms=int((time.time() - generation_started_at) * 1000),
                        )
                        st.success("Used template planner.")
                        handled_fast_path = True
                    else:
                        # Keep note but continue to LLM fallback.
                        st.session_state.generation_notes = [
                            "Template planner matched intent but SQL compile failed. Falling back to LLM."
                        ]

            # 4) LLM fallback + retry
            if not handled_fast_path:
                existing_notes = st.session_state.generation_notes or []
                if existing_notes:
                    st.session_state.generation_notes = existing_notes + [
                        "Deterministic path did not finish with valid SQL. Starting LLM fallback."
                    ]
                else:
                    st.session_state.generation_notes = [
                        "No cache/training/template match. Starting LLM fallback."
                    ]

                with st.spinner("Generating SQL..."):
                    vn = get_vanna_cached()
                    sql, sql_error_notes, sql_error = query_logic.generate_sql_with_retry(
                        generate_sql_fn=lambda prompt: vn.generate_sql(prompt),
                        question_text=question_text,
                        schema_tree=get_schema_tree(),
                        compile_sql_fn=validate_sql_compiles,
                        timeout_seconds=MODEL_TIMEOUT_SECONDS,
                        run_with_timeout_fn=run_with_timeout,
                    )

                    if sql_error:
                        st.session_state.generated_sql = ""
                        st.session_state.suggestions = build_suggested_questions(
                            question_text
                        )
                        st.session_state.metrics_generation_failed += 1
                        existing_notes = st.session_state.generation_notes or []
                        st.session_state.generation_notes = (
                            existing_notes
                            + sql_error_notes
                            + ["Final status: no reliable SQL generated."]
                        )
                        failure_category = query_logic.classify_generation_failure(
                            sql_error
                        )
                        record_metric_event(
                            "sql_generation",
                            success=False,
                            path="llm_fallback",
                            failure_reason=sql_error,
                            failure_category=failure_category,
                            duration_ms=int((time.time() - generation_started_at) * 1000),
                        )
                        st.caption(f"Failure type: {failure_category}")

                        if sql_error == "timeout":
                            st.error(
                                "Could not generate SQL right now due to model timeout. "
                                "Tip: first call can be slow (model cold start). Try once more."
                            )
                        elif sql_error == "model_error":
                            st.error(
                                "Could not generate SQL due to a model connection error. "
                                "Please verify Ollama is running and retry."
                            )
                        else:
                            st.error(
                                "I could not find matching tables/columns for that request in this database."
                            )
                    else:
                        st.session_state.generated_sql = sql
                        st.session_state.sql_cache[cache_key] = sql
                        st.session_state.suggestions = []
                        existing_notes = st.session_state.generation_notes or []
                        st.session_state.generation_notes = (
                            existing_notes
                            + sql_error_notes
                            + ["Final status: SQL generated successfully."]
                        )
                        st.session_state.metrics_generation_success += 1
                        record_metric_event(
                            "sql_generation",
                            success=True,
                            path="llm_fallback",
                            duration_ms=int((time.time() - generation_started_at) * 1000),
                        )

    if st.session_state.generation_notes:
        st.markdown("### How query was built")
        for note in st.session_state.generation_notes:
            st.info(note)

    if st.session_state.generated_sql:
        st.markdown("### Step 2: Review SQL")
        st.session_state.generated_sql = st.text_area(
            "Generated SQL (editable)",
            value=st.session_state.generated_sql,
            height=170,
        )
        explanation_key = build_explanation_cache_key(
            st.session_state.question,
            st.session_state.generated_sql,
        )
        st.session_state.generated_explanation = st.session_state.explanation_cache.get(
            explanation_key, ""
        )
        if st.session_state.generated_explanation:
            st.session_state.explanation_status = "ready"
        elif st.session_state.explanation_status == "ready":
            st.session_state.explanation_status = "idle"

        with st.expander("Explain this SQL", expanded=False):
            st.caption(
                "Optional: generate a short SQL explanation using local Ollama. "
                "This does not affect SQL execution."
            )
            if st.button("Generate Explanation", key="generate_explanation"):
                started = time.time()
                explanation_text, explanation_status, cache_hit = (
                    maybe_generate_explanation(
                        triggered=True,
                        question=st.session_state.question,
                        sql=st.session_state.generated_sql,
                        cache=st.session_state.explanation_cache,
                        model=EXPLANATION_MODEL,
                        ollama_url=OLLAMA_BASE_URL,
                        timeout_seconds=EXPLANATION_TIMEOUT_SECONDS,
                    )
                )
                st.session_state.generated_explanation = explanation_text
                st.session_state.explanation_status = explanation_status
                duration_ms = int((time.time() - started) * 1000)

                if explanation_status == "ready":
                    record_metric_event(
                        "sql_explanation",
                        success=True,
                        path="on_demand_ollama",
                        cache_hit=cache_hit,
                        duration_ms=duration_ms,
                    )
                else:
                    failure_category = (
                        "timeout"
                        if explanation_status == "timeout"
                        else "runtime_fail"
                    )
                    record_metric_event(
                        "sql_explanation",
                        success=False,
                        path="on_demand_ollama",
                        failure_reason=explanation_status,
                        failure_category=failure_category,
                        duration_ms=duration_ms,
                    )

            if st.session_state.explanation_status == "ready" and st.session_state.generated_explanation:
                st.success("Explanation ready.")
                st.write(st.session_state.generated_explanation)
            elif st.session_state.explanation_status == "timeout":
                st.info(
                    "Explanation unavailable: local model timeout. "
                    "You can still run the SQL query. "
                    "Try again after Ollama warms up, or increase EXPLANATION_TIMEOUT_SECONDS."
                )
            elif st.session_state.explanation_status in {"model_error", "empty_response"}:
                st.info(
                    "Explanation unavailable right now from local model. "
                    "You can still run the SQL query."
                )

    if st.session_state.generated_sql:
        st.markdown("### Step 3: Safety check")
        is_ok, guard_message = validate_read_only_sql(st.session_state.generated_sql)
        if is_ok:
            st.success("Read-only check passed.")
        else:
            st.error(guard_message)

        if st.button("Run Query", disabled=not is_ok):
            try:
                with st.spinner("Running query..."):
                    is_allowed, _, run_message = query_logic.run_query_if_read_only(
                        st.session_state.generated_sql,
                        validate_read_only_sql,
                        lambda _sql: None,
                    )
                    if not is_allowed:
                        st.session_state.metrics_blocked_requests += 1
                        st.session_state.metrics_query_failed += 1
                        record_metric_event(
                            "query_execution",
                            success=False,
                            failure_reason=run_message,
                            failure_category="blocked_read_only",
                            policy_reason="read_only",
                            policy_threshold=None,
                        )
                        st.error(f"Failure type: blocked_read_only. {run_message}")
                        st.stop()

                    complexity_ok, complexity_message = validate_sql_complexity(
                        st.session_state.generated_sql,
                        EXECUTION_POLICY,
                    )
                    if not complexity_ok:
                        policy_reason, policy_threshold = (
                            extract_complexity_policy_details(complexity_message)
                        )
                        st.session_state.metrics_blocked_requests += 1
                        st.session_state.metrics_query_failed += 1
                        record_metric_event(
                            "query_execution",
                            success=False,
                            failure_reason=complexity_message,
                            failure_category="blocked_complexity",
                            policy_reason=policy_reason,
                            policy_threshold=policy_threshold,
                        )
                        st.error(f"Failure type: blocked_complexity. {complexity_message}")
                        st.stop()

                    limited_sql = apply_row_limit(
                        st.session_state.generated_sql,
                        EXECUTION_POLICY.max_rows,
                    )
                    df, elapsed, exec_error = run_query_with_timeout(
                        DUCKDB_PATH,
                        limited_sql,
                        EXECUTION_POLICY.timeout_seconds,
                    )
                    if exec_error:
                        exec_category = query_logic.classify_execution_failure(exec_error)
                        st.session_state.metrics_query_failed += 1
                        record_metric_event(
                            "query_execution",
                            success=False,
                            failure_reason=exec_error,
                            failure_category=exec_category,
                            timeout_seconds=EXECUTION_POLICY.timeout_seconds,
                        )
                        st.error(f"Failure type: {exec_category}. {exec_error}")
                        st.stop()

                    st.session_state.metrics_query_success += 1
                    record_metric_event(
                        "query_execution",
                        success=True,
                        row_count=len(df),
                        execution_ms=int(elapsed * 1000),
                        row_limit=EXECUTION_POLICY.max_rows,
                    )
                    st.session_state.last_result_df = df
                    st.session_state.last_result_elapsed = elapsed
            except Exception as exc:
                failure_reason = str(exc)
                failure_category = query_logic.classify_execution_failure(failure_reason)
                st.session_state.metrics_query_failed += 1
                record_metric_event(
                    "query_execution",
                    success=False,
                    failure_reason=failure_reason,
                    failure_category=failure_category,
                )
                st.error(f"Failure type: {failure_category}. Query failed: {exc}")

    if st.session_state.last_result_df is not None:
        result_df = st.session_state.last_result_df
        result_elapsed = st.session_state.last_result_elapsed or 0.0
        st.markdown("### Step 4: Results")
        st.write(f"Rows: {len(result_df)} | Time: {result_elapsed:.3f}s")
        st.dataframe(result_df, width="stretch")
        st.markdown("### Chart")
        try_show_bar_chart(result_df)

        st.markdown("### Was this result helpful?")
        feedback_cols = st.columns([0.5, 0.5])
        question_hash = build_question_hash(st.session_state.question)
        with feedback_cols[0]:
            if st.button("Helpful", key="feedback_helpful", width="stretch"):
                st.session_state.metrics_feedback_up += 1
                st.session_state.feedback_last_rating = "up"
                st.session_state.feedback_last_question_hash = question_hash
                record_metric_event(
                    "user_feedback",
                    question_hash=question_hash,
                    rating="up",
                    has_result=True,
                )
                st.success("Feedback saved.")
        with feedback_cols[1]:
            if st.button("Not Helpful", key="feedback_not_helpful", width="stretch"):
                st.session_state.metrics_feedback_down += 1
                st.session_state.feedback_last_rating = "down"
                st.session_state.feedback_last_question_hash = question_hash
                record_metric_event(
                    "user_feedback",
                    question_hash=question_hash,
                    rating="down",
                    has_result=True,
                )
                st.info("Feedback saved.")


elif view == "Training Data":
    st.subheader("Training Data")
    st.caption(
        "Edit examples, save changes, optionally delete selected rows with confirmation, then train Vanna."
    )

    if st.session_state.training_working_df is None:
        st.session_state.training_working_df = load_training_examples()

    edited_df = st.data_editor(
        st.session_state.training_working_df,
        num_rows="dynamic",
        width="stretch",
        key="training_editor",
        disabled=["created_at", "updated_at"],
    )
    st.session_state.training_working_df = edited_df

    st.markdown("#### Delete Rows")
    row_labels: list[str] = []
    row_map: dict[str, int] = {}
    for idx, row in st.session_state.training_working_df.iterrows():
        question_preview = str(row.get("question", "")).strip()
        if len(question_preview) > 60:
            question_preview = question_preview[:57] + "..."
        label = f"{idx}: {question_preview if question_preview else '[empty question]'}"
        row_labels.append(label)
        row_map[label] = idx

    selected_for_delete = st.multiselect(
        "Select rows to delete",
        options=row_labels,
        key="training_delete_selection",
    )
    confirm_delete = st.checkbox(
        "I confirm permanent deletion of selected rows.",
        key="training_delete_confirm",
    )
    if st.button("Delete Selected Rows", width="content"):
        if not selected_for_delete:
            st.warning("Select at least one row before deleting.")
        elif not confirm_delete:
            st.warning("Please confirm deletion first.")
        else:
            delete_indices = [row_map[label] for label in selected_for_delete]
            st.session_state.training_working_df = (
                st.session_state.training_working_df.drop(index=delete_indices)
                .reset_index(drop=True)
            )
            record_metric_event(
                "training_rows_deleted",
                count=len(delete_indices),
            )
            st.success(f"Deleted {len(delete_indices)} row(s).")
            st.rerun()

    col_save, col_train = st.columns([0.25, 0.75])
    with col_save:
        if st.button("Save Examples", width="stretch"):
            cleaned_df, quality_stats = normalize_training_for_save(
                st.session_state.training_working_df
            )
            if quality_stats["dropped_missing_question_or_sql"] > 0:
                st.warning(
                    "Dropped "
                    f"{quality_stats['dropped_missing_question_or_sql']} row(s) with missing question or SQL."
                )
            if quality_stats["duplicate_question_sql_rows"] > 0:
                st.warning(
                    "Detected "
                    f"{quality_stats['duplicate_question_sql_rows']} duplicate question/SQL row(s)."
                )
            if (
                quality_stats["dropped_missing_question_or_sql"] > 0
                or quality_stats["duplicate_question_sql_rows"] > 0
            ):
                record_metric_event(
                    "training_quality_warning",
                    dropped_missing_question_or_sql=quality_stats[
                        "dropped_missing_question_or_sql"
                    ],
                    duplicate_question_sql_rows=quality_stats[
                        "duplicate_question_sql_rows"
                    ],
                )

            save_training_examples(cleaned_df)
            st.session_state.training_working_df = load_training_examples()
            record_metric_event(
                "training_examples_saved",
                row_count=len(st.session_state.training_working_df),
                dropped_missing_question_or_sql=quality_stats[
                    "dropped_missing_question_or_sql"
                ],
                duplicate_question_sql_rows=quality_stats["duplicate_question_sql_rows"],
            )
            st.success("Saved with timestamps.")
    with col_train:
        if st.button("Train Model", width="stretch"):
            training_df, _ = normalize_training_for_save(st.session_state.training_working_df)
            with st.spinner("Training model..."):
                vn = get_vanna()
                train_from_examples(vn, training_df)
            record_metric_event(
                "training_triggered",
                row_count=len(training_df),
            )
            st.success("Training complete.")
