"""Streamlit app: question input → SQL generation → safety check → DuckDB execution."""

import json
import re
import time
import hashlib
import os
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from datetime import datetime
from pathlib import Path
from typing import Optional

import duckdb
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from src.core import query_logic
from src.core.query_logic import extract_requested_years, nearest_year
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
from src.utils.telemetry import configure_app_logging, record_metric_event
from sql_guard import validate_read_only_sql
from vannaagent import (
    get_vanna,
    get_exact_training_sql,
    load_training_examples,
    normalize_training_for_save,
    save_training_examples,
    train_from_examples,
    upsert_training_example,
)

DUCKDB_PATH = "mxquerychat.duckdb"
ICON_PATH = Path("docs/images/icon.png")
LOGO_PATH = Path("docs/images/logo.png")
MODEL_TIMEOUT_SECONDS = 65
EXECUTION_POLICY = ExecutionPolicy()
HISTORY_FILE = "chat_history.json"
HISTORY_MAX_ENTRIES = 25
configure_app_logging()

EXAMPLE_QUESTIONS = [
    "Which 5 federal states generated the most ticket revenue in 2025?",
    "Compare monthly ticket revenue between 2024 and 2025 — which months grew?",
    "Show top 5 ticket types by revenue in 2025 with exact euro amounts.",
    "For each tariff association, show total revenue by federal state for 2025.",
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


def build_suggested_questions(question: str) -> list[str]:
    """Return up to 4 specific suggestions matched to the user's intent."""
    years = get_available_years()
    requested = extract_requested_years(question)
    q_lower = (question or "").lower()

    latest = max(years) if years else 2025
    prev = next((y for y in reversed(years) if y < latest), latest - 1) if years else latest - 1
    yr = (nearest_year(requested[0], years) or latest) if requested and years else latest

    if any(k in q_lower for k in ("state", "bundesland", "region", "federal")):
        candidates = [
            f"Which 5 federal states had the highest ticket revenue in {yr}?",
            f"Show revenue by federal state for {yr}, broken down by ticket type.",
            f"How did revenue per federal state change from {prev} to {yr}?",
            f"For each state in {yr}, which month generated the most ticket revenue?",
        ]
    elif any(k in q_lower for k in ("month", "monat", "trend", "quarter", "growth", "change", "compar")):
        candidates = [
            f"Show monthly ticket revenue for {yr} — which month peaked?",
            f"Compare revenue per month between {prev} and {yr}.",
            f"Show revenue per quarter for {yr} across all ticket types.",
            f"Which federal state had the highest revenue growth from {prev} to {yr}?",
        ]
    elif any(k in q_lower for k in ("ticket", "type", "product", "produkt")):
        candidates = [
            f"Which ticket types generated the top 5 revenues in {yr}?",
            f"Show revenue per ticket type broken down by federal state for {yr}.",
            f"Compare ticket type revenues between {prev} and {yr}.",
            f"Show average sale price vs. catalog price per ticket type in {yr}.",
        ]
    elif any(k in q_lower for k in ("tariff", "tarif", "verbund", "association")):
        candidates = [
            f"Show total revenue per tariff association for {yr}, ranked highest to lowest.",
            f"For each tariff association, which federal state contributed the most revenue in {yr}?",
            f"Show revenue by tariff association and ticket type for {yr}.",
            f"Compare tariff association revenues between {prev} and {yr}.",
        ]
    elif any(k in q_lower for k in ("city", "postal", "plz", "ort", "zip", "office", "meldestelle")):
        candidates = [
            f"Show the top 10 cities by ticket revenue in {yr}.",
            f"Which postal codes had the highest revenue in {yr}? Show top 10 with state.",
            f"Which reporting offices delivered the most revenue in {yr}?",
            f"Show top 20 postal codes by revenue with city and state for {yr}.",
        ]
    else:
        candidates = [
            f"Which 5 federal states generated the most ticket revenue in {yr}?",
            f"Show monthly ticket revenue for {yr} — identify the peak month.",
            f"Which ticket types earned the most in {yr}? Show top 5 with amounts.",
            f"For each tariff association, show total revenue by federal state for {yr}.",
        ]

    seen: set[str] = set()
    result = []
    for item in candidates:
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


def validate_sql_compiles(sql: str) -> str:
    """
    Return empty string if SQL compiles against DuckDB schema, else error message.
    Uses EXPLAIN as a dry-run parse: validates table/column names without executing the query.
    """
    if not sql:
        return "No SQL generated."
    con = duckdb.connect(DUCKDB_PATH, read_only=True)
    try:
        # EXPLAIN parses and plans without executing — safe for compile-time validation.
        con.execute(f"EXPLAIN {sql}")
        return ""
    except Exception as exc:
        return str(exc)
    finally:
        con.close()


def run_with_timeout(fn, timeout_seconds: int):
    """Run fn in a thread; return (result, None) or (None, error) on timeout/exception."""
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


def try_show_bar_chart(df: pd.DataFrame) -> None:
    """Show a bar chart when the data is suitable; explain briefly when not."""
    if df.shape[0] == 0 or df.shape[1] < 2:
        return

    METRIC_HINTS = {"umsatz", "revenue", "anzahl", "count", "sum", "total", "amount", "betrag"}
    DIMENSION_HINTS = {"monat", "month", "bundesland", "state", "region", "name", "tarif", "ticket", "plz", "postal"}

    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    text_cols = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]

    if not numeric_cols:
        st.caption("No chart: no numeric values to plot.")
        return

    # Y: prefer a metric-named column, else last numeric (likely the aggregated value)
    y_col = next(
        (c for c in numeric_cols if any(h in c.lower() for h in METRIC_HINTS)),
        numeric_cols[-1],
    )

    # X: prefer a text dimension column, then a numeric dimension (e.g. monat)
    x_col = next(
        (c for c in text_cols if any(h in c.lower() for h in DIMENSION_HINTS)),
        text_cols[0] if text_cols else None,
    )
    if x_col is None:
        remaining_numeric = [c for c in numeric_cols if c != y_col]
        x_col = next(
            (c for c in remaining_numeric if any(h in c.lower() for h in DIMENSION_HINTS)),
            None,
        )

    if x_col is None:
        st.caption("No chart: couldn't identify a category column.")
        return
    if df[x_col].nunique() < 2:
        st.caption("No chart: only one group in results.")
        return
    if len(df) > 50:
        st.caption("No chart: too many rows to display meaningfully.")
        return

    st.bar_chart(df[[x_col, y_col]].set_index(x_col))


def load_chat_history() -> list:
    if not os.path.exists(HISTORY_FILE):
        return []
    try:
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []


def _save_chat_history(history: list) -> None:
    try:
        with open(HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(history, f, ensure_ascii=False, indent=2)
    except Exception:
        pass


def add_history_entry(question: str, generated_sql: str, generation_notes: list) -> str:
    """Prepend a new entry; return its id."""
    entry_id = f"{time.time():.6f}"
    entry = {
        "id": entry_id,
        "timestamp": datetime.now().isoformat(),
        "question": question,
        "generated_sql": generated_sql,
        "generation_notes": list(generation_notes),
        "result_rows": None,
        "result_elapsed": None,
        "result_data": None,
    }
    history = load_chat_history()
    history.insert(0, entry)
    history = history[:HISTORY_MAX_ENTRIES]
    _save_chat_history(history)
    st.session_state.chat_history = history
    return entry_id


def update_history_with_result(entry_id: str, df: pd.DataFrame, elapsed: float) -> None:
    """Attach query results to an existing history entry."""
    history = load_chat_history()
    for entry in history:
        if entry["id"] == entry_id:
            entry["result_rows"] = len(df)
            entry["result_elapsed"] = elapsed
            # to_json handles numpy/datetime types; parse back to plain Python for storage
            entry["result_data"] = json.loads(df.head(500).to_json(orient="records"))
            break
    _save_chat_history(history)
    st.session_state.chat_history = history


def init_state():
    """
    Initialise all session-state keys with their defaults on first load.
    Keys that already exist are left untouched, so partial state survives reruns.
    """
    defaults = {
        # Active question and its generated SQL (both editable by the user).
        "question": "",
        "generated_sql": "",

        # Last successful query result and its wall-clock execution time.
        "last_result_df": None,
        "last_result_elapsed": None,

        # Lowercase question → SQL; populated on every successful generation for instant re-use.
        "sql_cache": {},

        # "Did you mean?" suggestions shown after a failed generation.
        "suggestions": [],

        # Trace of how the current SQL was produced (cache / template / LLM).
        "generation_notes": [],

        # Session-scoped counters written to the sidebar metrics panel.
        "metrics_questions_total": 0,
        "metrics_generation_success": 0,
        "metrics_generation_failed": 0,
        "metrics_blocked_requests": 0,
        "metrics_cache_hits": 0,
        "metrics_query_success": 0,
        "metrics_query_failed": 0,
        # Feedback deduplication: track last rating and which question it was for.
        "feedback_last_rating": None,
        "feedback_last_question_hash": "no_question",

        "generated_explanation": "",

        # Working copy of the training CSV for the editor; None until the page is opened.
        "training_working_df": None,

        # Chat history loaded from disk; updated after every Send and Run Query.
        "chat_history": load_chat_history(),
        "history_selected_id": None,
        "history_current_entry_id": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def reset_question_flow():
    query_logic.reset_question_flow_state(st.session_state)


def build_question_hash(question: str) -> str:
    """16-char SHA-256 prefix of the question, used for feedback deduplication."""
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
view = st.sidebar.radio("Choose page", ["New Question", "Training Data", "Chat History"])
st.sidebar.markdown("---")
st.sidebar.subheader("Session Metrics")
st.sidebar.metric("Questions", st.session_state.metrics_questions_total)
st.sidebar.metric("SQL Generated", st.session_state.metrics_generation_success)
st.sidebar.metric("Generation Failed", st.session_state.metrics_generation_failed)
st.sidebar.metric("Blocked Requests", st.session_state.metrics_blocked_requests)
st.sidebar.metric("Query Success", st.session_state.metrics_query_success)
st.sidebar.metric("Query Failed", st.session_state.metrics_query_failed)

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

if view == "Chat History":
    st.sidebar.markdown("---")
    st.sidebar.subheader("Recent Queries")
    _sidebar_history = st.session_state.chat_history
    if not _sidebar_history:
        st.sidebar.caption("No history yet.")
    else:
        for _h_entry in _sidebar_history:
            _h_ts = (_h_entry.get("timestamp") or "")[:16].replace("T", " ")
            _h_q = (_h_entry.get("question") or "")[:38]
            _h_label = f"{_h_ts}\n{_h_q}{'…' if len(_h_entry.get('question','')) > 38 else ''}"
            _h_active = st.session_state.history_selected_id == _h_entry["id"]
            if st.sidebar.button(
                _h_label,
                key=f"sidebar_h_{_h_entry['id']}",
                use_container_width=True,
                type="primary" if _h_active else "secondary",
            ):
                st.session_state.history_selected_id = _h_entry["id"]
                st.rerun()

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
            cache_key = question_text.lower()  # lowercase so casing differences don't create misses
            handled_fast_path = False  # set True by whichever generation tier succeeds first

            # Early-out: reject years missing from the database before hitting the LLM.
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

        # Save this interaction to persistent chat history.
        _hist_id = add_history_entry(
            st.session_state.question,
            st.session_state.generated_sql,
            st.session_state.generation_notes,
        )
        st.session_state.history_current_entry_id = _hist_id

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
        if st.button("Save to Training Examples", key="save_sql_to_training"):
            q = st.session_state.question.strip()
            s = st.session_state.generated_sql.strip()
            # Upsert by normalised question — re-saving after an edit updates the row, not duplicates.
            upsert_training_example(q, s)
            # Refresh session cache so the corrected SQL is used on the next re-ask.
            st.session_state.sql_cache[q.lower()] = s
            record_metric_event(
                "training_examples_saved",
                row_count=1,
                dropped_missing_question_or_sql=0,
                duplicate_question_sql_rows=0,
            )
            st.success("Saved to training examples. Future queries will use this SQL.")

        with st.expander("Explain this SQL", expanded=False):
            if st.button("Explain", key="explain_sql"):
                st.session_state.generated_explanation = query_logic.explain_sql_brief(
                    st.session_state.generated_sql
                )
            if st.session_state.get("generated_explanation"):
                st.info(st.session_state.generated_explanation)

    if st.session_state.generated_sql:
        st.markdown("### Step 3: Safety check")
        # First gate: reject any SQL that is not a plain SELECT/WITH (no writes, no DDL).
        is_ok, guard_message = validate_read_only_sql(st.session_state.generated_sql)
        if is_ok:
            st.success("Read-only check passed.")
        else:
            st.error(guard_message)

        # Run Query is disabled when the read-only check fails so the button cannot be clicked.
        if st.button("Run Query", disabled=not is_ok):
            try:
                with st.spinner("Running query..."):
                            # Re-validate inside the handler in case session state changed between renders.
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

                    # Complexity gate: block by join count, CTE count, and total SQL length.
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

                    # Wrap the user's query in an outer SELECT … LIMIT to cap returned rows.
                    # This prevents accidental full-table scans from flooding the UI.
                    limited_sql = apply_row_limit(
                        st.session_state.generated_sql,
                        EXECUTION_POLICY.max_rows,
                    )
                    # Execute in a subprocess with a hard timeout so a slow query cannot hang the app.
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
                    if st.session_state.get("history_current_entry_id"):
                        update_history_with_result(
                            st.session_state.history_current_entry_id, df, elapsed
                        )
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

        question_hash = build_question_hash(st.session_state.question)
        if st.button("SQL was correct — save as example", key="feedback_helpful", width="stretch"):
            st.session_state.feedback_last_rating = "up"
            st.session_state.feedback_last_question_hash = question_hash
            record_metric_event(
                "user_feedback",
                question_hash=question_hash,
                rating="up",
                has_result=True,
            )
            q = st.session_state.question.strip()
            s = st.session_state.generated_sql.strip()
            if q and s:
                upsert_training_example(q, s)
                st.session_state.sql_cache[q.lower()] = s
            st.success("Saved to training examples.")


elif view == "Training Data":
    st.subheader("Training Data")
    st.caption(
        "Edit examples, save changes, optionally delete selected rows with confirmation, then train Vanna."
    )

    if st.session_state.training_working_df is None:
        _df = load_training_examples()
        _df["_ts"] = pd.to_datetime(_df["created_at"], errors="coerce", utc=True)
        st.session_state.training_working_df = (
            _df.sort_values("_ts", ascending=False, na_position="last")
            .drop(columns=["_ts"])
            .reset_index(drop=True)
        )

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
            _df = load_training_examples()
            _df["_ts"] = pd.to_datetime(_df["created_at"], errors="coerce", utc=True)
            st.session_state.training_working_df = (
                _df.sort_values("_ts", ascending=False, na_position="last")
                .drop(columns=["_ts"])
                .reset_index(drop=True)
            )
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


elif view == "Chat History":
    st.markdown("### Chat History")
    history = st.session_state.chat_history
    if not history:
        st.info("No history yet. Ask a question on the New Question page to get started.")
    else:
        selected_id = st.session_state.history_selected_id
        selected = next((e for e in history if e["id"] == selected_id), None)

        if selected is None:
            # List view — newest first (already ordered)
            for entry in history:
                ts = (entry.get("timestamp") or "")[:16].replace("T", " ")
                q = entry.get("question") or ""
                has_sql = bool(entry.get("generated_sql"))
                has_result = entry.get("result_rows") is not None
                with st.container(border=True):
                    c1, c2 = st.columns([0.85, 0.15])
                    with c1:
                        st.write(f"**{q[:100]}{'…' if len(q) > 100 else ''}**")
                        st.caption(
                            f"{ts} | SQL: {'yes' if has_sql else 'no'}"
                            + (f" | {entry['result_rows']} rows" if has_result else "")
                        )
                    with c2:
                        if st.button("Open", key=f"open_{entry['id']}"):
                            st.session_state.history_selected_id = entry["id"]
                            st.rerun()
        else:
            # Detail view
            if st.button("← Back"):
                st.session_state.history_selected_id = None
                st.rerun()

            ts = (selected.get("timestamp") or "")[:19].replace("T", " ")
            st.caption(f"Asked: {ts}")
            st.markdown(f"**Question:** {selected.get('question', '')}")

            notes = selected.get("generation_notes") or []
            if notes:
                with st.expander("Generation notes", expanded=False):
                    for note in notes:
                        st.info(note)

            sql = selected.get("generated_sql") or ""
            if sql:
                st.markdown("**Generated SQL:**")
                st.code(sql, language="sql")
            else:
                st.warning("No SQL was generated for this query.")

            result_data = selected.get("result_data")
            if result_data is not None:
                result_df = pd.DataFrame(result_data)
                rows = selected.get("result_rows", len(result_df))
                elapsed = selected.get("result_elapsed") or 0.0
                st.markdown(f"**Results:** {rows} rows | {elapsed:.3f}s")
                st.dataframe(result_df, use_container_width=True)
                try_show_bar_chart(result_df)
            else:
                st.info("Query was not executed or results were not available.")
