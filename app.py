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
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from pathlib import Path

import duckdb
import pandas as pd
import streamlit as st

from sql_guard import validate_read_only_sql
from vannaagent import (
    get_vanna,
    get_exact_training_sql,
    load_training_examples,
    save_training_examples,
    train_from_examples,
)

DUCKDB_PATH = "mxquerychat.duckdb"
ICON_PATH = Path("docs/images/icon.png")
LOGO_PATH = Path("docs/images/logo.png")
MODEL_TIMEOUT_SECONDS = 65

EXAMPLE_QUESTIONS = [
    "What is the total revenue in 2025?",
    "Show revenue per month for 2024.",
    "Which ticket types generate the most revenue?",
    "Show revenue by federal state for 2025.",
]

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
        "sql_cache": {},
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def reset_question_flow():
    st.session_state.question = ""
    st.session_state.generated_sql = ""
    st.session_state.last_result_df = None


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
    if st.button("New Chat", use_container_width=True):
        reset_question_flow()
        st.rerun()

if LOGO_PATH.exists():
    st.sidebar.image(str(LOGO_PATH), use_container_width=True)
st.sidebar.header("Navigation")
view = st.sidebar.radio("Choose page", ["Ask", "Training Data", "Schema"])


if view == "Ask":
    st.markdown("### Step 1: Ask a data question")
    st.markdown('<div class="mx-card">Try one of these examples first.</div>', unsafe_allow_html=True)

    ex_cols = st.columns(len(EXAMPLE_QUESTIONS))
    for idx, ex_question in enumerate(EXAMPLE_QUESTIONS):
        with ex_cols[idx]:
            if st.button(f"Example {idx + 1}", use_container_width=True):
                st.session_state.question = ex_question

    st.session_state.question = st.text_input(
        "Question",
        value=st.session_state.question,
        placeholder="Example: Show revenue by state for 2025.",
    )

    if st.button("Generate SQL", type="primary"):
        local_block = get_local_guardrail_message(st.session_state.question)
        if local_block:
            st.session_state.generated_sql = ""
            st.warning(local_block)
        else:
            question_text = st.session_state.question.strip()
            cache_key = question_text.lower()
            handled_fast_path = False

            # 1) Session cache (fastest)
            cached_sql = st.session_state.sql_cache.get(cache_key)
            if cached_sql:
                st.session_state.generated_sql = cached_sql
                st.success("Used cached SQL (instant).")
                handled_fast_path = True

            # 2) Exact match in training examples (fast, no LLM)
            if not handled_fast_path:
                examples_df = load_training_examples()
                direct_sql = get_exact_training_sql(question_text, examples_df)
                if direct_sql:
                    st.session_state.generated_sql = direct_sql
                    st.session_state.sql_cache[cache_key] = direct_sql
                    st.success("Used training example SQL (instant).")
                    handled_fast_path = True

            if not handled_fast_path:
                with st.spinner("Generating SQL..."):
                    vn = get_vanna_cached()
                    sql, sql_error = run_with_timeout(
                        lambda: vn.generate_sql(question_text),
                        MODEL_TIMEOUT_SECONDS,
                    )

                    if sql_error:
                        st.session_state.generated_sql = ""
                        st.error(
                            "Could not generate SQL right now. "
                            f"Reason: {sql_error}. "
                            "Tip: first call can be slow (model cold start). Try once more."
                        )
                    else:
                        st.session_state.generated_sql = sql
                        st.session_state.sql_cache[cache_key] = sql

    if st.session_state.generated_sql:
        st.markdown("### Step 2: Review SQL")
        st.session_state.generated_sql = st.text_area(
            "Generated SQL (editable)",
            value=st.session_state.generated_sql,
            height=170,
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
                    df, elapsed = run_read_only_query(st.session_state.generated_sql)
                    st.session_state.last_result_df = df
                st.markdown("### Step 4: Results")
                st.write(f"Rows: {len(df)} | Time: {elapsed:.3f}s")
                st.dataframe(df, use_container_width=True)
                st.markdown("### Chart")
                try_show_bar_chart(df)
            except Exception as exc:
                st.error(f"Query failed: {exc}")


elif view == "Training Data":
    st.subheader("Training Data")
    st.caption("Edit examples, save them, then train Vanna.")

    examples_df = load_training_examples()
    edited_df = st.data_editor(
        examples_df,
        num_rows="dynamic",
        use_container_width=True,
        key="training_editor",
    )

    col_save, col_train = st.columns([0.25, 0.75])
    with col_save:
        if st.button("Save Examples", use_container_width=True):
            save_training_examples(edited_df)
            st.success("Saved.")
    with col_train:
        if st.button("Train Model", use_container_width=True):
            with st.spinner("Training model..."):
                vn = get_vanna()
                train_from_examples(vn, edited_df)
            st.success("Training complete.")


else:
    st.subheader("Database Schema")
    try:
        schema_tree = get_schema_tree()
        table_names = list(schema_tree.keys())
        selected = st.selectbox("Choose a table", table_names)
        rows = schema_tree[selected]
        schema_df = pd.DataFrame(rows, columns=["column_name", "data_type"])
        st.dataframe(schema_df, use_container_width=True)
    except Exception as exc:
        st.error(f"Could not load schema: {exc}")
