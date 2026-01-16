"""
app.py

mxQueryChat MVP (Streamlit)
- DuckDB schema tree (sidebar)
- New Question view:
  - ask question
  - generate SQL + explanation
  - user reviews/edits SQL
  - read-only guard
  - run query
  - show table + simple bar chart (if possible)
- Training Data view:
  - edit examples (question/sql/description)
  - save
  - train model
"""

import time
import duckdb
import pandas as pd
import streamlit as st

from sql_guard import validate_read_only_sql
from vannaagent import get_vanna, load_training_examples, save_training_examples, train_from_examples

DUCKDB_PATH = "mxquerychat.duckdb"


# -----------------------------
# Helpers
# -----------------------------
def get_schema_tree() -> dict:
    """
    Read DuckDB schema info and return:
    {table_name: [(col_name, col_type), ...], ...}
    """
    con = duckdb.connect(DUCKDB_PATH, read_only=True)

    tables = con.execute("""
        SELECT table_name
        FROM information_schema.tables
        WHERE table_schema = 'main'
        ORDER BY table_name
    """).fetchall()

    schema = {}
    for (table_name,) in tables:
        cols = con.execute(f"""
            SELECT column_name, data_type
            FROM information_schema.columns
            WHERE table_name = '{table_name}'
            ORDER BY ordinal_position
        """).fetchall()
        schema[table_name] = cols

    con.close()
    return schema


def run_read_only_query(sql: str) -> tuple[pd.DataFrame, float]:
    """
    Execute SQL on DuckDB in read-only mode and return (df, elapsed_seconds).
    """
    con = duckdb.connect(DUCKDB_PATH, read_only=True)
    start = time.time()
    df = con.execute(sql).df()
    elapsed = time.time() - start
    con.close()
    return df, elapsed


def try_show_bar_chart(df: pd.DataFrame) -> None:
    """
    Simple chart rule for MVP:
    If df has at least 2 columns and the 2nd is numeric, chart it.
    """
    if df.shape[1] < 2:
        return

    x_col = df.columns[0]
    y_col = df.columns[1]

    if pd.api.types.is_numeric_dtype(df[y_col]):
        chart_df = df[[x_col, y_col]].copy()
        st.bar_chart(chart_df.set_index(x_col))
    else:
        st.info("No bar chart shown (2nd column is not numeric).")


# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="mxQueryChat", layout="wide")

st.title("mxQueryChat")

# Session state init
if "question" not in st.session_state:
    st.session_state.question = ""
if "generated_sql" not in st.session_state:
    st.session_state.generated_sql = ""
if "explanation" not in st.session_state:
    st.session_state.explanation = ""
if "last_result_df" not in st.session_state:
    st.session_state.last_result_df = None

# Top-right: New Chat (reset question flow, keep training data)
col_left, col_right = st.columns([0.85, 0.15])
with col_right:
    if st.button("🧹 New Chat", use_container_width=True):
        st.session_state.question = ""
        st.session_state.generated_sql = ""
        st.session_state.explanation = ""
        st.session_state.last_result_df = None
        st.rerun()

# Sidebar navigation + schema
view = st.sidebar.radio("Navigation", ["New Question", "Training Data"])

st.sidebar.markdown("---")
st.sidebar.subheader("DuckDB Schema")

try:
    schema_tree = get_schema_tree()
    for table_name, cols in schema_tree.items():
        with st.sidebar.expander(table_name):
            for col_name, col_type in cols:
                st.sidebar.write(f"- `{col_name}` ({col_type})")
except Exception as e:
    st.sidebar.error(f"Schema load failed: {e}")

# -----------------------------
# View: New Question
# -----------------------------
if view == "New Question":
    st.subheader("Ask a question")

    st.session_state.question = st.text_area(
        "Your question (German or English)",
        value=st.session_state.question,
        height=80,
        placeholder="z.B. Zeige den Umsatz nach Bundesland für 2025.",
    )

    col_a, col_b = st.columns([0.2, 0.8])

    with col_a:
        if st.button("⚡ Generate SQL", use_container_width=True):
            with st.spinner("Generating SQL with Vanna..."):
                vn = get_vanna()

                # Generate SQL
                sql = vn.generate_sql(st.session_state.question)

                # Generate explanation (best effort)
                try:
                    explanation = vn.generate_explanation(sql)
                except Exception:
                    explanation = "Explanation not available (model/config)."

                st.session_state.generated_sql = sql
                st.session_state.explanation = explanation

    if st.session_state.generated_sql:
        st.markdown("### Generated SQL (editable)")
        st.session_state.generated_sql = st.text_area(
            "SQL",
            value=st.session_state.generated_sql,
            height=180,
        )

        st.markdown("### Explanation")
        st.write(st.session_state.explanation)

        st.markdown("### Run query (read-only)")
        is_ok, message = validate_read_only_sql(st.session_state.generated_sql)
        if is_ok:
            st.success("SQL guard: OK (read-only)")
        else:
            st.error(f"SQL guard blocked this query: {message}")

        if st.button("▶ Run Query", disabled=not is_ok):
            with st.spinner("Running query on DuckDB..."):
                df, elapsed = run_read_only_query(st.session_state.generated_sql)
                st.session_state.last_result_df = df

                st.write(f"Rows: {len(df)} | Execution time: {elapsed:.3f}s")
                st.dataframe(df, use_container_width=True)

                st.markdown("### Chart (optional)")
                try_show_bar_chart(df)

# -----------------------------
# View: Training Data
# -----------------------------
if view == "Training Data":
    st.subheader("Training Data (Vanna)")

    st.caption("Edit examples below. Use realistic questions + correct SQL. No sensitive data.")

    examples_df = load_training_examples()

    edited_df = st.data_editor(
        examples_df,
        num_rows="dynamic",
        use_container_width=True,
        key="training_editor",
    )

    col1, col2 = st.columns([0.2, 0.8])

    with col1:
        if st.button("💾 Save Examples", use_container_width=True):
            save_training_examples(edited_df)
            st.success("Saved training examples.")

    st.markdown("---")
    if st.button("🧠 Train Model (Update Vanna)"):
        with st.spinner("Training Vanna (synchronous)..."):
            vn = get_vanna()
            train_from_examples(vn, edited_df)
            st.success("Training completed. New generations should improve.")
