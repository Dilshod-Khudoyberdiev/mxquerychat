import logging
import re
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from pathlib import Path

import pandas as pd

# Vanna imports
try:
    from vanna.chromadb import ChromaDB_VectorStore
    from vanna.ollama import Ollama
    from vanna.flask import VannaFlaskApp
except ModuleNotFoundError:
    from vanna.legacy.chromadb import ChromaDB_VectorStore
    from vanna.legacy.ollama import Ollama
    from vanna.legacy.flask import VannaFlaskApp

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

DUCKDB_PATH = "mxquerychat.duckdb"            # local DB file
CHROMA_PATH = "vanna_chroma_store_demo"       # local vector store folder
OLLAMA_MODEL = "mistral"
DDL_OUTPUT_PATH = Path("docs/schema_ddl.sql") # save DDL here
TRAINING_EXAMPLES_CSV = Path("training_data/training_examples.csv") 
DEMO_QUESTIONS_PATH = Path("docs/demo_questions.md")
EXTRA_QUESTIONS_PATH = Path("docs/tricky_questions.md")
MAX_PROMPT_TOKENS = 3500
MAX_GENERATE_SQL_SECONDS = 30
MAX_REWRITE_SECONDS = 8
INITIAL_PROMPT = (
    "You are a SQL expert for the mxquerychat DuckDB. "
    "Only answer with SQL using the provided schema. "
    "Do not answer with natural language. "
    "If the question is out of scope, return a SQL SELECT with a short message."
)
GUARDRAIL_PREFIX = "GUARDRAIL:"

DOMAIN_HINTS = {
    "ticket",
    "tickets",
    "umsatz",
    "revenue",
    "sales",
    "tarif",
    "tarifverbund",
    "bundesland",
    "state",
    "plz",
    "postleitzahl",
    "meldestelle",
    "plan",
    "angebot",
    "verteilung",
    "region",
    "monat",
    "jahr",
    "month",
    "year",
}

SMALL_TALK_PATTERNS = [
    r"\bhi\b",
    r"\bhello\b",
    r"\bhey\b",
    r"\bgood (morning|afternoon|evening)\b",
    r"what is your name",
    r"who are you",
    r"how are you",
]

OUT_OF_SCOPE_PATTERNS = [
    r"presentation",
    r"slides",
    r"powerpoint",
    r"exam",
    r"homework",
    r"essay",
    r"write my",
    r"cover letter",
    r"cv",
    r"resume",
]

SMALL_TALK_RESPONSE = "Hello! Ask me a question about the mxquerychat dataset."
OUT_OF_SCOPE_RESPONSE = (
    "I can only answer questions about the mxquerychat DuckDB dataset "
    "(ticket sales, revenue, tariff associations, states, postal codes, plans)."
)
READ_ONLY_PATTERNS = [
    r"\binsert\b",
    r"\bupdate\b",
    r"\bdelete\b",
    r"\btruncate\b",
    r"\bdrop\b",
    r"\balter\b",
    r"\bmerge\b",
    r"\bcreate\b",
    r"\bgrant\b",
    r"\brevoke\b",
]
READ_ONLY_RESPONSE = "This demo is read-only. I cannot modify data or schema."
SLOW_MODEL_RESPONSE = (
    "The model took too long to answer. "
    "Please try a shorter data question with clear filters (year, month, ticket, state)."
)
INVALID_SQL_RESPONSE = (
    "I could not produce valid SQL for that question. "
    "Please rephrase with dataset terms like revenue, month, year, ticket, state, or PLZ."
)
NON_DATA_PATTERNS = [
    r"\bpoem\b",
    r"\bsong\b",
    r"\blyrics\b",
    r"\bjoke\b",
    r"\bstory\b",
    r"\bemail\b",
    r"\bessay\b",
    r"\bcover letter\b",
    r"\bresume\b",
]


class VannaAgent(ChromaDB_VectorStore, Ollama):
    """
    Vanna agent = (Vector Store) + (LLM)
    - ChromaDB stores training embeddings locally
    - Ollama runs the LLM locally (mistral)
    """
    def __init__(self, config=None):
        Ollama.__init__(self, config=config)
        ChromaDB_VectorStore.__init__(self, config=config)
        self.domain_terms = set()
        self.demo_sql_lookup = {}

    def generate_sql(self, question: str, allow_llm_to_see_data=False, **kwargs) -> str:
        guardrail = get_guardrail_message(question, self.domain_terms)
        if guardrail:
            return format_guardrail_message(guardrail)
        normalized = normalize_question(question)
        if normalized and normalized in self.demo_sql_lookup:
            return self.demo_sql_lookup[normalized]

        sql = run_with_timeout(
            lambda: super(VannaAgent, self).generate_sql(
                question, allow_llm_to_see_data=allow_llm_to_see_data, **kwargs
            ),
            timeout_seconds=MAX_GENERATE_SQL_SECONDS,
        )
        if sql is None:
            return format_guardrail_message(SLOW_MODEL_RESPONSE)
        if not looks_like_sql(sql):
            return format_guardrail_message(INVALID_SQL_RESPONSE)
        return sql

    def run_sql(self, sql: str, **kwargs):
        if isinstance(sql, str) and sql.startswith(GUARDRAIL_PREFIX):
            message = sql[len(GUARDRAIL_PREFIX):].strip()
            return pd.DataFrame({"warning": [message]})
        return super().run_sql(sql, **kwargs)

    def generate_rewritten_question(self, last_question: str, new_question: str) -> str:
        guardrail = get_guardrail_message(new_question, self.domain_terms)
        if guardrail:
            return new_question
        rewritten = run_with_timeout(
            lambda: super(VannaAgent, self).generate_rewritten_question(
                last_question, new_question
            ),
            timeout_seconds=MAX_REWRITE_SECONDS,
        )
        return rewritten if rewritten else new_question

def format_guardrail_message(message: str) -> str:
    return f"{GUARDRAIL_PREFIX} {message}"

def run_with_timeout(fn, timeout_seconds: int):
    executor = ThreadPoolExecutor(max_workers=1)
    future = executor.submit(fn)
    try:
        return future.result(timeout=timeout_seconds)
    except FutureTimeoutError:
        future.cancel()
        logging.warning("Timed out after %ss waiting for model response.", timeout_seconds)
        return None
    except Exception:
        logging.exception("Model call failed.")
        return None
    finally:
        executor.shutdown(wait=False, cancel_futures=True)


def normalize_question(question: str) -> str:
    if not question:
        return ""
    normalized = re.sub(r"[^a-z0-9]+", " ", question.lower()).strip()
    return " ".join(normalized.split())


def looks_like_sql(text: str) -> bool:
    if not text:
        return False
    return bool(re.search(r"\b(select|with)\b", text.strip(), re.IGNORECASE))


def is_small_talk(question: str) -> bool:
    if not question:
        return False
    q_lower = question.lower()
    return any(re.search(pattern, q_lower) for pattern in SMALL_TALK_PATTERNS)


def is_out_of_scope_question(question: str) -> bool:
    if not question:
        return False
    q_lower = question.lower()
    return any(re.search(pattern, q_lower) for pattern in OUT_OF_SCOPE_PATTERNS)

def is_read_only_violation(question: str) -> bool:
    if not question:
        return False
    q_lower = question.lower()
    return any(re.search(pattern, q_lower) for pattern in READ_ONLY_PATTERNS)

def is_non_data_request(question: str) -> bool:
    if not question:
        return False
    q_lower = question.lower()
    return any(re.search(pattern, q_lower) for pattern in NON_DATA_PATTERNS)


def is_subject_related_question(question: str, domain_terms: set[str]) -> bool:
    if not question:
        return False
    q_lower = question.lower()
    return any(term in q_lower for term in domain_terms)

def get_guardrail_message(question: str, domain_terms: set[str]) -> str:
    if is_small_talk(question):
        return SMALL_TALK_RESPONSE
    if is_read_only_violation(question):
        return READ_ONLY_RESPONSE
    if is_non_data_request(question):
        return OUT_OF_SCOPE_RESPONSE
    if is_out_of_scope_question(question):
        return OUT_OF_SCOPE_RESPONSE
    if domain_terms and not is_subject_related_question(question, domain_terms):
        return OUT_OF_SCOPE_RESPONSE + " Please ask a data question about those topics."
    return ""


def build_domain_terms(vn: VannaAgent) -> set[str]:
    df = vn.run_sql("""
        SELECT table_name, column_name
        FROM information_schema.columns
        WHERE table_schema = 'main'
        ORDER BY table_name, ordinal_position
    """)

    terms = set(DOMAIN_HINTS)
    for _, row in df.iterrows():
        for value in (row["table_name"], row["column_name"]):
            value_str = str(value).lower()
            for token in re.split(r"[^a-z0-9]+", value_str):
                if len(token) >= 3:
                    terms.add(token)
    return terms


def export_all_table_ddls(vn: VannaAgent) -> str:
    """
    Export DDL (CREATE TABLE statements) for all tables in DuckDB.
    We try SHOW CREATE TABLE first; if unavailable, we build a basic CREATE TABLE.
    Returns one big SQL string.
    """
    tables_df = vn.run_sql("""
        SELECT table_name
        FROM information_schema.tables
        WHERE table_schema = 'main'
        ORDER BY table_name
    """)

    if "table_name" in tables_df.columns:
        tables = tables_df["table_name"].tolist()
    else:
        tables = [row[0] for row in tables_df.values.tolist()]

    ddl_chunks = []
    ddl_chunks.append("-- Auto-generated schema DDL (synthetic dataset)\n")

    for table_name in tables:
        # Try DuckDB SHOW CREATE TABLE (works in most versions)
        try:
            ddl_df = vn.run_sql(f"SHOW CREATE TABLE {table_name}")
            if not ddl_df.empty:
                ddl_value = str(ddl_df.iloc[0, 0])
                if ddl_value:
                    ddl_chunks.append(ddl_value + ";\n")
                    continue
        except Exception:
            pass

        # Fallback: build a basic CREATE TABLE from information_schema.columns
        cols_df = vn.run_sql(f"""
            SELECT column_name, data_type
            FROM information_schema.columns
            WHERE table_name = '{table_name}'
            ORDER BY ordinal_position
        """)

        cols = [(row["column_name"], row["data_type"]) for _, row in cols_df.iterrows()]
        cols_sql = ",\n  ".join([f"{c} {t}" for c, t in cols])
        ddl_chunks.append(f"CREATE TABLE {table_name} (\n  {cols_sql}\n);\n")
    return "\n".join(ddl_chunks)


def load_information_schema_columns(vn: VannaAgent) -> pd.DataFrame:
    """
    Load schema columns into a DataFrame exactly like the video:
    used to create a training plan.
    """
    return vn.run_sql("SELECT * FROM information_schema.columns")


def train_schema_plan(vn: VannaAgent) -> None:
    """
    Create the schema training plan and train Vanna on it.
    This is the key 'video step'.
    """
    df_info = load_information_schema_columns(vn)
    plan = vn.get_training_plan_generic(df_info)
    logging.info("Schema training plan created.")
    vn.train(plan=plan)
    logging.info("Schema plan training completed.")


def train_from_examples_csv(vn: VannaAgent) -> dict[str, str]:
    """
    Optional: if you have question/sql training examples, train them too.
    Expected columns: question, sql (description optional).
    """
    if not TRAINING_EXAMPLES_CSV.exists():
        logging.info("No training_examples.csv found (skipping Q->SQL training).")
        return {}

    df = pd.read_csv(TRAINING_EXAMPLES_CSV)
    if "question" not in df.columns or "sql" not in df.columns:
        logging.warning("training_examples.csv must contain columns: question, sql")
        return {}

    lookup = {}
    for _, row in df.iterrows():
        question = str(row.get("question", "")).strip()
        sql = str(row.get("sql", "")).strip()
        description = str(row.get("description", "")).strip()

        if question and sql:
            lookup[normalize_question(question)] = sql
            vn.train(question=question, sql=sql, documentation=description if description else None)

    logging.info("Q->SQL example training completed.")
    return lookup


def extract_questions_from_file(path: Path) -> list[str]:
    content = path.read_text(encoding="utf-8")
    questions = []
    for line in content.splitlines():
        match = re.search(r"(?:DE|EN|Q):\s*(.+)", line.strip())
        if match:
            questions.append(match.group(1).strip())
    return questions


def train_from_question_docs(vn: VannaAgent) -> None:
    """
    Train Vanna with question lists as documentation to improve retrieval.
    """
    paths = [DEMO_QUESTIONS_PATH, EXTRA_QUESTIONS_PATH]
    questions = []
    for path in paths:
        if path.exists():
            questions.extend(extract_questions_from_file(path))

    if not questions:
        logging.info("No question documentation found (skipping).")
        return

    unique_questions = []
    seen = set()
    for question in questions:
        normalized = normalize_question(question)
        if normalized and normalized not in seen:
            seen.add(normalized)
            unique_questions.append(question)

    doc = "Demo and tricky questions:\n" + "\n".join(f"- {q}" for q in unique_questions)
    vn.train(documentation=doc)
    logging.info("Question documentation training completed.")


def main():
    # 1) Create agent (mistral + chroma local store)
    vn = VannaAgent(
        config={
            "model": OLLAMA_MODEL,
            "path": CHROMA_PATH,
            "ollama_timeout": 900,
            "initial_prompt": INITIAL_PROMPT,
            "max_tokens": MAX_PROMPT_TOKENS,
            "n_results_sql": 5,
            "n_results_documentation": 3,
            "n_results_ddl": 3,
            "options": {"num_ctx": 4096, "num_predict": 128},
        }
    )

    # 2) Connect to DuckDB (read-only)
    try:
        vn.connect_to_duckdb(path=DUCKDB_PATH)
    except TypeError:
        # Older Vanna versions use `url` instead of `path`.
        vn.connect_to_duckdb(url=DUCKDB_PATH)
    logging.info(f"Connected to DuckDB: {DUCKDB_PATH} (read-only)")

    # 3) Export DDL statements (for thesis + training documentation)
    ddl_text = export_all_table_ddls(vn)
    DDL_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    DDL_OUTPUT_PATH.write_text(ddl_text, encoding="utf-8")
    logging.info(f"Saved DDL to: {DDL_OUTPUT_PATH}")

    # Feed DDL as structured training (improves schema retrieval)
    vn.train(ddl=ddl_text)

    # 4) Schema plan training (video style)
    train_schema_plan(vn)

    # 5) Optional: train from your existing Q->SQL examples
    vn.demo_sql_lookup = train_from_examples_csv(vn)

    # Add demo and tricky questions as documentation for better retrieval
    train_from_question_docs(vn)

    # 5b) Build domain guardrail terms (tables + columns + hints)
    vn.domain_terms = build_domain_terms(vn)
    logging.info("Domain guardrails loaded.")

    # 6) Run Flask app UI (video style)
    logging.info("Starting Vanna Flask app...")
    app = VannaFlaskApp(vn)
    app.run(host="0.0.0.0", port=8084, debug=False)


if __name__ == "__main__":
    main()
