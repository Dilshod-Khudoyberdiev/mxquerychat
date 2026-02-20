"""
vannaagent.py

mxQueryChat Vanna setup for:
- DuckDB (local file)
- Ollama (self-hosted LLM)
- ChromaDB (local vector store)
- Training examples stored in a local CSV

This file exposes helper functions the Streamlit app can call.
"""

from pathlib import Path
import os
import re
import pandas as pd

try:
    from vanna.chromadb import ChromaDB_VectorStore
    from vanna.ollama import Ollama
except ModuleNotFoundError:
    from vanna.legacy.chromadb import ChromaDB_VectorStore
    from vanna.legacy.ollama import Ollama

DUCKDB_PATH = "mxquerychat.duckdb"
CHROMA_PATH = "vanna_chroma_store"
TRAINING_CSV_PATH = Path("training_data/training_examples.csv")

BASE_TRAINING_DOCUMENTS = [
    "mxquerychat uses DuckDB and must stay read-only. Generate SELECT/WITH queries only.",
    "Primary fact table: ticket_verkaeufe(monat, jahr, tarifverbund_id, ticket_code, umsatz_eur, anzahl, plz).",
    "State join path: ticket_verkaeufe.plz -> postleitzahlen.plz -> postleitzahlen.bundesland_code2 -> regionen_bundesland.bundesland_code2.",
    "Tariff join path: ticket_verkaeufe.tarifverbund_id -> tarifverbuende.tarifverbund_id.",
    "Ticket product join path: ticket_verkaeufe.ticket_code -> ticket_produkte.ticket_code.",
    "For state-level revenue, aggregate SUM(ticket_verkaeufe.umsatz_eur) grouped by regionen_bundesland.bundesland_name.",
    "For tariff-level revenue, aggregate SUM(ticket_verkaeufe.umsatz_eur) grouped by tarifverbuende.name.",
]


def normalize_question(text: str) -> str:
    if not text:
        return ""
    normalized = re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()
    return " ".join(normalized.split())


class MXQueryVanna(ChromaDB_VectorStore, Ollama):
    def __init__(self, config=None):
        # 1) Local LLM via Ollama
        Ollama.__init__(self, config=config)
        # 2) Local vector store via ChromaDB
        ChromaDB_VectorStore.__init__(self, config=config)


def get_vanna() -> MXQueryVanna:
    """
    Create and return a configured Vanna instance.
    """
    vn = MXQueryVanna(
        config={
            "model": os.getenv("OLLAMA_MODEL", "mistral"),
            "path": CHROMA_PATH,
            # Lower latency defaults for local CPU/GPU
            "ollama_timeout": float(os.getenv("OLLAMA_TIMEOUT_SECONDS", "60")),
            "keep_alive": os.getenv("OLLAMA_KEEP_ALIVE", "30m"),
            "options": {
                "num_ctx": int(os.getenv("OLLAMA_NUM_CTX", "2048")),
                "num_predict": int(os.getenv("OLLAMA_NUM_PREDICT", "96")),
                "temperature": float(os.getenv("OLLAMA_TEMPERATURE", "0")),
            },
        }
    )
    try:
        vn.connect_to_duckdb(path=DUCKDB_PATH, read_only=True)
    except TypeError:
        vn.connect_to_duckdb(url=DUCKDB_PATH, read_only=True)
    return vn


def load_training_examples() -> pd.DataFrame:
    """
    Load training examples from CSV (or create an empty file if missing).
    Columns: question, sql, description
    """
    TRAINING_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)

    if not TRAINING_CSV_PATH.exists():
        df = pd.DataFrame(columns=["question", "sql", "description"])
        df.to_csv(TRAINING_CSV_PATH, index=False, encoding="utf-8-sig")
        return df

    return pd.read_csv(TRAINING_CSV_PATH)


def save_training_examples(df: pd.DataFrame) -> None:
    """
    Save the training examples to CSV.
    """
    df = df.copy()
    # Keep only expected columns (avoid accidental extras)
    df = df[["question", "sql", "description"]]
    df.to_csv(TRAINING_CSV_PATH, index=False, encoding="utf-8-sig")


def train_from_examples(vn: MXQueryVanna, examples_df: pd.DataFrame) -> None:
    """
    Train Vanna using all examples in the CSV.
    """
    # Base documentation improves schema-grounded SQL generation.
    for doc in BASE_TRAINING_DOCUMENTS:
        vn.train(documentation=doc)

    for _, row in examples_df.iterrows():
        question = str(row.get("question", "")).strip()
        sql = str(row.get("sql", "")).strip()
        description = str(row.get("description", "")).strip()

        if question and sql:
            # Vanna supports training with question/sql and optional documentation
            vn.train(question=question, sql=sql, documentation=description if description else None)


def get_exact_training_sql(question: str, examples_df: pd.DataFrame) -> str | None:
    """
    Return SQL from CSV if question matches exactly after normalization.
    This avoids LLM latency for known demo questions.
    """
    target = normalize_question(question)
    if not target or examples_df is None or examples_df.empty:
        return None

    for _, row in examples_df.iterrows():
        q = str(row.get("question", "")).strip()
        sql = str(row.get("sql", "")).strip()
        if sql and normalize_question(q) == target:
            return sql

    return None
