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
            "model": os.getenv("OLLAMA_MODEL", "phi3"),
            "path": CHROMA_PATH,
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
    # Optional: basic dataset description helps the model
    vn.train(documentation="This DuckDB contains synthetic German public transport ticket analytics data.")

    for _, row in examples_df.iterrows():
        question = str(row.get("question", "")).strip()
        sql = str(row.get("sql", "")).strip()
        description = str(row.get("description", "")).strip()

        if question and sql:
            # Vanna supports training with question/sql and optional documentation
            vn.train(question=question, sql=sql, documentation=description if description else None)
