# MXQueryChat - Technical Documentation (Easy Language)

This document explains what works now. It uses simple words and short steps.
The focus is the Flask demo app.

## What Works Now
- A Flask web app that answers data questions with SQL.
- Local LLM inference with Ollama (no external API).
- Vanna for text-to-SQL, training, and retrieval.
- Local DuckDB file for data queries (read-only dataset).
- Local Chroma vector store for training data and schema context.
- Training from schema, examples, and question lists.
- Guardrails for small talk and out-of-scope questions.

## How the Flask App Works (Step by Step)
1. The app connects to `mxquerychat.duckdb`.
2. It exports table DDL into `docs/schema_ddl.sql`.
3. Vanna trains on the schema plan from `information_schema`.
4. Vanna also trains on:
   - `training_data/training_examples.csv`
   - `docs/demo_questions.md`
   - `docs/tricky_questions.md`
5. A user asks a question in the Flask UI.
6. Vanna builds a prompt, adds context from Chroma, and calls Ollama.
7. The SQL runs on DuckDB and results show in the UI.

## Main Files (Short Map)
- `vanna_flask_demo.py`: main Flask demo app and training flow.
- `mxquerychat.duckdb`: local DuckDB data file.
- `docs/schema_ddl.sql`: exported schema DDL for training and reference.
- `training_data/training_examples.csv`: Q-to-SQL examples for training.
- `docs/demo_questions.md`: demo questions used as training docs.
- `docs/tricky_questions.md`: edge-case questions used as training docs.
- `vanna_chroma_store_demo/`: vector store used by the Flask demo.
- `docs/data_dictionary.md`: dataset description for human reference.
- `docs/images/`: workflow diagrams used in `README.md`.

## Guardrails (Simple)
- Small talk and off-topic questions return a safe SQL message.
- Domain terms are built from table and column names to keep questions on-topic.
- If the model output is not SQL, similar trained SQL is used as a fallback.

## How to Run (Flask Demo)
```bash
ollama pull mistral
python vanna_flask_demo.py
```

The Flask UI runs at `http://localhost:8084`.

## Data and Utilities
- Mock data CSVs are in `training_data/mock_csv_v3/`.
- Ingest SQL scripts are in `sql/`.
- Data generation scripts are in `tools/`.
