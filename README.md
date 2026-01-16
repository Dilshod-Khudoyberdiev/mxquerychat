# MXQueryChat - Project Overview (Flask Demo)

MXQueryChat is a company-style project demo that turns natural-language questions into SQL,
executes them on a local DuckDB file, and returns results through a Flask UI powered by Vanna.
This README is a short presentation of the workflow and technical design of the running Flask app.

## Project Snapshot
- Goal: ask business questions in plain language and get SQL + results.
- UI: Flask web app via `VannaFlaskApp`.
- Model stack: Vanna + Ollama (`phi3:latest`) with a local Chroma vector store.
- Data: local DuckDB file `mxquerychat.duckdb` (used as a read-only dataset).

## End-to-End Workflow (Flask App)
1. Load DuckDB and export full schema DDL to `docs/schema_ddl.sql`.
2. Train Vanna on the schema plan from `information_schema`.
3. (Optional) Train on Q-to-SQL examples from `training_data/training_examples.csv`.
4. (Optional) Train on demo and tricky question docs from `docs/demo_questions.md` and `docs/tricky_questions.md`.
5. User asks a question in the Flask UI.
6. Vanna builds a prompt, uses retrieval from Chroma, and calls Ollama to generate SQL.
7. SQL runs against DuckDB and results are returned in the UI.

## Technical Details (What Runs Today)
- Entry point: `vanna_flask_demo.py`.
- Agent: `VannaAgent` combines `ChromaDB_VectorStore` + `Ollama`.
- Guardrails:
  - Small-talk and out-of-scope questions return a safe `SELECT` message.
  - Domain-term filtering nudges users toward dataset topics.
  - If the LLM response is not SQL, similar trained SQL is used as fallback.
- Storage:
  - Vector store: `vanna_chroma_store_demo`.
  - DuckDB file: `mxquerychat.duckdb`.

## Run the Flask Demo
```bash
# Ensure Ollama is running and the model is available
ollama pull phi3:latest

# Start the Flask demo
python vanna_flask_demo.py
```

The server starts on `http://localhost:8084`.

## Workflow Diagrams

### 1) Training vs. Question Flow
![vanna1](https://github.com/user-attachments/assets/525060e3-9310-4415-82eb-ab15de0733ab)


This diagram shows the two lanes:
- Training: schema DDL + examples + question docs feed the Chroma vector store.
- Asking: the user question is paired with retrieved context, sent to the LLM, and executed on DuckDB.

### 2) Vanna Feedback Loop
<img width="261" height="193" alt="vanna2" src="https://github.com/user-attachments/assets/f9cf9997-0d43-4857-813f-657d14d30180" />


This loop highlights how questions become SQL, results are validated, and corrections can be added
as new training data to improve future answers.

### 3) Vanna AI Ecosystem View

<img width="224" height="225" alt="vanna3" src="https://github.com/user-attachments/assets/810ed3cb-ea44-4548-a9ea-2e3d7028a728" />

The project stays modular: Vanna sits in the middle and connects a SQL database, a vector store,
an LLM, and a web front end (Flask in this demo).
