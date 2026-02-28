# mxQueryChat

mxQueryChat is a local, self-hosted NL-to-SQL MVP.
Users ask a data question in plain language, review generated SQL, run it in read-only mode on DuckDB, and see table/chart results in Streamlit.

## Core Constraints
- Self-hosted only (no external LLM APIs)
- DuckDB only (`mxquerychat.duckdb`)
- Strict read-only SQL execution
- Multi-database connectors are future work, not part of this MVP.

## Current App Mode
- Primary app: Streamlit (`app.py`)
- LLM stack: Vanna + Ollama + Chroma (`vannaagent.py`)
- SQL safety guard: `sql_guard.py`
- Training data: `training_data/training_examples.csv`

## Project Layout
```text
mxquerychat/
  app.py
  vannaagent.py
  sql_guard.py
  mxquerychat.duckdb
  training_data/
    training_examples.csv
  src/
    core/
      query_logic.py
    db/
      data_source.py
      execution_policy.py
    utils/
      telemetry.py
  requirements.txt
  .env.example
  tests/
    test_sql_guard.py
    test_query_logic_templates.py
    test_query_logic_retry.py
    test_query_logic_guardrails.py
    test_query_logic_execution.py
    test_execution_policy.py
    test_benchmark_harness.py
```

## Run Streamlit App
```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate

python -m pip install --upgrade pip
python -m pip install -r requirements.txt

streamlit run app.py
```

Optional startup flags (Azure-compatible):
```bash
streamlit run app.py --server.address=0.0.0.0 --server.port=8501
```

## Environment Configuration
Copy `.env.example` to `.env` and set values as needed.

Key variables:
- `OLLAMA_MODEL` and related `OLLAMA_*` settings for local model behavior
- `APP_LOG_LEVEL` for log verbosity
- `APP_LOG_PATH` for app logs
- `APP_METRICS_LOG_PATH` for structured metrics events

## Training Workflow
1. Open the `Training Data` page in Streamlit.
2. Edit/save examples in `training_data/training_examples.csv`.
3. Click `Train Model` to retrain Vanna on base docs + examples.

## Testing
Use the project virtual environment:

```bash
# Windows
.venv\Scripts\python -m pytest

# POSIX
.venv/bin/python -m pytest
```

Test coverage includes:
- `sql_guard.py` read-only checks
- deterministic template SQL planner
- LLM retry/fallback flow (mocked)
- local guardrails
- read-only execution wrapper behavior

## Logging and Basic Metrics
The app writes:
- human-readable logs to `logs/app.log`
- structured metric events (JSON lines) to `logs/metrics.jsonl`

Current metrics include:
- question submissions
- SQL generation path and outcome (`cache`, `training`, `template`, `llm`, `blocked`)
- generation duration
- query execution success/failure, row count, and execution time
- user feedback (`up` / `down`) with anonymized question hash

The sidebar also shows simple session counters for quick monitoring.

## Azure Deployment Notes
- Install dependencies using `requirements.txt`.
- Set environment variables in Azure App Service (or equivalent managed runtime).
- Use this startup command:

```bash
streamlit run app.py --server.address=0.0.0.0 --server.port=8501
```

## Benchmark Harness
Run benchmark on `docs/demo_questions.md` and `docs/tricky_questions.md`:

```bash
# Deterministic/template-only
.venv\Scripts\python tools/run_benchmark.py

# Optional LLM fallback enabled
.venv\Scripts\python tools/run_benchmark.py --use-llm
```

Optional flags:
- `--output-dir reports`
- `--max-questions 20`

Outputs:
- JSON summary report (`reports/benchmark_*.json`)
- CSV row-level report (`reports/benchmark_*.csv`)

Per-question outcomes:
- `success`
- `compile_fail`
- `safe_fail`
- `runtime_fail`

## Legacy Flask Demo (Optional)
The repository still contains `vanna_flask_demo.py` as a legacy demo script.
The active MVP workflow is Streamlit-first, and new features should target `app.py`.

## Repository Hygiene
Runtime artifacts are generated locally and should not be committed:
- Python cache folders
- Chroma runtime stores
- notebook checkpoints
- temp files
- logs
- local `.env` files

Use `.gitignore` in this repo to keep those artifacts out of version control.
