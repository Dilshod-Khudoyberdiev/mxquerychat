# MXQueryChat - Technical Documentation

## Current MVP Scope
- UI and app runtime: Streamlit (`app.py`)
- Data source: DuckDB only (`mxquerychat.duckdb`)
- SQL generation: Vanna + Ollama (`vannaagent.py`)
- SQL safety: read-only guard + execution policy limits
- Training data management: CSV-backed editor in Streamlit

This MVP is intentionally DuckDB-only. Multi-database support is future work.

## Main Runtime Flow (New Question)
1. User asks a question in `New Question`.
2. App resolves SQL in this order:
   - session cache
   - exact training example match
   - deterministic template planner
   - LLM fallback with retry prompts
3. User reviews/edits SQL and reads optional explanation.
4. App enforces safety:
   - read-only SQL guard (`sql_guard.py`)
   - complexity limits and hard row cap (`src/db/execution_policy.py`)
   - execution timeout (worker process)
5. Results are shown as table + simple bar chart.
6. User can provide feedback (thumbs up/down), logged as metrics.

## Training Data Flow
1. `Training Data` view loads `training_data/training_examples.csv`.
2. CSV schema is normalized to:
   - `question`
   - `sql`
   - `description`
   - `created_at`
   - `updated_at`
3. Save updates timestamps and preserves existing values for unchanged rows.
4. Selected rows can be deleted with explicit confirmation.
5. `Train Model` retrains Vanna from base docs + examples.

## Key Modules
- `app.py`: Streamlit pages and interaction flow.
- `vannaagent.py`: Vanna/Ollama setup + training example I/O helpers.
- `sql_guard.py`: read-only SQL validation.
- `src/core/query_logic.py`: deterministic planner, retry prompts, local guardrails.
- `src/db/data_source.py`: active DuckDB info + cache refresh/reload.
- `src/db/execution_policy.py`: complexity checks, row limits, timeout execution.
- `src/utils/telemetry.py`: app logs + structured metric events.
- `tools/run_benchmark.py`: benchmark runner for demo/tricky question sets.
- `tools/summarize_metrics.py`: aggregate KPI summary from metrics JSONL.

## Metrics and Logs
- App log: `logs/app.log`
- Metrics log: `logs/metrics.jsonl`
- Event coverage includes generation path/outcome, execution stats, and feedback.
- Standard failure categories used in telemetry:
  - `blocked_read_only`
  - `blocked_complexity`
  - `timeout`
  - `compile_fail`
  - `runtime_fail`
  - `no_match`

## Test Execution
Use the project virtual environment:

```bash
# Windows
.venv\Scripts\python -m pytest

# POSIX
.venv/bin/python -m pytest
```

## Benchmark Harness
Source sets:
- `docs/demo_questions.md`
- `docs/tricky_questions.md`

Command examples:
```bash
# Windows
.venv\Scripts\python tools/run_benchmark.py
.venv\Scripts\python tools/run_benchmark.py --use-llm --max-questions 20

# POSIX
.venv/bin/python tools/run_benchmark.py
```

Outputs:
- `reports/benchmark_*.json` (summary + full records)
- `reports/benchmark_*.csv` (row-level results)

PRD-facing summary fields include compile rate, safe-fail rate, and median latency.

## Metrics Summary Command
```bash
# Windows
.venv\Scripts\python tools/summarize_metrics.py

# POSIX
.venv/bin/python tools/summarize_metrics.py
```
