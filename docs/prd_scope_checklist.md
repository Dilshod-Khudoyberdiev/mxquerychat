# PRD Scope Checklist (MVP)

Status legend:
- `done`: implemented and verified in current codebase
- `remaining`: still needs implementation

## Scope Lock
- `done` DuckDB-only MVP boundary is enforced in app/docs.
- `done` Self-hosted model stack (Vanna + Ollama), no external LLM API usage.
- `done` Out-of-scope items (multi-DB execution, enterprise auth/secrets, async training) are deferred in docs.

## New Question Journey
- `done` Navigation includes `New Question` and `Training Data` only (`app.py`).
- `done` `New Question` has schema visibility in sidebar and source panel.
- `done` Data-source controls include `Reload Dataset` and `Refresh Schema`.
- `done` SQL generation pipeline includes cache/training/template/LLM fallback order.
- `done` SQL review is editable before execution.
- `done` SQL explanation is optional and collapsible.
- `done` Safety gate is two-layer: read-only guard + execution policy.
- `done` Results view shows table + simple bar chart.
- `done` Feedback controls capture thumbs up/down telemetry.
- `done` `New Chat` resets question flow state including results and feedback state.

## Safety and Execution
- `done` Read-only validation blocks write/DDL attempts.
- `done` Complexity limits exist (JOIN/CTE/length).
- `done` Row cap enforcement exists (`LIMIT 1000` wrapper).
- `done` Execution timeout exists (worker process, deterministic timeout message).
- `done` Failure categories are standardized for execution/generation telemetry.

## Training Data Journey
- `done` Training data schema includes `question, sql, description, created_at, updated_at`.
- `done` Save path auto-manages timestamps and preserves `created_at` when updating rows.
- `done` Delete requires explicit confirmation.
- `done` Empty `question/sql` rows are dropped before save.
- `done` Duplicate `(question, sql)` rows raise a warning.
- `done` Retraining uses current in-memory working set.

## Metrics and Evaluation
- `done` Structured app logs and JSONL metrics are written locally.
- `done` User feedback events include `question_hash`, `rating`, `has_result`.
- `done` Benchmark runner uses `docs/demo_questions.md` + `docs/tricky_questions.md`.
- `done` Benchmark outputs JSON + CSV with outcome classes.
- `done` Benchmark summary includes compile rate, safe-fail rate, and median latency.
- `done` Metrics summarizer tool outputs PRD-facing KPI summary.

## Tests and Operability
- `done` Existing unit/integration tests for guardrails/template/retry/safety remain present.
- `done` Added tests for reset helper, failure category mapping, duplicate warning path, and metrics summarizer schema.
- `done` README/docs provide venv-first test commands for Windows and POSIX.
