# Repository Guidelines

mxQueryChat is a project focused on **learning by building**.
The application allows users to ask natural language questions, converts them to SQL using a self-hosted model (Vanna), executes the SQL in a **strictly read-only** way on **DuckDB**, and visualizes the results in **Streamlit**.

This file defines how humans and agentic coding tools should work in this repository.

## 1. Project Goals & Constraints (Must Follow)

### Primary Goal
Build an MVP Streamlit application that demonstrates:
- Natural language → SQL generation
- SQL explanation
- User review before execution
- Safe, read-only SQL execution
- Result display (table + simple chart)
- Training data management for the model

### Non-Negotiable Constraints
- **Self-hosted only** (no external LLM APIs).
- **DuckDB only** for the database (local file).
- **Read-only SQL execution**:
  - No data modification.
  - No schema modification.
- Code should prioritize **clarity, safety, and learning**, not performance or cleverness.

Agentic tools must **never bypass these constraints**.

---

## Project Structure & Module Organization

Keep the repository simple and predictable.

Recommended structure:

```text
mxquerychat/
  app.py                     # Streamlit entry point
  src/
    __init__.py
    ui/
      new_question_page.py    # "New Question" page UI + logic
      training_data_page.py   # "Training Data" page UI + logic
    db/
      duckdb_connection.py    # DuckDB connection helpers
      sql_guard.py            # Read-only SQL validation (critical)
      query_runner.py         # Executes validated SQL
    llm/
      vanna_client.py         # Vanna model setup + text-to-SQL
      training.py             # Training data → model training
    utils/
      session_state.py        # Streamlit session state helpers
      logging_config.py       # Basic logging
  data/
    duckdb/
      mxquerychat.duckdb      # Local DuckDB database
  training_data/
    examples.csv              # Optional starter training data
  tests/
    test_sql_guard.py         # Tests for SQL safety rules
  README.md                   # Setup & usage instructions
  .env.example                # Example environment variables
```
## Build, Test, and Development Commands

python -m venv .venv
# Windows: .venv\Scripts\activate
source .venv/bin/activate

pip install -r requirements.txt
# Run the application
streamlit run app.py

# Run tests
pytest

## Coding Style & Naming Conventions

No formatting or linting configuration is present. Until standards are defined, follow the defaults of the chosen language and keep naming consistent within a module. When you add tooling (for example, `prettier`, `eslint`, `black`), record the required indentation, line length, and naming rules here.

## Testing Guidelines

No test framework is configured yet. When tests are added, specify the framework (for example, `pytest`, `jest`, `go test`), the expected coverage level (if any), and naming conventions (for example, `*_test.py`, `*.spec.ts`). Include the exact command used to run the full test suite.

## Commit & Pull Request Guidelines

There is no commit history yet, so no established message convention exists. Use clear, imperative messages (for example, "Add initial API scaffold") and include a scope if helpful. For pull requests, include a short description of the change, link related issues, and add screenshots for UI changes. Note any migration or configuration steps required to run the update.

## Security & Configuration Tips

Do not commit secrets or local configuration. If you add runtime configuration, prefer environment variables and document required keys in a `README` or `.env.example`.

## 2. Agent Operating Rules (Codex) — Autonomy After Initial Approval

These rules apply specifically to the **mxQueryChat** project and repository.

### One-time approval, then proceed
- Once the user has granted permission to make changes in this session, **Codex should not ask again** before editing files.
- Codex is explicitly authorized to **create, edit, move, and delete** files in this repo to complete tasks, as long as the constraints in this document are followed.

### When Codex MUST still ask (exceptions)
Codex should request confirmation only if a change would:
- Violate or weaken **read-only SQL execution** or the SQL safety rules in `sql_guard.py`.
- Introduce **external LLM APIs** or otherwise break the **self-hosted only** requirement.
- Switch away from **DuckDB** or change DB/storage approach.
- Modify actual data assets under `data/duckdb/` (changing the DB file contents) rather than code.
- Add/remove significant dependencies, or require non-trivial environment/setup changes beyond documenting them.

Otherwise: **make the change immediately**, then summarize what was done.

### Commenting & readability standard (required)
- At the top of **every source file**, include a detailed header comment that explains:
  - The file’s purpose
  - What it contains (major classes/functions)
  - Key invariants / safety guarantees (especially around read-only SQL)
  - How it is used by other modules (briefly)
- Do **not** write comments using “you/your”.
- Write comments as if the author is the developer/maintainer of mxQueryChat (confident, declarative tone).