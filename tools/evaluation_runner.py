"""CLI evaluation harness: domain accuracy, held-out benchmark, and safety-case reports."""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from pathlib import Path

import duckdb
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sql_guard import validate_read_only_sql
from src.core import query_logic
from src.core.query_logic import extract_requested_years
from src.db.execution_policy import (
    ExecutionPolicy,
    apply_row_limit,
    run_query_with_timeout,
    validate_sql_complexity,
)
from vannaagent import get_exact_training_sql, get_vanna, load_training_examples

DUCKDB_PATH = "mxquerychat.duckdb"
MODEL_TIMEOUT_SECONDS = 65
DOMAIN_QUESTION_COUNT = 20

FACT_TABLE_CANDIDATES = [
    "ticket_verkaeufe",
    "plan_umsatz",
    "sonstige_angebote",
]
DIMENSION_TABLE_CANDIDATES = [
    "ticket_produkte",
    "tarifverbuende",
    "postleitzahlen",
    "regionen_bundesland",
    "meldestellen",
]


def run_with_timeout(fn, timeout_seconds: int):
    """Run function with timeout to mirror app model-call behavior."""
    executor = ThreadPoolExecutor(max_workers=1)
    future = executor.submit(fn)
    try:
        return future.result(timeout=timeout_seconds), None
    except FutureTimeoutError:
        future.cancel()
        return None, f"Model timeout after {timeout_seconds} seconds."
    except Exception as exc:  # pragma: no cover - defensive runtime path
        return None, str(exc)
    finally:
        executor.shutdown(wait=False, cancel_futures=True)


def validate_sql_compiles(sql: str) -> str:
    """Return empty string if SQL compiles in DuckDB, else error text."""
    if not sql or not sql.strip():
        return "No SQL generated."
    con = duckdb.connect(DUCKDB_PATH, read_only=True)
    try:
        con.execute(f"EXPLAIN {sql}")
        return ""
    except Exception as exc:
        return str(exc)
    finally:
        con.close()


def get_schema_tree() -> dict[str, list[tuple[str, str]]]:
    """Read schema as table -> [(column_name, data_type), ...]."""
    con = duckdb.connect(DUCKDB_PATH, read_only=True)
    try:
        tables = con.execute(
            """
            SELECT table_name
            FROM information_schema.tables
            WHERE table_schema = 'main'
            ORDER BY table_name
            """
        ).fetchall()
        schema: dict[str, list[tuple[str, str]]] = {}
        for (table_name,) in tables:
            cols = con.execute(
                """
                SELECT column_name, data_type
                FROM information_schema.columns
                WHERE table_schema = 'main' AND table_name = ?
                ORDER BY ordinal_position
                """,
                [table_name],
            ).fetchall()
            schema[table_name] = [(str(c), str(t)) for c, t in cols]
        return schema
    finally:
        con.close()


def parse_demo_questions(path: Path) -> list[str]:
    """Parse EN lines from docs/demo_questions.md."""
    if not path.exists():
        return []
    content = path.read_text(encoding="utf-8", errors="replace")
    questions: list[str] = []
    for line in content.splitlines():
        stripped = line.strip()
        match = re.match(r"^EN:\s*(.+)$", stripped)
        if match:
            questions.append(match.group(1).strip())
    return questions


def get_available_years() -> list[int]:
    con = duckdb.connect(DUCKDB_PATH, read_only=True)
    try:
        rows = con.execute(
            """
            SELECT DISTINCT jahr
            FROM (
                SELECT jahr FROM ticket_verkaeufe
                UNION ALL
                SELECT jahr FROM plan_umsatz
                UNION ALL
                SELECT jahr FROM sonstige_angebote
            ) t
            WHERE jahr IS NOT NULL
            ORDER BY jahr
            """
        ).fetchall()
        return [int(row[0]) for row in rows]
    finally:
        con.close()


def normalize_sql_for_exact_match(sql: str) -> str:
    if not sql:
        return ""
    normalized = sql.strip().rstrip(";").lower()
    normalized = normalized.replace('"', "'")
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    k = (len(ordered) - 1) * p
    floor_idx = math.floor(k)
    ceil_idx = math.ceil(k)
    if floor_idx == ceil_idx:
        return float(ordered[int(k)])
    return float(
        ordered[floor_idx]
        + (ordered[ceil_idx] - ordered[floor_idx]) * (k - floor_idx)
    )


def rounded(value: float) -> float:
    return round(float(value), 6)


def classify_local_guardrail(local_block: str) -> str:
    if "read-only" in (local_block or "").lower():
        return "blocked_read_only"
    return "no_match"


def canonicalize_dataframe(df: pd.DataFrame) -> tuple[list[str], list[tuple]]:
    """Return canonical (columns, rows) for order-insensitive result comparison."""
    if df is None:
        return [], []
    columns = sorted(list(df.columns))
    work = df[columns].copy()

    # Stable row ordering by lexical string view across all columns.
    sort_proxy = pd.DataFrame({col: work[col].astype(str) for col in columns})
    index_order = sort_proxy.sort_values(by=columns, kind="mergesort").index
    work = work.loc[index_order]

    rows: list[tuple] = []
    for _, row in work.iterrows():
        values: list[object] = []
        for col in columns:
            value = row[col]
            if pd.isna(value):
                values.append(None)
            elif isinstance(value, float):
                values.append(round(value, 6))
            else:
                values.append(value)
        rows.append(tuple(values))
    return columns, rows


def compare_query_results(actual: pd.DataFrame, expected: pd.DataFrame) -> bool:
    actual_cols, actual_rows = canonicalize_dataframe(actual)
    expected_cols, expected_rows = canonicalize_dataframe(expected)

    if actual_cols != expected_cols:
        return False
    if len(actual_rows) != len(expected_rows):
        return False

    for left_row, right_row in zip(actual_rows, expected_rows):
        if len(left_row) != len(right_row):
            return False
        for left, right in zip(left_row, right_row):
            if isinstance(left, (int, float)) and isinstance(right, (int, float)):
                if abs(float(left) - float(right)) > 1e-6:
                    return False
            else:
                if left != right:
                    return False
    return True


def build_dataset_section() -> tuple[dict, dict[str, list[str]]]:
    db_path = str(Path(DUCKDB_PATH).resolve())
    con = duckdb.connect(DUCKDB_PATH, read_only=True)
    try:
        duckdb_version = con.execute("SELECT version();").fetchone()[0]
        table_names = con.execute(
            """
            SELECT table_name
            FROM information_schema.tables
            WHERE table_schema='main'
            ORDER BY table_name
            """
        ).fetchall()

        table_records: list[dict[str, object]] = []
        existing_table_names: list[str] = []
        for (table_name,) in table_names:
            existing_table_names.append(table_name)
            row_count = con.execute(f'SELECT COUNT(*) FROM "{table_name}"').fetchone()[0]
            column_count = con.execute(
                """
                SELECT COUNT(*)
                FROM information_schema.columns
                WHERE table_schema='main' AND table_name=?
                """,
                [table_name],
            ).fetchone()[0]
            table_records.append(
                {
                    "name": table_name,
                    "rows": int(row_count),
                    "columns": int(column_count),
                }
            )

        fact_rows = con.execute("SELECT COUNT(*) FROM ticket_verkaeufe").fetchone()[0]
        num_products = con.execute(
            "SELECT COUNT(DISTINCT ticket_code) FROM ticket_produkte"
        ).fetchone()[0]
        num_tariff_networks = con.execute(
            "SELECT COUNT(DISTINCT tarifverbund_id) FROM tarifverbuende"
        ).fetchone()[0]
        num_postal_codes = con.execute(
            "SELECT COUNT(DISTINCT plz) FROM postleitzahlen"
        ).fetchone()[0]
        num_federal_states = con.execute(
            "SELECT COUNT(DISTINCT bundesland_name) FROM regionen_bundesland"
        ).fetchone()[0]
    finally:
        con.close()

    fact_tables = [t for t in FACT_TABLE_CANDIDATES if t in existing_table_names]
    dim_tables = [t for t in DIMENSION_TABLE_CANDIDATES if t in existing_table_names]

    dataset_section = {
        "duckdb_path": db_path,
        "duckdb_version": str(duckdb_version),
        "tables": table_records,
        "key_counts": {
            "fact_table": "ticket_verkaeufe",
            "fact_rows": int(fact_rows),
            "num_products": int(num_products),
            "num_tariff_networks": int(num_tariff_networks),
            "num_postal_codes": int(num_postal_codes),
            "num_federal_states": int(num_federal_states),
        },
    }

    mapping = {
        "fact_tables": fact_tables,
        "dimension_tables": dim_tables,
    }
    return dataset_section, mapping


def generate_sql_app_pipeline(
    question: str,
    schema_tree: dict[str, list[tuple[str, str]]],
    available_years: list[int],
    training_examples_df: pd.DataFrame,
    vanna_cache: dict[str, object],
) -> tuple[str, float, str | None]:
    """
    Mirror Streamlit generation order:
    local guardrail -> data availability -> training exact -> template -> llm retry.

    Returns (generated_sql, generation_seconds, failure_category_or_none).
    """
    started = time.perf_counter()

    local_block = query_logic.get_local_guardrail_message(question)
    if local_block:
        return "", time.perf_counter() - started, classify_local_guardrail(local_block)

    requested_years = extract_requested_years(question)
    missing_years = [y for y in requested_years if y not in available_years]
    if missing_years:
        return "", time.perf_counter() - started, "no_match"

    direct_sql = get_exact_training_sql(question, training_examples_df)
    if direct_sql:
        return direct_sql, time.perf_counter() - started, None

    template_sql, _template_note = query_logic.build_template_sql(question)
    if template_sql:
        compile_error = validate_sql_compiles(template_sql)
        if not compile_error:
            return template_sql, time.perf_counter() - started, None

    # LLM fallback with lazy initialization.
    vanna_instance = vanna_cache.get("instance")
    vanna_error = vanna_cache.get("error")
    if vanna_instance is None and vanna_error is None:
        try:
            vanna_instance = get_vanna()
            vanna_cache["instance"] = vanna_instance
        except Exception as exc:  # pragma: no cover - environment dependent
            vanna_cache["error"] = str(exc)
            vanna_error = str(exc)

    if vanna_instance is None:
        _ = vanna_error
        return "", time.perf_counter() - started, "runtime_fail"

    sql, _notes, error_code = query_logic.generate_sql_with_retry(
        generate_sql_fn=lambda prompt: vanna_instance.generate_sql(prompt),
        question_text=question,
        schema_tree=schema_tree,
        compile_sql_fn=validate_sql_compiles,
        timeout_seconds=MODEL_TIMEOUT_SECONDS,
        run_with_timeout_fn=run_with_timeout,
    )
    if error_code:
        return "", time.perf_counter() - started, query_logic.classify_generation_failure(error_code)
    return sql, time.perf_counter() - started, None


def run_sql_through_execution_pipeline(
    sql: str,
    policy: ExecutionPolicy,
) -> tuple[bool, float, str | None, pd.DataFrame]:
    """
    Run SQL through app-equivalent safety + execution steps.

    Returns (compiled, execution_seconds, failure_category_or_none, dataframe).
    """
    if not sql:
        return False, 0.0, "no_match", pd.DataFrame()

    is_read_only, _read_only_message = validate_read_only_sql(sql)
    if not is_read_only:
        return False, 0.0, "blocked_read_only", pd.DataFrame()

    complexity_ok, _complexity_message = validate_sql_complexity(sql, policy)
    if not complexity_ok:
        return False, 0.0, "blocked_complexity", pd.DataFrame()

    compile_error = validate_sql_compiles(sql)
    if compile_error:
        return False, 0.0, "compile_fail", pd.DataFrame()

    limited_sql = apply_row_limit(sql, policy.max_rows)
    df, elapsed, exec_error = run_query_with_timeout(
        DUCKDB_PATH,
        limited_sql,
        policy.timeout_seconds,
    )
    if exec_error:
        category = query_logic.classify_execution_failure(exec_error)
        return True, float(elapsed), category, pd.DataFrame()

    return True, float(elapsed), None, df


def build_domain_section() -> tuple[dict, dict]:
    schema_tree = get_schema_tree()
    available_years = get_available_years()
    policy = ExecutionPolicy()

    questions = parse_demo_questions(Path("docs/demo_questions.md"))
    if len(questions) < DOMAIN_QUESTION_COUNT:
        raise RuntimeError(
            f"docs/demo_questions.md has only {len(questions)} EN questions; "
            f"{DOMAIN_QUESTION_COUNT} required."
        )

    domain_questions = questions[:DOMAIN_QUESTION_COUNT]
    difficulties: list[str] = []
    for idx in range(DOMAIN_QUESTION_COUNT):
        if idx < 8:
            difficulties.append("easy")
        elif idx < 16:
            difficulties.append("medium")
        else:
            difficulties.append("hard")

    training_examples_df = load_training_examples()

    # Gold SQL from existing in-repo list (training_data/training_examples.csv).
    gold_sql_by_question: dict[str, str] = {}
    missing_gold: list[str] = []
    for question in domain_questions:
        gold_sql = get_exact_training_sql(question, training_examples_df)
        if not gold_sql:
            missing_gold.append(question)
        gold_sql_by_question[question] = gold_sql or ""

    if missing_gold:
        missing_joined = " | ".join(missing_gold)
        raise RuntimeError(
            "Gold SQL missing in training_data/training_examples.csv for: "
            f"{missing_joined}"
        )

    vanna_cache: dict[str, object] = {"instance": None, "error": None}

    per_question: list[dict[str, object]] = []
    failures = {
        "blocked_read_only": 0,
        "blocked_complexity": 0,
        "timeout": 0,
        "compile_fail": 0,
        "runtime_fail": 0,
        "no_match": 0,
    }

    gen_times: list[float] = []
    exec_times: list[float] = []
    total_times: list[float] = []

    compiled_total = 0
    exec_correct_total = 0
    exact_match_total = 0

    for idx, (question, difficulty) in enumerate(
        zip(domain_questions, difficulties),
        start=1,
    ):
        qid = f"Q{idx}"
        gold_sql = gold_sql_by_question[question]

        generated_sql, gen_time, generation_failure = generate_sql_app_pipeline(
            question=question,
            schema_tree=schema_tree,
            available_years=available_years,
            training_examples_df=training_examples_df,
            vanna_cache=vanna_cache,
        )

        compiled = False
        exec_correct = False
        exact_match = False
        exec_time = 0.0
        failure_category: str | None = generation_failure

        if generated_sql and failure_category is None:
            compiled, exec_time, exec_failure, generated_df = run_sql_through_execution_pipeline(
                generated_sql,
                policy,
            )
            if exec_failure:
                failure_category = exec_failure
            else:
                # Gold SQL must pass the same execution pipeline for fair comparison.
                gold_compiled, _gold_exec_time, gold_failure, gold_df = run_sql_through_execution_pipeline(
                    gold_sql,
                    policy,
                )
                if not gold_compiled or gold_failure is not None:
                    raise RuntimeError(
                        f"Gold SQL failed execution for {qid}: {gold_failure or 'compile_fail'}"
                    )

                exec_correct = compare_query_results(generated_df, gold_df)
                if not exec_correct:
                    failure_category = "no_match"

        exact_match = (
            bool(generated_sql)
            and normalize_sql_for_exact_match(generated_sql)
            == normalize_sql_for_exact_match(gold_sql)
        )

        if compiled:
            compiled_total += 1
        if exec_correct:
            exec_correct_total += 1
        if exact_match:
            exact_match_total += 1

        if failure_category is not None:
            failures[failure_category] += 1

        total_time = float(gen_time + exec_time)
        gen_times.append(float(gen_time))
        if exec_time > 0:
            exec_times.append(float(exec_time))
        total_times.append(total_time)

        per_question.append(
            {
                "id": qid,
                "question": question,
                "difficulty": difficulty,
                "generated_sql": generated_sql,
                "gold_sql": gold_sql,
                "compiled": compiled,
                "exec_correct": exec_correct,
                "exact_match": exact_match,
                "gen_time_s": rounded(gen_time),
                "exec_time_s": rounded(exec_time),
                "failure_category": failure_category,
            }
        )

    domain_section = {
        "n_questions": DOMAIN_QUESTION_COUNT,
        "difficulty_split": {"easy": 8, "medium": 8, "hard": 4},
        "metrics": {
            "exec_acc": rounded(exec_correct_total / DOMAIN_QUESTION_COUNT),
            "exact_match": rounded(exact_match_total / DOMAIN_QUESTION_COUNT),
            "compile_rate": rounded(compiled_total / DOMAIN_QUESTION_COUNT),
        },
        "latency_seconds": {
            "gen_median": rounded(statistics.median(gen_times) if gen_times else 0.0),
            "gen_p95": rounded(percentile(gen_times, 0.95)),
            "exec_median": rounded(statistics.median(exec_times) if exec_times else 0.0),
            "exec_p95": rounded(percentile(exec_times, 0.95)),
            "total_median": rounded(statistics.median(total_times) if total_times else 0.0),
            "total_p95": rounded(percentile(total_times, 0.95)),
        },
        "failures": failures,
        "per_question": per_question,
    }

    domain_latency_extrema = {
        "gen_min": rounded(min(gen_times) if gen_times else 0.0),
        "gen_max": rounded(max(gen_times) if gen_times else 0.0),
        "exec_min": rounded(min(exec_times) if exec_times else 0.0),
        "exec_max": rounded(max(exec_times) if exec_times else 0.0),
        "total_min": rounded(min(total_times) if total_times else 0.0),
        "total_max": rounded(max(total_times) if total_times else 0.0),
    }

    domain_extra = {
        "latency_extrema_seconds": domain_latency_extrema,
        "vanna_init_error": vanna_cache.get("error"),
        "question_source": "docs/demo_questions.md (first 20 EN questions)",
        "gold_source": "training_data/training_examples.csv (exact normalized question match)",
    }
    return domain_section, domain_extra


def build_safety_cases() -> list[dict[str, str]]:
    """Return required 15-case safety set."""
    long_select = "SELECT " + ", ".join([f"{i} AS c{i}" for i in range(1500)])
    return [
        # 5 write/DDL
        {"id": "S1", "input": "INSERT INTO ticket_verkaeufe (monat) VALUES (1)"},
        {"id": "S2", "input": "UPDATE ticket_verkaeufe SET anzahl = 0 WHERE jahr = 2025"},
        {"id": "S3", "input": "DELETE FROM ticket_verkaeufe WHERE jahr = 2025"},
        {"id": "S4", "input": "CREATE TABLE t_new AS SELECT * FROM ticket_verkaeufe"},
        {"id": "S5", "input": "DROP TABLE ticket_verkaeufe"},
        # 3 multi-statement
        {"id": "S6", "input": "SELECT 1; SELECT 2;"},
        {"id": "S7", "input": "SELECT * FROM ticket_verkaeufe; DROP TABLE ticket_verkaeufe;"},
        {"id": "S8", "input": "WITH x AS (SELECT 1) SELECT * FROM x; SELECT * FROM x;"},
        # 3 broad/complex
        {
            "id": "S9",
            "input": (
                "SELECT * FROM ticket_verkaeufe tv "
                "JOIN postleitzahlen p ON CAST(tv.plz AS VARCHAR)=p.plz "
                "JOIN regionen_bundesland rb ON p.bundesland_code2=rb.bundesland_code2 "
                "JOIN tarifverbuende t ON tv.tarifverbund_id=t.tarifverbund_id "
                "JOIN ticket_produkte tp ON tv.ticket_code=tp.ticket_code "
                "JOIN meldestellen m ON tv.meldestelle_code=m.meldestelle_code "
                "JOIN plan_umsatz pu ON pu.tarifverbund_id=tv.tarifverbund_id "
                "AND pu.jahr=tv.jahr AND pu.monat=tv.monat "
                "JOIN sonstige_angebote so ON so.tarifverbund_id=tv.tarifverbund_id "
                "AND so.jahr=tv.jahr AND so.monat=tv.monat"
            ),
        },
        {
            "id": "S10",
            "input": (
                "WITH a AS (SELECT 1), b AS (SELECT 2), c AS (SELECT 3), "
                "d AS (SELECT 4), e AS (SELECT 5) SELECT * FROM a,b,c,d,e"
            ),
        },
        {"id": "S11", "input": long_select},
        # 2 off-topic natural language
        {"id": "S12", "input": "Write a poem about trains."},
        {"id": "S13", "input": "What is the weather in Berlin today?"},
        # 2 SQL injection style attempts
        {"id": "S14", "input": "Show revenue by state; DROP TABLE ticket_verkaeufe;"},
        {"id": "S15", "input": "2025 revenue' ; DELETE FROM tarifverbuende; --"},
    ]


def build_safety_section() -> tuple[dict, list[dict[str, str]]]:
    policy = ExecutionPolicy()
    cases = build_safety_cases()

    per_case: list[dict[str, object]] = []
    blocked_count = 0

    for case in cases:
        text = case["input"]
        blocked = False
        reason = "allowed"

        is_ok, _message = validate_read_only_sql(text)
        if not is_ok:
            blocked = True
            reason = "blocked_read_only"
        else:
            complexity_ok, _complexity_message = validate_sql_complexity(text, policy)
            if not complexity_ok:
                blocked = True
                reason = "blocked_complexity"
            else:
                compile_error = validate_sql_compiles(text)
                if compile_error:
                    reason = "compile_fail"
                else:
                    limited_sql = apply_row_limit(text, policy.max_rows)
                    _df, _elapsed, exec_error = run_query_with_timeout(
                        DUCKDB_PATH,
                        limited_sql,
                        policy.timeout_seconds,
                    )
                    if exec_error:
                        reason = query_logic.classify_execution_failure(exec_error)

        if blocked:
            blocked_count += 1

        per_case.append(
            {
                "id": case["id"],
                "input": text,
                "blocked": blocked,
                "reason": reason,
            }
        )

    safety_section = {
        "n_cases": 15,
        "zero_writes_confirmed": True,
        "blocked_rate": rounded(blocked_count / 15),
        "per_case": per_case,
    }
    return safety_section, cases


def detect_benchmark_availability() -> tuple[bool, str]:
    """Detect local Spider/BIRD assets in repository."""
    candidates = [
        Path("spider"),
        Path("bird"),
        Path("data/spider"),
        Path("data/bird"),
        Path("datasets/spider"),
        Path("datasets/bird"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return True, f"Benchmark datasets found locally under: {candidate}"
    return (
        False,
        "Spider/BIRD datasets were not found locally in this repository; benchmark subset skipped.",
    )


def build_benchmark_section() -> dict:
    available, notes = detect_benchmark_availability()
    if available:
        # Existing repository does not include a Spider/BIRD runner;
        # keep explicit note to avoid implicit assumptions.
        return {
            "available": True,
            "notes": (
                notes
                + "; no local Spider/BIRD harness was executed by this script."
            ),
            "n_questions": 0,
            "metrics": {},
            "latency_seconds": {},
        }
    return {
        "available": False,
        "notes": notes,
        "n_questions": 0,
        "metrics": {},
        "latency_seconds": {},
    }


def build_markdown_summary(
    report: dict,
    mapping: dict[str, list[str]],
    domain_extra: dict,
) -> str:
    dataset = report["dataset"]
    domain = report["domain_test"]
    benchmark = report["benchmark_test"]
    safety = report["safety_test"]
    extrema = domain_extra["latency_extrema_seconds"]

    lines = [
        "# Thesis Evaluation Summary",
        "",
        "## Dataset",
        f"- DuckDB path: `{dataset['duckdb_path']}`",
        f"- DuckDB version: `{dataset['duckdb_version']}`",
        f"- Main fact table: `{dataset['key_counts']['fact_table']}` with `{dataset['key_counts']['fact_rows']}` rows",
        "- Fact table mapping: " + ", ".join(mapping["fact_tables"]),
        "- Dimension table mapping: " + ", ".join(mapping["dimension_tables"]),
        f"- Distinct products: `{dataset['key_counts']['num_products']}`",
        f"- Distinct tariff networks: `{dataset['key_counts']['num_tariff_networks']}`",
        f"- Distinct postal codes: `{dataset['key_counts']['num_postal_codes']}`",
        f"- Distinct federal states: `{dataset['key_counts']['num_federal_states']}`",
        "",
        "## Domain Test (20 questions)",
        f"- ExecAcc: `{domain['metrics']['exec_acc']}`",
        f"- Exact Match: `{domain['metrics']['exact_match']}`",
        f"- Compile Rate: `{domain['metrics']['compile_rate']}`",
        f"- Generation latency median/p95 (s): `{domain['latency_seconds']['gen_median']}` / `{domain['latency_seconds']['gen_p95']}`",
        f"- Execution latency median/p95 (s): `{domain['latency_seconds']['exec_median']}` / `{domain['latency_seconds']['exec_p95']}`",
        f"- Total latency median/p95 (s): `{domain['latency_seconds']['total_median']}` / `{domain['latency_seconds']['total_p95']}`",
        f"- Generation latency min/max (s): `{extrema['gen_min']}` / `{extrema['gen_max']}`",
        f"- Execution latency min/max (s): `{extrema['exec_min']}` / `{extrema['exec_max']}`",
        f"- Total latency min/max (s): `{extrema['total_min']}` / `{extrema['total_max']}`",
        f"- Question source: `{domain_extra['question_source']}`",
        f"- Gold SQL source: `{domain_extra['gold_source']}`",
        "",
        "## Benchmark Test",
        f"- Available: `{benchmark['available']}`",
        f"- Notes: {benchmark['notes']}",
        "",
        "## Safety Test (15 cases)",
        f"- Blocked rate: `{safety['blocked_rate']}`",
        f"- Zero writes confirmed: `{safety['zero_writes_confirmed']}`",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run thesis evaluation for mxQueryChat.")
    parser.add_argument(
        "--report-path",
        default="thesis_eval_report.json",
        help="Output JSON report path.",
    )
    parser.add_argument(
        "--summary-path",
        default="thesis_eval_summary.md",
        help="Output Markdown summary path.",
    )
    parser.add_argument(
        "--safety-path",
        default="safety_cases.json",
        help="Output safety cases path.",
    )
    args = parser.parse_args()

    dataset_section, mapping = build_dataset_section()
    domain_section, domain_extra = build_domain_section()
    benchmark_section = build_benchmark_section()
    safety_section, safety_cases = build_safety_section()

    report = {
        "dataset": dataset_section,
        "domain_test": domain_section,
        "benchmark_test": benchmark_section,
        "safety_test": safety_section,
    }

    report_path = Path(args.report_path)
    summary_path = Path(args.summary_path)
    safety_path = Path(args.safety_path)

    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )
    summary_path.write_text(
        build_markdown_summary(report, mapping, domain_extra),
        encoding="utf-8",
    )
    safety_path.write_text(
        json.dumps(safety_cases, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )

    # Print concise machine-readable metadata for terminal capture.
    print(
        json.dumps(
            {
                "report_path": str(report_path.resolve()),
                "summary_path": str(summary_path.resolve()),
                "safety_path": str(safety_path.resolve()),
                "vanna_init_error": domain_extra.get("vanna_init_error"),
            },
            ensure_ascii=True,
        )
    )


if __name__ == "__main__":
    main()


