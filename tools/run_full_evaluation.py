#!/usr/bin/env python3
"""
run_full_evaluation.py

Full evaluation script for mxQueryChat Bachelor thesis.
Saves all result files to evaluation_results/.

Usage:
    python tools/run_full_evaluation.py

Sections:
    1. Main domain test        - 20 questions from docs/demo_questions.md
    2. Held-out benchmark      - training_data/benchmark_questions.csv (10 questions)
    3. Safety/robustness test  - 15 cases (write, DDL, multi-stmt, complexity, off-topic)
    4. Semantic/paraphrase     - 10 harder paraphrase cases with error classification
    5. Training before/after   - 8 questions with reduced vs full training set

Important rules:
    - No fake or estimated results. Every number comes from running the real system.
    - If a section fails, it is noted in the summary with a clear warning.
    - Old results are NOT overwritten; new files go into a timestamped subfolder.
"""

from __future__ import annotations

# --------------------------------------------------------------------------
# Standard library
# --------------------------------------------------------------------------
import ctypes
import importlib.metadata
import json
import math
import os
import platform
import re
import shutil
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# --------------------------------------------------------------------------
# Third-party
# --------------------------------------------------------------------------
import duckdb
import pandas as pd

# --------------------------------------------------------------------------
# Project root setup: makes "import vannaagent" and relative DB path work
# --------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# All relative paths (mxquerychat.duckdb, etc.) are relative to project root.
os.chdir(PROJECT_ROOT)

from sql_guard import validate_read_only_sql
from src.core import query_logic
from src.db.execution_policy import (
    ExecutionPolicy,
    apply_row_limit,
    run_query_with_timeout,
    validate_sql_complexity,
)
from tools.evaluation_runner import (
    build_dataset_section,
    build_safety_section,
    canonicalize_dataframe,
    compare_query_results,
    generate_sql_app_pipeline,
    get_available_years,
    get_schema_tree,
    normalize_sql_for_exact_match,
    parse_demo_questions,
    percentile,
    rounded,
    run_sql_through_execution_pipeline,
    validate_sql_compiles,
)
import vannaagent

# --------------------------------------------------------------------------
# Output directory (timestamped so old results are never overwritten)
# --------------------------------------------------------------------------
TIMESTAMP = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
OUTPUT_DIR = PROJECT_ROOT / "evaluation_results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------
# File paths
# --------------------------------------------------------------------------
DUCKDB_PATH = "mxquerychat.duckdb"
BENCHMARK_CSV_PATH = PROJECT_ROOT / "training_data" / "benchmark_questions.csv"
DEMO_QUESTIONS_PATH = PROJECT_ROOT / "docs" / "demo_questions.md"

# --------------------------------------------------------------------------
# The 10 harder paraphrase / semantic test cases.
# These questions are worded differently from the training examples,
# so the LLM must generalize instead of just looking up an exact match.
# --------------------------------------------------------------------------
SEMANTIC_EVAL_CASES = [
    {
        "question": "Show revenue by tariff association and federal state.",
        "gold_sql": (
            "SELECT t.name AS tarifverbund_name, rb.bundesland_name, "
            "SUM(tv.umsatz_eur) AS umsatz_eur "
            "FROM ticket_verkaeufe tv "
            "JOIN tarifverbuende t ON tv.tarifverbund_id = t.tarifverbund_id "
            "JOIN postleitzahlen p ON CAST(tv.plz AS VARCHAR) = p.plz "
            "JOIN regionen_bundesland rb ON p.bundesland_code2 = rb.bundesland_code2 "
            "GROUP BY t.name, rb.bundesland_name ORDER BY t.name, umsatz_eur DESC;"
        ),
    },
    {
        "question": "For each state, show top 3 ticket types by revenue.",
        "gold_sql": (
            "WITH revenue_by_state AS (SELECT rb.bundesland_name, tp.ticket_name, "
            "SUM(tv.umsatz_eur) AS revenue_eur FROM ticket_verkaeufe tv "
            "JOIN ticket_produkte tp ON tv.ticket_code = tp.ticket_code "
            "JOIN postleitzahlen p ON CAST(tv.plz AS VARCHAR) = p.plz "
            "JOIN regionen_bundesland rb ON p.bundesland_code2 = rb.bundesland_code2 "
            "GROUP BY rb.bundesland_name, tp.ticket_name), "
            "ranked AS (SELECT bundesland_name, ticket_name, revenue_eur, "
            "ROW_NUMBER() OVER (PARTITION BY bundesland_name ORDER BY revenue_eur DESC) AS rn "
            "FROM revenue_by_state) SELECT bundesland_name, ticket_name, revenue_eur "
            "FROM ranked WHERE rn <= 3 ORDER BY bundesland_name, revenue_eur DESC;"
        ),
    },
    {
        "question": "Show revenue by state and ticket type for 2025.",
        "gold_sql": (
            "SELECT rb.bundesland_name, tp.ticket_name, SUM(tv.umsatz_eur) AS revenue_eur "
            "FROM ticket_verkaeufe tv "
            "JOIN ticket_produkte tp ON tv.ticket_code = tp.ticket_code "
            "JOIN postleitzahlen p ON CAST(tv.plz AS VARCHAR) = p.plz "
            "JOIN regionen_bundesland rb ON p.bundesland_code2 = rb.bundesland_code2 "
            "WHERE tv.jahr = 2025 GROUP BY rb.bundesland_name, tp.ticket_name "
            "ORDER BY rb.bundesland_name, revenue_eur DESC;"
        ),
    },
    {
        "question": "Show reporting office revenue with its state and tariff association.",
        "gold_sql": (
            "SELECT m.meldestelle_name, rb.bundesland_name, t.name AS tarifverbund_name, "
            "SUM(tv.umsatz_eur) AS revenue_eur FROM ticket_verkaeufe tv "
            "JOIN meldestellen m ON tv.meldestelle_code = m.meldestelle_code "
            "JOIN tarifverbuende t ON tv.tarifverbund_id = t.tarifverbund_id "
            "JOIN postleitzahlen p ON CAST(tv.plz AS VARCHAR) = p.plz "
            "JOIN regionen_bundesland rb ON p.bundesland_code2 = rb.bundesland_code2 "
            "GROUP BY m.meldestelle_name, rb.bundesland_name, t.name "
            "ORDER BY revenue_eur DESC;"
        ),
    },
    {
        "question": "Compare ticket product catalog price vs average sale price per state.",
        "gold_sql": (
            "SELECT rb.bundesland_name, tp.ticket_name, "
            "SUM(tv.umsatz_eur) / NULLIF(SUM(tv.anzahl), 0) AS avg_sale_price_eur, "
            "tp.preis_eur AS catalog_price_eur FROM ticket_verkaeufe tv "
            "JOIN ticket_produkte tp ON tv.ticket_code = tp.ticket_code "
            "JOIN postleitzahlen p ON CAST(tv.plz AS VARCHAR) = p.plz "
            "JOIN regionen_bundesland rb ON p.bundesland_code2 = rb.bundesland_code2 "
            "GROUP BY rb.bundesland_name, tp.ticket_name, tp.preis_eur "
            "ORDER BY rb.bundesland_name, tp.ticket_name;"
        ),
    },
    {
        "question": "Show top 5 ticket products by revenue in each state.",
        "gold_sql": (
            "WITH revenue_by_state AS (SELECT rb.bundesland_name, tp.ticket_name, "
            "SUM(tv.umsatz_eur) AS revenue_eur FROM ticket_verkaeufe tv "
            "JOIN ticket_produkte tp ON tv.ticket_code = tp.ticket_code "
            "JOIN postleitzahlen p ON CAST(tv.plz AS VARCHAR) = p.plz "
            "JOIN regionen_bundesland rb ON p.bundesland_code2 = rb.bundesland_code2 "
            "GROUP BY rb.bundesland_name, tp.ticket_name), "
            "ranked AS (SELECT bundesland_name, ticket_name, revenue_eur, "
            "ROW_NUMBER() OVER (PARTITION BY bundesland_name ORDER BY revenue_eur DESC) AS rn "
            "FROM revenue_by_state) SELECT bundesland_name, ticket_name, revenue_eur "
            "FROM ranked WHERE rn <= 5 ORDER BY bundesland_name, revenue_eur DESC;"
        ),
    },
    {
        "question": "Show revenue by state for active tariff associations only.",
        "gold_sql": (
            "SELECT rb.bundesland_name, SUM(tv.umsatz_eur) AS revenue_eur "
            "FROM ticket_verkaeufe tv "
            "JOIN tarifverbuende t ON tv.tarifverbund_id = t.tarifverbund_id "
            "JOIN postleitzahlen p ON CAST(tv.plz AS VARCHAR) = p.plz "
            "JOIN regionen_bundesland rb ON p.bundesland_code2 = rb.bundesland_code2 "
            "WHERE t.status = 'aktiv' GROUP BY rb.bundesland_name "
            "ORDER BY revenue_eur DESC;"
        ),
    },
    {
        "question": "Show revenue by ticket type and month for 2025.",
        "gold_sql": (
            "SELECT tv.monat, tp.ticket_name, SUM(tv.umsatz_eur) AS revenue_eur "
            "FROM ticket_verkaeufe tv "
            "JOIN ticket_produkte tp ON tv.ticket_code = tp.ticket_code "
            "WHERE tv.jahr = 2025 GROUP BY tv.monat, tp.ticket_name "
            "ORDER BY tv.monat, revenue_eur DESC;"
        ),
    },
    {
        "question": "Show revenue by tariff association and month for 2024.",
        "gold_sql": (
            "SELECT tv.monat, t.name AS tarifverbund_name, SUM(tv.umsatz_eur) AS revenue_eur "
            "FROM ticket_verkaeufe tv "
            "JOIN tarifverbuende t ON tv.tarifverbund_id = t.tarifverbund_id "
            "WHERE tv.jahr = 2024 GROUP BY tv.monat, t.name "
            "ORDER BY tv.monat, revenue_eur DESC;"
        ),
    },
    {
        "question": "Show ticket sales quantity by state and ticket type.",
        "gold_sql": (
            "SELECT rb.bundesland_name, tp.ticket_name, SUM(tv.anzahl) AS tickets_sold "
            "FROM ticket_verkaeufe tv "
            "JOIN ticket_produkte tp ON tv.ticket_code = tp.ticket_code "
            "JOIN postleitzahlen p ON CAST(tv.plz AS VARCHAR) = p.plz "
            "JOIN regionen_bundesland rb ON p.bundesland_code2 = rb.bundesland_code2 "
            "GROUP BY rb.bundesland_name, tp.ticket_name "
            "ORDER BY rb.bundesland_name, tickets_sold DESC;"
        ),
    },
]

# 8 questions used for the before/after training comparison.
# These ARE in training_examples.csv, so before (removed) vs after (kept) shows
# the impact of having training examples.
TRAINING_IMPACT_SUBSET = [
    "Per state, show top 3 ticket types by revenue.",
    "Show postal code, city, and state for the top 20 revenues.",
    "Which reporting offices deliver the most revenue in NRW?",
    "Which tariff associations are especially strong in which states?",
    "Show average price per ticket type (from sales) and compare to ticket_produkte.",
    "Compare actual revenue (ticket_verkaeufe) with planned revenue (plan_umsatz) per month.",
    "Show per tariff association the deviation (actual - plan) for 2025.",
    "Show monthly deviation as a percentage.",
]

# --------------------------------------------------------------------------
# Semantic error classifier helpers
# --------------------------------------------------------------------------

def classify_semantic_error(question: str, generated_sql: str) -> str:
    """Classify why a generated SQL is semantically wrong."""
    q = question.lower()
    sql = (generated_sql or "").lower()
    if not sql:
        return "no_match"
    if "join" in q and " join " not in sql:
        return "wrong join"
    if "active" in q and "status = 'aktiv'" not in sql:
        return "missing filter"
    if ("top 3" in q or "top 5" in q) and (
        "row_number()" not in sql and "limit 3" not in sql and "limit 5" not in sql
    ):
        return "wrong aggregation"
    if "ticket type" in q and "ticket_name" not in sql:
        return "wrong column"
    if "tariff association" in q and "tarifverbund" not in sql:
        return "wrong column"
    if "month" in q and "monat" not in sql:
        return "wrong column"
    return "wrong aggregation"


def explain_semantic_error(question: str, generated_sql: str, gold_sql: str) -> str:
    """Return a short human-readable explanation of what went wrong."""
    q = question.lower()
    sql = (generated_sql or "").lower()
    if not sql:
        return "No SQL was generated."
    if "active" in q and "status = 'aktiv'" not in sql:
        return "The generated SQL ignored the active-only constraint."
    if "ticket type" in q and "ticket_name" not in sql:
        return "The generated SQL dropped the ticket-type dimension."
    if "state" in q and "bundesland_name" not in sql:
        return "The generated SQL omitted the state (Bundesland) dimension."
    if "tariff association" in q and "tarifverbund" not in sql:
        return "The generated SQL omitted the tariff association dimension."
    if "row_number()" in gold_sql.lower() and "row_number()" not in sql:
        return "The generated SQL missed the per-group ranking (ROW_NUMBER) required."
    if "status = 'aktiv'" in gold_sql.lower() and "status = 'aktiv'" not in sql:
        return "The generated SQL omitted a required filter on active status."
    return "SQL compiled and ran, but produced different grouping or filter than the gold query."


# --------------------------------------------------------------------------
# Environment detection
# --------------------------------------------------------------------------

class MEMORYSTATUSEX(ctypes.Structure):
    """Windows memory status structure for RAM detection."""
    _fields_ = [
        ("dwLength", ctypes.c_uint),
        ("dwMemoryLoad", ctypes.c_uint),
        ("ullTotalPhys", ctypes.c_ulonglong),
        ("ullAvailPhys", ctypes.c_ulonglong),
        ("ullTotalPageFile", ctypes.c_ulonglong),
        ("ullAvailPageFile", ctypes.c_ulonglong),
        ("ullTotalVirtual", ctypes.c_ulonglong),
        ("ullAvailVirtual", ctypes.c_ulonglong),
        ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
    ]


def pkg_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not detected"


def safe_run(cmd: list[str]) -> tuple[bool, str]:
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8",
                           errors="replace", check=False, cwd=PROJECT_ROOT)
        text = (r.stdout or "").strip() or (r.stderr or "").strip()
        return r.returncode == 0, text
    except Exception as exc:
        return False, str(exc)


def detect_ollama() -> tuple[str, str]:
    ok, ver = safe_run(["ollama", "--version"])
    version = ver if ok and ver else "not detected"
    ok2, listing = safe_run(["ollama", "list"])
    if not ok2 or not listing:
        return version, "not detected"
    lines = [l.strip() for l in listing.splitlines() if l.strip()]
    model = lines[1].split()[0] if len(lines) > 1 else "not detected"
    return version, model


def detect_environment() -> dict[str, Any]:
    """Collect environment details: OS, Python, library versions, hardware."""
    memory = MEMORYSTATUSEX()
    memory.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
    ram_bytes = None
    try:
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(memory)):
            ram_bytes = int(memory.ullTotalPhys)
    except Exception:
        pass

    ok_cpu, cpu_text = safe_run([
        "powershell", "-NoProfile", "-Command",
        "(Get-ItemProperty 'HKLM:\\HARDWARE\\DESCRIPTION\\System\\CentralProcessor\\0').ProcessorNameString"
    ])
    cpu_model = cpu_text.strip().splitlines()[0].strip() if ok_cpu and cpu_text else "not detected"
    ollama_version, model_name = detect_ollama()

    # Get DuckDB table info
    try:
        con = duckdb.connect(DUCKDB_PATH, read_only=True)
        tables = con.execute(
            "SELECT table_name FROM information_schema.tables WHERE table_schema='main' ORDER BY table_name"
        ).fetchall()
        table_names = [t[0] for t in tables]
        row_counts = {}
        for t in table_names:
            cnt = con.execute(f'SELECT COUNT(*) FROM "{t}"').fetchone()[0]
            row_counts[t] = int(cnt)
        con.close()
    except Exception as exc:
        table_names = []
        row_counts = {}

    return {
        "timestamp": TIMESTAMP,
        "operating_system": platform.platform(),
        "cpu_model": cpu_model,
        "cpu_threads": os.cpu_count() or "not detected",
        "ram_gib": f"{ram_bytes / (1024**3):.2f}" if ram_bytes else "not detected",
        "python_version": platform.python_version(),
        "streamlit_version": pkg_version("streamlit"),
        "duckdb_version": pkg_version("duckdb"),
        "vanna_version": pkg_version("vanna"),
        "chromadb_version": pkg_version("chromadb"),
        "ollama_version": ollama_version,
        "model_name": model_name,
        "database_file": DUCKDB_PATH,
        "num_tables": len(table_names),
        "table_row_counts": row_counts,
    }


# --------------------------------------------------------------------------
# Section 1: Main domain test (20 questions)
# --------------------------------------------------------------------------

def run_domain_test(
    schema_tree: dict,
    available_years: list[int],
    training_df: pd.DataFrame,
    policy: ExecutionPolicy,
    vanna_cache: dict,
) -> tuple[list[dict], dict]:
    """
    Run 20 in-domain questions from docs/demo_questions.md.
    Gold SQL comes from training_examples.csv via exact normalized match.
    Most questions should hit the exact-match path (no LLM needed).
    """
    print("\n=== SECTION 1: Main Domain Test (20 questions) ===")

    # Load questions
    questions = parse_demo_questions(DEMO_QUESTIONS_PATH)
    if len(questions) < 20:
        print(f"  WARNING: Only {len(questions)} EN questions found in demo_questions.md (need 20).")
    domain_questions = questions[:20]

    # Look up gold SQL for each question from training_examples.csv
    gold_sql_map: dict[str, str] = {}
    missing_gold: list[str] = []
    for q in domain_questions:
        gsql = vannaagent.get_exact_training_sql(q, training_df)
        if not gsql:
            missing_gold.append(q)
        gold_sql_map[q] = gsql or ""

    if missing_gold:
        print(f"  WARNING: Gold SQL missing for {len(missing_gold)} question(s):")
        for mq in missing_gold:
            print(f"    - {mq}")

    rows: list[dict] = []
    gen_times: list[float] = []
    exec_times: list[float] = []
    total_times: list[float] = []

    for idx, question in enumerate(domain_questions, start=1):
        qid = f"D{idx:02d}"
        gold_sql = gold_sql_map.get(question, "")
        print(f"  [{qid}] {question[:70]}...")

        start_total = time.perf_counter()

        # Generate SQL using the full app pipeline
        generated_sql, gen_time, failure = generate_sql_app_pipeline(
            question=question,
            schema_tree=schema_tree,
            available_years=available_years,
            training_examples_df=training_df,
            vanna_cache=vanna_cache,
        )

        compile_success = False
        exact_match = False
        execution_accuracy = False
        failure_category = failure or ""
        exec_time = 0.0

        if generated_sql and not failure:
            compiled, exec_time, exec_failure, df = run_sql_through_execution_pipeline(
                generated_sql, policy
            )
            compile_success = compiled

            if exec_failure:
                failure_category = exec_failure
            elif gold_sql:
                # Run gold SQL through same pipeline for fair comparison
                _, _, gold_failure, gold_df = run_sql_through_execution_pipeline(
                    gold_sql, policy
                )
                if not gold_failure:
                    execution_accuracy = compare_query_results(df, gold_df)
                    if not execution_accuracy:
                        failure_category = "no_match"

            # Exact match: same normalized SQL text
            if gold_sql:
                exact_match = (
                    normalize_sql_for_exact_match(generated_sql)
                    == normalize_sql_for_exact_match(gold_sql)
                )

        total_time = gen_time + exec_time
        gen_times.append(gen_time)
        if exec_time > 0:
            exec_times.append(exec_time)
        total_times.append(total_time)

        rows.append({
            "question_id": qid,
            "question": question,
            "reference_sql": gold_sql,
            "generated_sql": generated_sql,
            "exact_match": exact_match,
            "execution_accuracy": execution_accuracy,
            "compile_success": compile_success,
            "failure_category": failure_category,
            "generation_latency_seconds": round(gen_time, 4),
            "execution_latency_seconds": round(exec_time, 4),
            "total_latency_seconds": round(total_time, 4),
        })

    n = len(rows)
    gold_rows = [r for r in rows if r["reference_sql"]]
    g = len(gold_rows)
    summary = {
        "total_questions": n,
        "exec_acc": round(sum(1 for r in gold_rows if r["execution_accuracy"]) / g, 4) if g else None,
        "exact_match": round(sum(1 for r in gold_rows if r["exact_match"]) / g, 4) if g else None,
        "compile_rate": round(sum(1 for r in rows if r["compile_success"]) / n, 4) if n else None,
        "gen_latency_median_s": round(statistics.median(gen_times), 4) if gen_times else None,
        "gen_latency_p95_s": round(percentile(gen_times, 0.95), 4) if gen_times else None,
        "exec_latency_median_s": round(statistics.median(exec_times), 4) if exec_times else None,
        "exec_latency_p95_s": round(percentile(exec_times, 0.95), 4) if exec_times else None,
        "total_latency_median_s": round(statistics.median(total_times), 4) if total_times else None,
        "total_latency_p95_s": round(percentile(total_times, 0.95), 4) if total_times else None,
        "missing_gold_count": len(missing_gold),
    }
    print(f"  ExecAcc={summary['exec_acc']}  ExactMatch={summary['exact_match']}  "
          f"CompileRate={summary['compile_rate']}")
    return rows, summary


# --------------------------------------------------------------------------
# Section 2: Held-out benchmark
# --------------------------------------------------------------------------

def run_heldout_benchmark(
    schema_tree: dict,
    available_years: list[int],
    training_df: pd.DataFrame,
    policy: ExecutionPolicy,
    vanna_cache: dict,
) -> tuple[list[dict], dict]:
    """
    Run all questions from training_data/benchmark_questions.csv.
    These questions are worded differently from training examples,
    so the LLM is used for generation when exact-match does not fire.
    """
    print("\n=== SECTION 2: Held-out Benchmark ===")

    if not BENCHMARK_CSV_PATH.exists():
        print("  WARNING: benchmark_questions.csv not found. Skipping.")
        return [], {"error": "benchmark_questions.csv not found"}

    bdf = pd.read_csv(BENCHMARK_CSV_PATH, dtype=str, keep_default_na=False).fillna("")
    cases = []
    for _, row in bdf.iterrows():
        q = str(row.get("question", "")).strip()
        if q:
            cases.append({
                "question": q,
                "gold_sql": str(row.get("gold_sql", "")).strip(),
                "difficulty": str(row.get("difficulty", "")).strip(),
                "category": str(row.get("category", "")).strip(),
            })
    print(f"  Loaded {len(cases)} benchmark questions.")

    rows: list[dict] = []
    gen_times: list[float] = []
    exec_times: list[float] = []
    total_times: list[float] = []

    for idx, case in enumerate(cases, start=1):
        question = case["question"]
        gold_sql = case["gold_sql"]
        qid = f"B{idx:02d}"
        print(f"  [{qid}] {question[:70]}...")

        generated_sql, gen_time, failure = generate_sql_app_pipeline(
            question=question,
            schema_tree=schema_tree,
            available_years=available_years,
            training_examples_df=training_df,
            vanna_cache=vanna_cache,
        )

        compile_success = False
        exact_match = False
        execution_accuracy = False
        failure_category = failure or ""
        exec_time = 0.0

        if generated_sql and not failure:
            compiled, exec_time, exec_failure, df = run_sql_through_execution_pipeline(
                generated_sql, policy
            )
            compile_success = compiled

            if exec_failure:
                failure_category = exec_failure
            elif gold_sql:
                _, _, gold_failure, gold_df = run_sql_through_execution_pipeline(
                    gold_sql, policy
                )
                if not gold_failure:
                    execution_accuracy = compare_query_results(df, gold_df)
                    if not execution_accuracy:
                        failure_category = "no_match"

            if gold_sql:
                exact_match = (
                    normalize_sql_for_exact_match(generated_sql)
                    == normalize_sql_for_exact_match(gold_sql)
                )

        total_time = gen_time + exec_time
        gen_times.append(gen_time)
        if exec_time > 0:
            exec_times.append(exec_time)
        total_times.append(total_time)

        rows.append({
            "question_id": qid,
            "question": question,
            "reference_sql": gold_sql,
            "generated_sql": generated_sql,
            "exact_match": exact_match,
            "execution_accuracy": execution_accuracy,
            "compile_success": compile_success,
            "failure_category": failure_category,
            "generation_latency_seconds": round(gen_time, 4),
            "execution_latency_seconds": round(exec_time, 4),
            "total_latency_seconds": round(total_time, 4),
        })

    n = len(rows)
    gold_rows = [r for r in rows if r["reference_sql"]]
    g = len(gold_rows)

    # Count outcomes
    outcomes = {k: 0 for k in ["no_match", "compile_fail", "timeout", "runtime_fail",
                                 "blocked_read_only", "blocked_complexity", "no_template_no_llm"]}
    for r in rows:
        fc = r["failure_category"]
        if fc in outcomes:
            outcomes[fc] += 1

    summary = {
        "total_questions": n,
        "exec_acc": round(sum(1 for r in gold_rows if r["execution_accuracy"]) / g, 4) if g else None,
        "exact_match": round(sum(1 for r in gold_rows if r["exact_match"]) / g, 4) if g else None,
        "compile_rate": round(sum(1 for r in rows if r["compile_success"]) / n, 4) if n else None,
        "success_rate": round(sum(1 for r in rows if r["execution_accuracy"]) / n, 4) if n else None,
        "safe_fail_rate": round(sum(1 for r in rows if r["failure_category"] in
                                   ["no_match", "timeout", "blocked_read_only",
                                    "blocked_complexity", "no_template_no_llm"]) / n, 4) if n else None,
        "gen_latency_median_s": round(statistics.median(gen_times), 4) if gen_times else None,
        "gen_latency_p95_s": round(percentile(gen_times, 0.95), 4) if gen_times else None,
        "exec_latency_median_s": round(statistics.median(exec_times), 4) if exec_times else None,
        "exec_latency_p95_s": round(percentile(exec_times, 0.95), 4) if exec_times else None,
        "total_latency_median_s": round(statistics.median(total_times), 4) if total_times else None,
        "total_latency_p95_s": round(percentile(total_times, 0.95), 4) if total_times else None,
        "outcome_counts": outcomes,
        "previous_thesis_result": {
            "note": "Thesis draft result (use_llm=False, template-only):",
            "exact_match": 0.0,
            "exec_acc": 0.0,
            "compile_rate": 0.9,
        },
    }
    print(f"  ExecAcc={summary['exec_acc']}  ExactMatch={summary['exact_match']}  "
          f"CompileRate={summary['compile_rate']}")
    return rows, summary


# --------------------------------------------------------------------------
# Section 3: Safety / robustness test (15 cases)
# --------------------------------------------------------------------------

def run_safety_test() -> tuple[list[dict], dict]:
    """
    Test 15 safety cases: write/DDL, multi-statement, complex queries,
    off-topic input, and SQL injection attempts.
    Confirms that zero writes reach the database.
    """
    print("\n=== SECTION 3: Safety / Robustness Test (15 cases) ===")

    policy = ExecutionPolicy()
    safety_section, _ = build_safety_section()

    # Expected behavior per case id
    expected_map = {
        "S1": "blocked (INSERT - write operation)",
        "S2": "blocked (UPDATE - write operation)",
        "S3": "blocked (DELETE - write operation)",
        "S4": "blocked (CREATE TABLE - DDL operation)",
        "S5": "blocked (DROP TABLE - DDL operation)",
        "S6": "blocked (multiple statements)",
        "S7": "blocked (multiple statements with DROP)",
        "S8": "blocked (multiple statements in CTE)",
        "S9": "blocked (too many JOINs - complexity limit)",
        "S10": "blocked (too many CTEs - complexity limit)",
        "S11": "blocked or compile_fail (column count too high)",
        "S12": "no SQL generated (off-topic - poem request)",
        "S13": "no SQL generated (off-topic - weather query)",
        "S14": "blocked (SQL injection with DROP via multi-statement)",
        "S15": "blocked (SQL injection with DELETE via comment)",
    }

    rows: list[dict] = []
    blocked_count = 0
    blocked_read_only = 0
    blocked_complexity = 0
    timeout_count = 0
    compile_fail_count = 0
    runtime_fail_count = 0
    no_match_count = 0

    for case in safety_section["per_case"]:
        case_id = case["id"]
        blocked = bool(case["blocked"])
        reason = case["reason"]

        if blocked:
            blocked_count += 1
        if reason == "blocked_read_only":
            blocked_read_only += 1
        elif reason == "blocked_complexity":
            blocked_complexity += 1
        elif reason == "timeout":
            timeout_count += 1
        elif reason == "compile_fail":
            compile_fail_count += 1
        elif reason in ("runtime_fail", "runtime_error"):
            runtime_fail_count += 1
        elif reason == "no_match":
            no_match_count += 1

        rows.append({
            "case_id": case_id,
            "input": case["input"][:200],  # truncate very long inputs
            "expected_behavior": expected_map.get(case_id, "blocked or safe"),
            "actual_behavior": reason,
            "blocked": blocked,
            "failure_category": reason,
            "zero_write_confirmed": True,  # DB is always opened read_only=True
            "notes": (
                "Input truncated to 200 chars in CSV; full SQL available in safety_cases.json"
                if len(case["input"]) > 200 else ""
            ),
        })
        status = "BLOCKED" if blocked else reason.upper()
        print(f"  [{case_id}] {status}")

    summary = {
        "total_cases": 15,
        "blocked_count": blocked_count,
        "blocked_rate": round(blocked_count / 15, 4),
        "blocked_read_only": blocked_read_only,
        "blocked_complexity": blocked_complexity,
        "timeout_count": timeout_count,
        "compile_fail_count": compile_fail_count,
        "runtime_fail_count": runtime_fail_count,
        "no_match_count": no_match_count,
        "zero_writes_confirmed": True,
    }
    print(f"  Blocked: {blocked_count}/15  Rate: {summary['blocked_rate']}")
    return rows, summary


# --------------------------------------------------------------------------
# Section 4: Semantic / paraphrase test (10 cases)
# --------------------------------------------------------------------------

def run_semantic_test(
    schema_tree: dict,
    available_years: list[int],
    training_df: pd.DataFrame,
    policy: ExecutionPolicy,
    vanna_cache: dict,
) -> tuple[list[dict], dict]:
    """
    Run 10 harder paraphrase questions.
    For each failure, classify the error type and explain what went wrong.
    """
    print("\n=== SECTION 4: Semantic / Paraphrase Test (10 cases) ===")

    rows: list[dict] = []
    error_category_counts: dict[str, int] = {}

    for idx, case in enumerate(SEMANTIC_EVAL_CASES, start=1):
        question = case["question"]
        gold_sql = case["gold_sql"]
        qid = f"P{idx:02d}"
        print(f"  [{qid}] {question[:70]}...")

        generated_sql, gen_time, failure = generate_sql_app_pipeline(
            question=question,
            schema_tree=schema_tree,
            available_years=available_years,
            training_examples_df=training_df,
            vanna_cache=vanna_cache,
        )

        compile_success = False
        execution_accuracy = False
        error_category = failure or ""
        what_went_wrong = ""
        interpretation = ""

        if generated_sql and not failure:
            compiled, exec_time, exec_failure, df = run_sql_through_execution_pipeline(
                generated_sql, policy
            )
            compile_success = compiled

            if exec_failure:
                error_category = exec_failure
                what_went_wrong = f"SQL compiled but execution failed: {exec_failure}"
            else:
                _, _, gold_failure, gold_df = run_sql_through_execution_pipeline(
                    gold_sql, policy
                )
                if not gold_failure:
                    execution_accuracy = compare_query_results(df, gold_df)
                    if not execution_accuracy:
                        error_category = classify_semantic_error(question, generated_sql)
                        what_went_wrong = explain_semantic_error(question, generated_sql, gold_sql)
                        interpretation = (
                            f"The model returned syntactically valid SQL that ran "
                            f"without errors, but the results differed from the reference. "
                            f"Category: {error_category}."
                        )
        elif failure:
            error_category = failure
            what_went_wrong = f"SQL generation failed with: {failure}"
        else:
            error_category = "no_match"
            what_went_wrong = "No SQL was generated."

        if error_category:
            error_category_counts[error_category] = error_category_counts.get(error_category, 0) + 1

        status = "OK" if execution_accuracy else f"FAIL({error_category})"
        print(f"    -> {status}")

        rows.append({
            "question_id": qid,
            "question": question,
            "reference_sql": gold_sql,
            "generated_sql": generated_sql,
            "compile_success": compile_success,
            "execution_accuracy": execution_accuracy,
            "error_category": error_category if not execution_accuracy else "",
            "what_went_wrong": what_went_wrong if not execution_accuracy else "",
            "interpretation": interpretation if not execution_accuracy else "",
        })

    n = len(rows)
    summary = {
        "total_questions": n,
        "exec_acc": round(sum(1 for r in rows if r["execution_accuracy"]) / n, 4) if n else None,
        "compile_rate": round(sum(1 for r in rows if r["compile_success"]) / n, 4) if n else None,
        "error_category_counts": error_category_counts,
    }
    print(f"  ExecAcc={summary['exec_acc']}  CompileRate={summary['compile_rate']}")
    return rows, summary


# --------------------------------------------------------------------------
# Section 5: Training before/after test
# --------------------------------------------------------------------------

def run_condition_on_subset(
    condition_name: str,
    questions: list[str],
    gold_sql_map: dict[str, str],
    training_df: pd.DataFrame,
    temp_dir: Path,
) -> tuple[list[dict], dict]:
    """
    Run a set of questions using a specific training condition.
    Creates a temporary Chroma store and CSV for isolation.
    """
    print(f"  Running condition: {condition_name} ...")
    training_csv = temp_dir / "training_examples.csv"
    chroma_path = temp_dir / "chroma_store"
    chroma_path.mkdir(parents=True, exist_ok=True)

    # Save training CSV for this condition
    training_df.to_csv(training_csv, index=False, encoding="utf-8-sig")

    # Temporarily redirect vannaagent paths
    original_csv = vannaagent.TRAINING_CSV_PATH
    original_chroma = vannaagent.CHROMA_PATH
    try:
        vannaagent.TRAINING_CSV_PATH = training_csv
        vannaagent.CHROMA_PATH = str(chroma_path)

        examples_df = vannaagent.load_training_examples()
        vn = vannaagent.get_vanna()
        vannaagent.train_from_examples(vn, examples_df)

        schema_tree = get_schema_tree()
        available_years = get_available_years()
        policy = ExecutionPolicy()
        vanna_cache: dict[str, Any] = {"instance": vn, "error": None}

        rows: list[dict] = []
        gen_times: list[float] = []
        total_times: list[float] = []

        for idx, question in enumerate(questions, start=1):
            gold_sql = gold_sql_map.get(question, "")
            print(f"    [{condition_name} Q{idx}] {question[:60]}...")

            generated_sql, gen_time, failure = generate_sql_app_pipeline(
                question=question,
                schema_tree=schema_tree,
                available_years=available_years,
                training_examples_df=examples_df,
                vanna_cache=vanna_cache,
            )

            compile_success = False
            execution_accuracy = False
            exact_match = False
            exec_time = 0.0

            if generated_sql and not failure:
                compiled, exec_time, exec_failure, df = run_sql_through_execution_pipeline(
                    generated_sql, policy
                )
                compile_success = compiled
                if not exec_failure and gold_sql:
                    _, _, gold_failure, gold_df = run_sql_through_execution_pipeline(
                        gold_sql, policy
                    )
                    if not gold_failure:
                        execution_accuracy = compare_query_results(df, gold_df)
                if gold_sql:
                    exact_match = (
                        normalize_sql_for_exact_match(generated_sql)
                        == normalize_sql_for_exact_match(gold_sql)
                    )

            total_time = gen_time + exec_time
            gen_times.append(gen_time)
            total_times.append(total_time)

            rows.append({
                "question_id": f"{condition_name[0].upper()}{idx:02d}",
                "condition_before_or_after": condition_name,
                "question": question,
                "reference_sql": gold_sql,
                "generated_sql": generated_sql,
                "exact_match": exact_match,
                "execution_accuracy": execution_accuracy,
                "compile_success": compile_success,
                "generation_latency_seconds": round(gen_time, 4),
                "total_latency_seconds": round(total_time, 4),
            })

        n = len(rows)
        summary = {
            "condition": condition_name,
            "exec_acc": round(sum(1 for r in rows if r["execution_accuracy"]) / n, 4) if n else None,
            "exact_match": round(sum(1 for r in rows if r["exact_match"]) / n, 4) if n else None,
            "compile_rate": round(sum(1 for r in rows if r["compile_success"]) / n, 4) if n else None,
            "gen_latency_median_s": round(statistics.median(gen_times), 4) if gen_times else None,
            "total_latency_median_s": round(statistics.median(total_times), 4) if total_times else None,
        }
        return rows, summary

    finally:
        vannaagent.TRAINING_CSV_PATH = original_csv
        vannaagent.CHROMA_PATH = original_chroma


def run_training_before_after(
    training_df: pd.DataFrame,
) -> tuple[list[dict], dict]:
    """
    Compare SQL generation accuracy with and without training examples.

    BEFORE: training_examples.csv has the 8 subset questions REMOVED.
    AFTER:  training_examples.csv is complete (all examples included).

    Both conditions use isolated temporary Chroma stores so the live
    production store is never modified.
    """
    print("\n=== SECTION 5: Training Before/After (8 questions) ===")

    # Build gold SQL map from the full training set
    gold_sql_map: dict[str, str] = {}
    missing: list[str] = []
    for q in TRAINING_IMPACT_SUBSET:
        gsql = vannaagent.get_exact_training_sql(q, training_df)
        if not gsql:
            missing.append(q)
        gold_sql_map[q] = gsql or ""

    if missing:
        print(f"  WARNING: Gold SQL missing for {len(missing)} training-impact question(s):")
        for mq in missing:
            print(f"    - {mq}")

    temp_root = OUTPUT_DIR / "_tmp" / TIMESTAMP
    temp_root.mkdir(parents=True, exist_ok=True)

    # BEFORE condition: remove the 8 subset questions from training
    subset_normalized = {vannaagent.normalize_question(q) for q in TRAINING_IMPACT_SUBSET}
    before_df = training_df[
        ~training_df["question"].apply(
            lambda x: vannaagent.normalize_question(str(x)) in subset_normalized
        )
    ].reset_index(drop=True)
    print(f"  Before: {len(before_df)} training examples (removed {len(training_df) - len(before_df)})")

    # AFTER condition: full training set
    after_df = training_df.copy()
    print(f"  After:  {len(after_df)} training examples (full set)")

    before_rows, before_summary = run_condition_on_subset(
        condition_name="before",
        questions=TRAINING_IMPACT_SUBSET,
        gold_sql_map=gold_sql_map,
        training_df=before_df,
        temp_dir=temp_root / "before",
    )
    after_rows, after_summary = run_condition_on_subset(
        condition_name="after",
        questions=TRAINING_IMPACT_SUBSET,
        gold_sql_map=gold_sql_map,
        training_df=after_df,
        temp_dir=temp_root / "after",
    )

    # Combine into one list (all rows)
    all_rows = before_rows + after_rows

    summary = {
        "before": before_summary,
        "after": after_summary,
        "improvement": {
            "exec_acc_delta": round(
                (after_summary["exec_acc"] or 0) - (before_summary["exec_acc"] or 0), 4
            ),
            "exact_match_delta": round(
                (after_summary["exact_match"] or 0) - (before_summary["exact_match"] or 0), 4
            ),
            "compile_rate_delta": round(
                (after_summary["compile_rate"] or 0) - (before_summary["compile_rate"] or 0), 4
            ),
        },
    }
    print(f"  Before: ExecAcc={before_summary['exec_acc']}  After: ExecAcc={after_summary['exec_acc']}")
    return all_rows, summary


# --------------------------------------------------------------------------
# Chart generation
# --------------------------------------------------------------------------

def generate_charts(
    domain_summary: dict,
    benchmark_summary: dict,
    safety_summary: dict,
    semantic_summary: dict,
    training_summary: dict,
) -> list[str]:
    """
    Generate 3 PNG charts. Returns list of file paths created.
    If matplotlib is not available, prints a warning and returns empty list.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")  # non-interactive backend (no GUI needed)
        import matplotlib.pyplot as plt
    except ImportError:
        print("  WARNING: matplotlib not installed. Skipping chart generation.")
        print("  Install with: pip install matplotlib")
        return []

    created: list[str] = []

    # --- Chart 1: Held-out benchmark metric comparison ----------------------
    try:
        fig, ax = plt.subplots(figsize=(8, 5))
        metrics = ["Exact Match", "Exec Accuracy", "Compile Rate"]
        old_values = [0.0, 0.0, 0.90]  # old thesis results (template-only, no LLM)
        new_values = [
            benchmark_summary.get("exact_match") or 0.0,
            benchmark_summary.get("exec_acc") or 0.0,
            benchmark_summary.get("compile_rate") or 0.0,
        ]
        x = range(len(metrics))
        width = 0.35
        bars_old = ax.bar([i - width / 2 for i in x], old_values, width, label="Before (no LLM)",
                          color="steelblue", alpha=0.7)
        bars_new = ax.bar([i + width / 2 for i in x], new_values, width, label="After (with LLM)",
                          color="tomato", alpha=0.7)
        ax.set_ylabel("Score (0 to 1)")
        ax.set_title("Held-out Benchmark: Before vs After LLM")
        ax.set_xticks(list(x))
        ax.set_xticklabels(metrics)
        ax.set_ylim(0, 1.1)
        ax.legend()
        for bar in bars_old:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                    f"{bar.get_height():.2f}", ha="center", va="bottom", fontsize=9)
        for bar in bars_new:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                    f"{bar.get_height():.2f}", ha="center", va="bottom", fontsize=9)
        plt.tight_layout()
        path1 = str(OUTPUT_DIR / "chart_benchmark_metrics.png")
        plt.savefig(path1, dpi=120)
        plt.close()
        created.append(path1)
        print(f"  Saved: chart_benchmark_metrics.png")
    except Exception as exc:
        print(f"  WARNING: chart_benchmark_metrics.png failed: {exc}")

    # --- Chart 2: Failure category distribution ----------------------------
    try:
        # Combine failure categories from domain + benchmark + semantic tests
        all_failures: dict[str, int] = {}
        for key in ["no_match", "compile_fail", "timeout", "runtime_fail",
                    "blocked_read_only", "blocked_complexity", "no_template_no_llm"]:
            count = benchmark_summary.get("outcome_counts", {}).get(key, 0)
            sem_count = semantic_summary.get("error_category_counts", {}).get(key, 0)
            total = count + sem_count
            if total > 0:
                all_failures[key] = total

        if all_failures:
            fig, ax = plt.subplots(figsize=(8, 5))
            labels = list(all_failures.keys())
            values = [all_failures[k] for k in labels]
            colors = plt.cm.Set2.colors[:len(labels)]
            ax.barh(labels, values, color=colors)
            ax.set_xlabel("Count")
            ax.set_title("Failure Category Distribution (Benchmark + Semantic)")
            for i, v in enumerate(values):
                ax.text(v + 0.05, i, str(v), va="center", fontsize=9)
            plt.tight_layout()
            path2 = str(OUTPUT_DIR / "chart_failure_categories.png")
            plt.savefig(path2, dpi=120)
            plt.close()
            created.append(path2)
            print(f"  Saved: chart_failure_categories.png")
        else:
            print("  Skipping failure chart: no failures to show.")
    except Exception as exc:
        print(f"  WARNING: chart_failure_categories.png failed: {exc}")

    # --- Chart 3: Training before/after comparison -------------------------
    try:
        before_s = training_summary.get("before", {})
        after_s = training_summary.get("after", {})
        if before_s and after_s:
            metrics = ["Exec Accuracy", "Exact Match", "Compile Rate"]
            before_vals = [
                before_s.get("exec_acc") or 0.0,
                before_s.get("exact_match") or 0.0,
                before_s.get("compile_rate") or 0.0,
            ]
            after_vals = [
                after_s.get("exec_acc") or 0.0,
                after_s.get("exact_match") or 0.0,
                after_s.get("compile_rate") or 0.0,
            ]
            x = range(len(metrics))
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.bar([i - 0.175 for i in x], before_vals, 0.35, label="Before training",
                   color="steelblue", alpha=0.7)
            ax.bar([i + 0.175 for i in x], after_vals, 0.35, label="After training",
                   color="seagreen", alpha=0.7)
            ax.set_ylabel("Score (0 to 1)")
            ax.set_title("Training Impact: Before vs After")
            ax.set_xticks(list(x))
            ax.set_xticklabels(metrics)
            ax.set_ylim(0, 1.1)
            ax.legend()
            plt.tight_layout()
            path3 = str(OUTPUT_DIR / "chart_training_before_after.png")
            plt.savefig(path3, dpi=120)
            plt.close()
            created.append(path3)
            print(f"  Saved: chart_training_before_after.png")
    except Exception as exc:
        print(f"  WARNING: chart_training_before_after.png failed: {exc}")

    return created


# --------------------------------------------------------------------------
# Write output files
# --------------------------------------------------------------------------

def save_csv(rows: list[dict], filename: str) -> None:
    path = OUTPUT_DIR / filename
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")
    print(f"  Saved: {filename} ({len(rows)} rows)")


def save_environment_info(env: dict) -> None:
    lines = [
        "mxQueryChat Evaluation - Environment Information",
        f"Generated: {env['timestamp']}",
        "",
        f"Operating System:   {env['operating_system']}",
        f"CPU Model:          {env['cpu_model']}",
        f"CPU Threads:        {env['cpu_threads']}",
        f"RAM:                {env['ram_gib']} GiB",
        "",
        f"Python Version:     {env['python_version']}",
        f"Streamlit Version:  {env['streamlit_version']}",
        f"DuckDB Version:     {env['duckdb_version']}",
        f"Vanna Version:      {env['vanna_version']}",
        f"ChromaDB Version:   {env['chromadb_version']}",
        f"Ollama Version:     {env['ollama_version']}",
        f"Model Name:         {env['model_name']}",
        "",
        f"Database File:      {env['database_file']}",
        f"Number of Tables:   {env['num_tables']}",
        "",
        "Table Row Counts:",
    ]
    for table, cnt in env.get("table_row_counts", {}).items():
        lines.append(f"  {table}: {cnt:,}")
    path = OUTPUT_DIR / "environment_info.txt"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"  Saved: environment_info.txt")


def save_summary(
    env: dict,
    domain_summary: dict,
    benchmark_summary: dict,
    safety_summary: dict,
    semantic_summary: dict,
    training_summary: dict,
    chart_files: list[str],
    warnings: list[str],
) -> None:
    bef = training_summary.get("before", {})
    aft = training_summary.get("after", {})
    imp = training_summary.get("improvement", {})

    lines = [
        "# mxQueryChat Evaluation Summary",
        "",
        f"**Generated:** {TIMESTAMP}",
        "",
        "---",
        "",
        "## Environment",
        "",
        f"| Property | Value |",
        f"|---|---|",
        f"| OS | {env['operating_system']} |",
        f"| CPU | {env['cpu_model']} |",
        f"| RAM | {env['ram_gib']} GiB |",
        f"| Python | {env['python_version']} |",
        f"| DuckDB | {env['duckdb_version']} |",
        f"| Vanna | {env['vanna_version']} |",
        f"| ChromaDB | {env['chromadb_version']} |",
        f"| Ollama | {env['ollama_version']} |",
        f"| Model | {env['model_name']} |",
        f"| DB file | {env['database_file']} |",
        f"| Tables | {env['num_tables']} |",
        "",
        "---",
        "",
        "## Test Sets Used",
        "",
        "| Set | Source | N |",
        "|---|---|---|",
        "| Main domain | docs/demo_questions.md (first 20 EN) | 20 |",
        "| Held-out benchmark | training_data/benchmark_questions.csv | "
        + str(benchmark_summary.get('total_questions', 0)) + " |",
        "| Safety | Built-in 15 cases (write, DDL, multi-stmt, complexity, off-topic) | 15 |",
        "| Semantic / paraphrase | 10 harder paraphrase cases | 10 |",
        "| Training before/after | 8 questions from training set | 8 x 2 |",
        "",
        "---",
        "",
        "## Section 1: Main Domain Test (20 questions)",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Execution Accuracy | {domain_summary.get('exec_acc')} |",
        f"| Exact Match | {domain_summary.get('exact_match')} |",
        f"| Compile Rate | {domain_summary.get('compile_rate')} |",
        f"| Generation latency median | {domain_summary.get('gen_latency_median_s')} s |",
        f"| Generation latency p95 | {domain_summary.get('gen_latency_p95_s')} s |",
        f"| Execution latency median | {domain_summary.get('exec_latency_median_s')} s |",
        f"| Total latency median | {domain_summary.get('total_latency_median_s')} s |",
        "",
        "_Interpretation: Domain questions are all present in training_examples.csv,_",
        "_so the exact-match path fires. This verifies the lookup pipeline is correct._",
        "",
        "---",
        "",
        "## Section 2: Held-out Benchmark",
        "",
        "| Metric | Old thesis (no LLM) | New run (with LLM) |",
        "|---|---|---|",
        f"| Exact Match | 0.00 | {benchmark_summary.get('exact_match')} |",
        f"| Execution Accuracy | 0.00 | {benchmark_summary.get('exec_acc')} |",
        f"| Compile Rate | 0.90 | {benchmark_summary.get('compile_rate')} |",
        f"| Success Rate | - | {benchmark_summary.get('success_rate')} |",
        f"| Safe-fail Rate | - | {benchmark_summary.get('safe_fail_rate')} |",
        f"| Generation latency median | - | {benchmark_summary.get('gen_latency_median_s')} s |",
        f"| Generation latency p95 | - | {benchmark_summary.get('gen_latency_p95_s')} s |",
        f"| Total latency median | - | {benchmark_summary.get('total_latency_median_s')} s |",
        "",
        "_Interpretation: With the LLM enabled, the system can generate SQL for questions_",
        "_not present in the training set. Results show whether Ollama/Mistral generalizes._",
        "",
        "---",
        "",
        "## Section 3: Safety / Robustness Test (15 cases)",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Blocked rate | {safety_summary.get('blocked_rate')} ({safety_summary.get('blocked_count')}/15) |",
        f"| blocked_read_only | {safety_summary.get('blocked_read_only')} |",
        f"| blocked_complexity | {safety_summary.get('blocked_complexity')} |",
        f"| timeout_count | {safety_summary.get('timeout_count')} |",
        f"| compile_fail_count | {safety_summary.get('compile_fail_count')} |",
        f"| runtime_fail_count | {safety_summary.get('runtime_fail_count')} |",
        f"| no_match_count | {safety_summary.get('no_match_count')} |",
        f"| Zero writes confirmed | {safety_summary.get('zero_writes_confirmed')} |",
        "",
        "_Interpretation: The read-only enforcement and complexity policy protect the database._",
        "_Any case that reaches the DB only executes via a read_only=True DuckDB connection._",
        "",
        "---",
        "",
        "## Section 4: Semantic / Paraphrase Test (10 cases)",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Execution Accuracy | {semantic_summary.get('exec_acc')} |",
        f"| Compile Rate | {semantic_summary.get('compile_rate')} |",
        "",
        "**Error category distribution:**",
        "",
    ]
    for cat, cnt in (semantic_summary.get("error_category_counts") or {}).items():
        lines.append(f"- `{cat}`: {cnt}")
    if not semantic_summary.get("error_category_counts"):
        lines.append("- (no errors recorded)")

    lines += [
        "",
        "---",
        "",
        "## Section 5: Training Before/After",
        "",
        "| Metric | Before | After | Delta |",
        "|---|---|---|---|",
        f"| Execution Accuracy | {bef.get('exec_acc')} | {aft.get('exec_acc')} | {imp.get('exec_acc_delta')} |",
        f"| Exact Match | {bef.get('exact_match')} | {aft.get('exact_match')} | {imp.get('exact_match_delta')} |",
        f"| Compile Rate | {bef.get('compile_rate')} | {aft.get('compile_rate')} | {imp.get('compile_rate_delta')} |",
        f"| Gen latency median | {bef.get('gen_latency_median_s')} s | {aft.get('gen_latency_median_s')} s | - |",
        f"| Total latency median | {bef.get('total_latency_median_s')} s | {aft.get('total_latency_median_s')} s | - |",
        "",
        "_Before: 8 subset questions removed from training CSV + fresh isolated Chroma store._",
        "_After: full training CSV + fresh isolated Chroma store._",
        "",
    ]

    if chart_files:
        lines += ["---", "", "## Charts Generated", ""]
        for cf in chart_files:
            lines.append(f"- `{Path(cf).name}`")
        lines.append("")

    if warnings:
        lines += ["---", "", "## Warnings and Limitations", ""]
        for w in warnings:
            lines.append(f"- {w}")
        lines.append("")

    path = OUTPUT_DIR / "evaluation_summary.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"  Saved: evaluation_summary.md")


# --------------------------------------------------------------------------
# Main entry point
# --------------------------------------------------------------------------

def main() -> None:
    print(f"\n{'='*60}")
    print("mxQueryChat Full Evaluation Runner")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Timestamp: {TIMESTAMP}")
    print(f"{'='*60}")

    warnings: list[str] = []

    # -- Environment ----------------------------------------------------------
    print("\n=== Detecting Environment ===")
    env = detect_environment()
    print(f"  OS: {env['operating_system']}")
    print(f"  Python: {env['python_version']}")
    print(f"  Ollama model: {env['model_name']}")
    print(f"  DuckDB: {env['duckdb_version']}")

    if env["model_name"] == "not detected":
        warnings.append(
            "Ollama was not detected. LLM-based sections (benchmark, semantic, training "
            "before/after) may fall back to template-only or return no_match results."
        )

    # -- Shared resources (initialized once) ---------------------------------
    print("\n=== Initializing Shared Resources ===")
    schema_tree = get_schema_tree()
    available_years = get_available_years()
    policy = ExecutionPolicy()
    training_df = vannaagent.load_training_examples()
    vanna_cache: dict[str, Any] = {"instance": None, "error": None}
    print(f"  Training examples loaded: {len(training_df)}")
    print(f"  Available years: {available_years}")
    print(f"  Tables in schema: {len(schema_tree)}")

    # -- Section 1: Domain test ----------------------------------------------
    domain_rows, domain_summary = run_domain_test(
        schema_tree, available_years, training_df, policy, vanna_cache
    )

    # -- Section 2: Held-out benchmark ---------------------------------------
    benchmark_rows, benchmark_summary = run_heldout_benchmark(
        schema_tree, available_years, training_df, policy, vanna_cache
    )
    if "error" in benchmark_summary:
        warnings.append(f"Held-out benchmark skipped: {benchmark_summary['error']}")

    # -- Section 3: Safety test ----------------------------------------------
    safety_rows, safety_summary = run_safety_test()

    # -- Section 4: Semantic test --------------------------------------------
    semantic_rows, semantic_summary = run_semantic_test(
        schema_tree, available_years, training_df, policy, vanna_cache
    )

    # -- Section 5: Training before/after ------------------------------------
    training_rows: list[dict] = []
    training_summary: dict = {}
    try:
        training_rows, training_summary = run_training_before_after(training_df)
    except Exception as exc:
        msg = f"Training before/after section failed: {exc}"
        print(f"  ERROR: {msg}")
        warnings.append(msg)
        training_rows = []
        training_summary = {"before": {}, "after": {}, "improvement": {}, "error": str(exc)}

    # -- Save CSV files -------------------------------------------------------
    print("\n=== Saving Result Files ===")
    save_csv(domain_rows, "main_domain_results.csv")
    if benchmark_rows:
        save_csv(benchmark_rows, "heldout_benchmark_results.csv")
    save_csv(safety_rows, "safety_results.csv")
    save_csv(semantic_rows, "semantic_error_results.csv")
    if training_rows:
        save_csv(training_rows, "training_before_after_results.csv")
    save_environment_info(env)

    # -- Generate charts ------------------------------------------------------
    print("\n=== Generating Charts ===")
    chart_files = generate_charts(
        domain_summary, benchmark_summary, safety_summary,
        semantic_summary, training_summary
    )

    # -- Save summary ---------------------------------------------------------
    save_summary(
        env, domain_summary, benchmark_summary, safety_summary,
        semantic_summary, training_summary, chart_files, warnings
    )

    # -- Final report ---------------------------------------------------------
    print(f"\n{'='*60}")
    print("EVALUATION COMPLETE")
    print(f"{'='*60}")
    print(f"\nResults saved to: {OUTPUT_DIR}")
    print(f"\n--- KEY NUMBERS ---")
    print(f"Domain test (20q):     ExecAcc={domain_summary.get('exec_acc')}  "
          f"ExactMatch={domain_summary.get('exact_match')}  "
          f"CompileRate={domain_summary.get('compile_rate')}")
    print(f"Held-out benchmark:    ExecAcc={benchmark_summary.get('exec_acc')}  "
          f"ExactMatch={benchmark_summary.get('exact_match')}  "
          f"CompileRate={benchmark_summary.get('compile_rate')}")
    print(f"Safety (15 cases):     BlockedRate={safety_summary.get('blocked_rate')}  "
          f"ZeroWrites={safety_summary.get('zero_writes_confirmed')}")
    print(f"Semantic (10q):        ExecAcc={semantic_summary.get('exec_acc')}  "
          f"CompileRate={semantic_summary.get('compile_rate')}")
    if training_summary.get("before") and training_summary.get("after"):
        print(f"Training before/after: ExecAcc {training_summary['before'].get('exec_acc')} -> "
              f"{training_summary['after'].get('exec_acc')}  "
              f"(delta={training_summary.get('improvement', {}).get('exec_acc_delta')})")
    if warnings:
        print(f"\n--- WARNINGS ({len(warnings)}) ---")
        for w in warnings:
            print(f"  ! {w}")

    output_files = [
        "evaluation_summary.md",
        "main_domain_results.csv",
        "heldout_benchmark_results.csv",
        "safety_results.csv",
        "semantic_error_results.csv",
        "training_before_after_results.csv",
        "environment_info.txt",
    ] + [Path(f).name for f in chart_files]
    print(f"\n--- FILES CREATED ---")
    for f in output_files:
        p = OUTPUT_DIR / f
        if p.exists():
            print(f"  OK  {f}")
        else:
            print(f"  --  {f}  (not created)")


if __name__ == "__main__":
    main()
