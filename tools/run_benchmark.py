"""

Purpose:
This script runs repeatable benchmark experiments over local question sets and reports quality plus
latency metrics for the NL-to-SQL pipeline.

What This File Contains:
- Markdown and CSV benchmark-question loaders.
- Per-question execution pipeline with guardrails, template/LLM generation, validation, and execution.
- Optional gold-SQL comparison helpers for Exact Match and Execution Accuracy.
- Report writers that emit JSON and CSV benchmark artifacts.

Key Invariants and Safety Guarantees:
- Queries are validated through read-only and complexity checks before execution.
- DuckDB execution remains read-only and row-limited.
- Outcome taxonomy remains stable across runs for comparable benchmarking.
- Gold-result comparisons are only computed when benchmark rows provide gold SQL.

How Other Modules Use This File:
This script is run manually from the command line. Its outputs support progress tracking, thesis tables,
and local held-out benchmark evaluation without requiring Spider/BIRD adapters.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from datetime import datetime, timezone
from pathlib import Path

import duckdb
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sql_guard import validate_read_only_sql
from src.core import query_logic
from src.db.execution_policy import (
    ExecutionPolicy,
    apply_row_limit,
    validate_sql_complexity,
)

DUCKDB_PATH = "mxquerychat.duckdb"
MODEL_TIMEOUT_SECONDS = 65
DEFAULT_BENCHMARK_CSV = "training_data/benchmark_questions.csv"


def run_with_timeout(fn, timeout_seconds: int):
    executor = ThreadPoolExecutor(max_workers=1)
    future = executor.submit(fn)
    try:
        return future.result(timeout=timeout_seconds), None
    except FutureTimeoutError:
        future.cancel()
        return None, f"Model timeout after {timeout_seconds} seconds."
    except Exception as exc:
        return None, str(exc)
    finally:
        executor.shutdown(wait=False, cancel_futures=True)


def validate_sql_compiles(sql: str) -> str:
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


def normalize_sql_for_exact_match(sql: str) -> str:
    if not sql:
        return ""
    normalized = sql.strip().rstrip(";").lower()
    normalized = normalized.replace('"', "'")
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized


def canonicalize_dataframe(df: pd.DataFrame) -> tuple[list[str], list[tuple]]:
    if df is None:
        return [], []
    columns = sorted(list(df.columns))
    work = df[columns].copy()
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
    return actual_cols == expected_cols and actual_rows == expected_rows


def execute_sql_read_only(sql: str, policy: ExecutionPolicy) -> tuple[pd.DataFrame, int, str]:
    limited_sql = apply_row_limit(sql, policy.max_rows)
    started = time.time()
    con = duckdb.connect(DUCKDB_PATH, read_only=True)
    try:
        df = con.execute(limited_sql).fetchdf()
        elapsed_ms = int((time.time() - started) * 1000)
        return df, elapsed_ms, ""
    except Exception as exc:
        elapsed_ms = int((time.time() - started) * 1000)
        _ = elapsed_ms
        return pd.DataFrame(), elapsed_ms, str(exc)
    finally:
        con.close()


def get_schema_tree() -> dict:
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
        schema = {}
        for (table_name,) in tables:
            cols = con.execute(
                f"""
                SELECT column_name, data_type
                FROM information_schema.columns
                WHERE table_name = '{table_name}'
                ORDER BY ordinal_position
                """
            ).fetchall()
            schema[table_name] = cols
        return schema
    finally:
        con.close()


def parse_questions_from_markdown(path: Path) -> list[str]:
    content = path.read_text(encoding="utf-8", errors="replace")
    questions: list[str] = []
    for line in content.splitlines():
        stripped = line.strip()
        en_match = re.match(r"^\d+\.\s+EN:\s*(.+)$", stripped)
        if en_match:
            questions.append(en_match.group(1).strip())
            continue
        q_match = re.match(r"^Q:\s*(.+)$", stripped)
        if q_match:
            questions.append(q_match.group(1).strip())
    return questions


def load_benchmark_questions() -> list[str]:
    paths = [Path("docs/demo_questions.md"), Path("docs/tricky_questions.md")]
    all_questions: list[str] = []
    for path in paths:
        if path.exists():
            all_questions.extend(parse_questions_from_markdown(path))

    seen: set[str] = set()
    unique: list[str] = []
    for question in all_questions:
        norm = " ".join(question.lower().split())
        if norm not in seen:
            seen.add(norm)
            unique.append(question)
    return unique


def load_benchmark_cases(csv_path: Path | None = None) -> list[dict]:
    if csv_path and csv_path.exists():
        df = pd.read_csv(csv_path, dtype=str, keep_default_na=False).fillna("")
        required = {"question"}
        if not required.issubset(df.columns):
            raise ValueError(f"Benchmark CSV must contain columns: {sorted(required)}")
        cases: list[dict] = []
        for _, row in df.iterrows():
            question = str(row.get("question", "")).strip()
            if not question:
                continue
            cases.append(
                {
                    "question": question,
                    "gold_sql": str(row.get("gold_sql", "")).strip(),
                    "difficulty": str(row.get("difficulty", "")).strip(),
                    "category": str(row.get("category", "")).strip(),
                }
            )
        return cases

    return [{"question": question, "gold_sql": "", "difficulty": "", "category": ""} for question in load_benchmark_questions()]


def classify_safe_fail(reason: str) -> bool:
    lowered = (reason or "").lower()
    tokens = [
        "guardrail",
        "blocked",
        "no_match",
        "timeout",
        "read-only",
        "complexity",
        "no_template_no_llm",
    ]
    return any(token in lowered for token in tokens)


def run_single_question(
    case: dict,
    schema_tree: dict,
    policy: ExecutionPolicy,
    use_llm: bool,
    vanna_instance,
) -> dict:
    question = case["question"]
    gold_sql = str(case.get("gold_sql", "")).strip()
    start_total = time.time()
    generation_start = time.time()

    local_block = query_logic.get_local_guardrail_message(question)
    if local_block:
        return {
            "question": question,
            "difficulty": case.get("difficulty", ""),
            "category": case.get("category", ""),
            "gold_sql": gold_sql,
            "outcome": "safe_fail",
            "reason": f"guardrail:{local_block}",
            "sql": "",
            "compiled": False,
            "exact_match": False,
            "exec_correct": False,
            "generation_ms": int((time.time() - generation_start) * 1000),
            "execution_ms": 0,
            "rows": 0,
            "total_ms": int((time.time() - start_total) * 1000),
        }

    template_sql, template_note = query_logic.build_template_sql(question)
    sql = ""
    generation_path = ""
    generation_reason = ""
    if template_sql:
        sql = template_sql
        generation_path = "template"
        generation_reason = template_note
    elif use_llm and vanna_instance is not None:
        sql, notes, error_code = query_logic.generate_sql_with_retry(
            generate_sql_fn=lambda prompt: vanna_instance.generate_sql(prompt),
            question_text=question,
            schema_tree=schema_tree,
            compile_sql_fn=validate_sql_compiles,
            timeout_seconds=MODEL_TIMEOUT_SECONDS,
            run_with_timeout_fn=run_with_timeout,
        )
        generation_path = "llm"
        generation_reason = "; ".join(notes[-2:]) if notes else ""
        if error_code:
            return {
                "question": question,
                "difficulty": case.get("difficulty", ""),
                "category": case.get("category", ""),
                "gold_sql": gold_sql,
                "outcome": "safe_fail",
                "reason": f"llm_{error_code}",
                "sql": "",
                "compiled": False,
                "exact_match": False,
                "exec_correct": False,
                "generation_ms": int((time.time() - generation_start) * 1000),
                "execution_ms": 0,
                "rows": 0,
                "total_ms": int((time.time() - start_total) * 1000),
            }
    else:
        return {
            "question": question,
            "difficulty": case.get("difficulty", ""),
            "category": case.get("category", ""),
            "gold_sql": gold_sql,
            "outcome": "safe_fail",
            "reason": "no_template_no_llm",
            "sql": "",
            "compiled": False,
            "exact_match": False,
            "exec_correct": False,
            "generation_ms": int((time.time() - generation_start) * 1000),
            "execution_ms": 0,
            "rows": 0,
            "total_ms": int((time.time() - start_total) * 1000),
        }

    generation_ms = int((time.time() - generation_start) * 1000)

    is_read_only, read_only_message = validate_read_only_sql(sql)
    if not is_read_only:
        return {
            "question": question,
            "difficulty": case.get("difficulty", ""),
            "category": case.get("category", ""),
            "gold_sql": gold_sql,
            "outcome": "safe_fail",
            "reason": f"read_only:{read_only_message}",
            "sql": sql,
            "compiled": False,
            "exact_match": False,
            "exec_correct": False,
            "generation_ms": generation_ms,
            "execution_ms": 0,
            "rows": 0,
            "total_ms": int((time.time() - start_total) * 1000),
        }

    complexity_ok, complexity_message = validate_sql_complexity(sql, policy)
    if not complexity_ok:
        return {
            "question": question,
            "difficulty": case.get("difficulty", ""),
            "category": case.get("category", ""),
            "gold_sql": gold_sql,
            "outcome": "safe_fail",
            "reason": f"complexity:{complexity_message}",
            "sql": sql,
            "compiled": False,
            "exact_match": False,
            "exec_correct": False,
            "generation_ms": generation_ms,
            "execution_ms": 0,
            "rows": 0,
            "total_ms": int((time.time() - start_total) * 1000),
        }

    compile_error = validate_sql_compiles(sql)
    if compile_error:
        return {
            "question": question,
            "difficulty": case.get("difficulty", ""),
            "category": case.get("category", ""),
            "gold_sql": gold_sql,
            "outcome": "compile_fail",
            "reason": compile_error,
            "sql": sql,
            "compiled": False,
            "exact_match": False,
            "exec_correct": False,
            "generation_ms": generation_ms,
            "execution_ms": 0,
            "rows": 0,
            "total_ms": int((time.time() - start_total) * 1000),
        }

    df, execution_ms, exec_error = execute_sql_read_only(sql, policy)
    if exec_error:
        return {
            "question": question,
            "difficulty": case.get("difficulty", ""),
            "category": case.get("category", ""),
            "gold_sql": gold_sql,
            "outcome": "safe_fail" if classify_safe_fail(exec_error) else "runtime_fail",
            "reason": exec_error,
            "sql": sql,
            "compiled": True,
            "exact_match": False,
            "exec_correct": False,
            "generation_ms": generation_ms,
            "execution_ms": execution_ms,
            "rows": 0,
            "total_ms": int((time.time() - start_total) * 1000),
            "generation_path": generation_path,
            "generation_detail": generation_reason,
        }

    compiled = True
    exact_match = bool(gold_sql) and (
        normalize_sql_for_exact_match(sql) == normalize_sql_for_exact_match(gold_sql)
    )
    exec_correct = False
    if gold_sql:
        gold_compile_error = validate_sql_compiles(gold_sql)
        if not gold_compile_error:
            gold_df, _gold_execution_ms, gold_exec_error = execute_sql_read_only(gold_sql, policy)
            if not gold_exec_error:
                exec_correct = compare_query_results(df, gold_df)

    return {
        "question": question,
        "difficulty": case.get("difficulty", ""),
        "category": case.get("category", ""),
        "gold_sql": gold_sql,
        "outcome": "success",
        "reason": "ok",
        "sql": sql,
        "compiled": compiled,
        "exact_match": exact_match,
        "exec_correct": exec_correct,
        "generation_ms": generation_ms,
        "execution_ms": execution_ms,
        "rows": int(len(df)),
        "total_ms": int((time.time() - start_total) * 1000),
        "generation_path": generation_path,
        "generation_detail": generation_reason,
    }


def build_summary(results: list[dict]) -> dict:
    total = len(results)
    counts = {
        "success": sum(1 for r in results if r["outcome"] == "success"),
        "compile_fail": sum(1 for r in results if r["outcome"] == "compile_fail"),
        "safe_fail": sum(1 for r in results if r["outcome"] == "safe_fail"),
        "runtime_fail": sum(1 for r in results if r["outcome"] == "runtime_fail"),
    }
    generation_ms = [r.get("generation_ms", 0) for r in results]
    execution_ms = [
        r.get("execution_ms", 0) for r in results if r.get("execution_ms", 0) > 0
    ]
    total_ms = [r.get("total_ms", 0) for r in results if r.get("total_ms", 0) > 0]
    gold_results = [r for r in results if str(r.get("gold_sql", "")).strip()]
    gold_total = len(gold_results)
    exact_total = sum(1 for r in gold_results if bool(r.get("exact_match")))
    exec_total = sum(1 for r in gold_results if bool(r.get("exec_correct")))
    compile_total = sum(1 for r in gold_results if bool(r.get("compiled")))

    def _ratio(value: int) -> float:
        return round((value / total) * 100.0, 2) if total else 0.0

    return {
        "total_questions": total,
        "counts": counts,
        "rates_percent": {
            "success_rate": _ratio(counts["success"]),
            "compile_fail_rate": _ratio(counts["compile_fail"]),
            "safe_fail_rate": _ratio(counts["safe_fail"]),
            "runtime_fail_rate": _ratio(counts["runtime_fail"]),
        },
        "latency_ms": {
            "avg_generation_ms": round(sum(generation_ms) / total, 2) if total else 0.0,
            "avg_execution_ms": round(sum(execution_ms) / len(execution_ms), 2)
            if execution_ms
            else 0.0,
            "median_generation_ms": round(statistics.median(generation_ms), 2)
            if generation_ms
            else 0.0,
            "median_execution_ms": round(statistics.median(execution_ms), 2)
            if execution_ms
            else 0.0,
            "median_total_ms": round(statistics.median(total_ms), 2)
            if total_ms
            else 0.0,
        },
        "prd_kpis": {
            "compile_rate": _ratio(total - counts["compile_fail"]),
            "safe_fail_rate": _ratio(counts["safe_fail"]),
            "median_latency_ms": round(statistics.median(total_ms), 2)
            if total_ms
            else 0.0,
        },
        "gold_metrics": {
            "gold_question_count": gold_total,
            "exact_match_rate": round(exact_total / gold_total, 4) if gold_total else None,
            "exec_acc": round(exec_total / gold_total, 4) if gold_total else None,
            "compile_rate": round(compile_total / gold_total, 4) if gold_total else None,
        },
    }


def write_reports(results: list[dict], summary: dict, output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    json_path = output_dir / f"benchmark_{stamp}.json"
    csv_path = output_dir / f"benchmark_{stamp}.csv"

    json_path.write_text(
        json.dumps({"summary": summary, "results": results}, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )
    pd.DataFrame(results).to_csv(csv_path, index=False, encoding="utf-8-sig")
    return json_path, csv_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run mxquerychat benchmark harness.")
    parser.add_argument(
        "--output-dir",
        default="reports",
        help="Directory where benchmark reports are written.",
    )
    parser.add_argument(
        "--max-questions",
        type=int,
        default=0,
        help="Limit number of questions (0 means all).",
    )
    parser.add_argument(
        "--use-llm",
        action="store_true",
        help="Allow LLM fallback when no deterministic template matches.",
    )
    parser.add_argument(
        "--questions-csv",
        default=DEFAULT_BENCHMARK_CSV,
        help="Optional benchmark CSV with columns question,gold_sql,difficulty,category.",
    )
    args = parser.parse_args()

    csv_path = Path(args.questions_csv) if args.questions_csv else None
    cases = load_benchmark_cases(csv_path if csv_path and csv_path.exists() else None)
    if args.max_questions and args.max_questions > 0:
        cases = cases[: args.max_questions]

    schema_tree = get_schema_tree()
    policy = ExecutionPolicy()

    vanna_instance = None
    if args.use_llm:
        from vannaagent import get_vanna

        vanna_instance = get_vanna()

    results = [
        run_single_question(
            case=case,
            schema_tree=schema_tree,
            policy=policy,
            use_llm=args.use_llm,
            vanna_instance=vanna_instance,
        )
        for case in cases
    ]
    summary = build_summary(results)
    json_path, csv_path = write_reports(results, summary, Path(args.output_dir))

    print("Benchmark complete.")
    print(f"Questions: {summary['total_questions']}")
    print(f"Success rate: {summary['rates_percent']['success_rate']}%")
    print(f"Safe-fail rate: {summary['rates_percent']['safe_fail_rate']}%")
    print(f"Compile-fail rate: {summary['rates_percent']['compile_fail_rate']}%")
    print(f"Runtime-fail rate: {summary['rates_percent']['runtime_fail_rate']}%")
    print(f"Compile rate: {summary['prd_kpis']['compile_rate']}%")
    print(f"Gold Exact Match: {summary['gold_metrics']['exact_match_rate']}")
    print(f"Gold ExecAcc: {summary['gold_metrics']['exec_acc']}")
    print(f"Median latency: {summary['prd_kpis']['median_latency_ms']} ms")
    print(f"JSON report: {json_path}")
    print(f"CSV report: {csv_path}")


if __name__ == "__main__":
    main()


