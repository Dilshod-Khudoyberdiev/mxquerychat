"""Run NL-to-SQL benchmark on docs/demo_questions.md + docs/tricky_questions.md."""

from __future__ import annotations

import argparse
import json
import re
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
    run_query_with_timeout,
    validate_sql_complexity,
)

DUCKDB_PATH = "mxquerychat.duckdb"
MODEL_TIMEOUT_SECONDS = 65


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
    question: str,
    schema_tree: dict,
    policy: ExecutionPolicy,
    use_llm: bool,
    vanna_instance,
) -> dict:
    start_total = time.time()
    generation_start = time.time()

    local_block = query_logic.get_local_guardrail_message(question)
    if local_block:
        return {
            "question": question,
            "outcome": "safe_fail",
            "reason": f"guardrail:{local_block}",
            "sql": "",
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
                "outcome": "safe_fail",
                "reason": f"llm_{error_code}",
                "sql": "",
                "generation_ms": int((time.time() - generation_start) * 1000),
                "execution_ms": 0,
                "rows": 0,
                "total_ms": int((time.time() - start_total) * 1000),
            }
    else:
        return {
            "question": question,
            "outcome": "safe_fail",
            "reason": "no_template_no_llm",
            "sql": "",
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
            "outcome": "safe_fail",
            "reason": f"read_only:{read_only_message}",
            "sql": sql,
            "generation_ms": generation_ms,
            "execution_ms": 0,
            "rows": 0,
            "total_ms": int((time.time() - start_total) * 1000),
        }

    complexity_ok, complexity_message = validate_sql_complexity(sql, policy)
    if not complexity_ok:
        return {
            "question": question,
            "outcome": "safe_fail",
            "reason": f"complexity:{complexity_message}",
            "sql": sql,
            "generation_ms": generation_ms,
            "execution_ms": 0,
            "rows": 0,
            "total_ms": int((time.time() - start_total) * 1000),
        }

    compile_error = validate_sql_compiles(sql)
    if compile_error:
        return {
            "question": question,
            "outcome": "compile_fail",
            "reason": compile_error,
            "sql": sql,
            "generation_ms": generation_ms,
            "execution_ms": 0,
            "rows": 0,
            "total_ms": int((time.time() - start_total) * 1000),
        }

    limited_sql = apply_row_limit(sql, policy.max_rows)
    df, execution_seconds, exec_error = run_query_with_timeout(
        DUCKDB_PATH, limited_sql, policy.timeout_seconds
    )
    execution_ms = int(execution_seconds * 1000)
    if exec_error:
        return {
            "question": question,
            "outcome": "safe_fail" if classify_safe_fail(exec_error) else "runtime_fail",
            "reason": exec_error,
            "sql": sql,
            "generation_ms": generation_ms,
            "execution_ms": execution_ms,
            "rows": 0,
            "total_ms": int((time.time() - start_total) * 1000),
            "generation_path": generation_path,
            "generation_detail": generation_reason,
        }

    return {
        "question": question,
        "outcome": "success",
        "reason": "ok",
        "sql": sql,
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
    execution_ms = [r.get("execution_ms", 0) for r in results if r.get("execution_ms", 0) > 0]

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
    args = parser.parse_args()

    questions = load_benchmark_questions()
    if args.max_questions and args.max_questions > 0:
        questions = questions[: args.max_questions]

    schema_tree = get_schema_tree()
    policy = ExecutionPolicy()

    vanna_instance = None
    if args.use_llm:
        from vannaagent import get_vanna

        vanna_instance = get_vanna()

    results = [
        run_single_question(
            question=question,
            schema_tree=schema_tree,
            policy=policy,
            use_llm=args.use_llm,
            vanna_instance=vanna_instance,
        )
        for question in questions
    ]
    summary = build_summary(results)
    json_path, csv_path = write_reports(results, summary, Path(args.output_dir))

    print("Benchmark complete.")
    print(f"Questions: {summary['total_questions']}")
    print(f"Success rate: {summary['rates_percent']['success_rate']}%")
    print(f"Safe-fail rate: {summary['rates_percent']['safe_fail_rate']}%")
    print(f"Compile-fail rate: {summary['rates_percent']['compile_fail_rate']}%")
    print(f"Runtime-fail rate: {summary['rates_percent']['runtime_fail_rate']}%")
    print(f"JSON report: {json_path}")
    print(f"CSV report: {csv_path}")


if __name__ == "__main__":
    main()
