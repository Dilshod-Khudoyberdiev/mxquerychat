"""CLI utility: measures remaining Chapter 6 evaluation items and writes machine-readable + Markdown artifacts to outputs/chapter6_missing_eval."""

from __future__ import annotations

import ctypes
import importlib.metadata
import json
import os
import shutil
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools import evaluation_runner
from tools import run_benchmark
from tools.summarize_metrics import load_metric_events, summarize_metric_events
import vannaagent

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "chapter6_missing_eval"
METRICS_LOG_PATH = PROJECT_ROOT / "logs" / "metrics.jsonl"
LOCAL_BENCHMARK_CSV_PATH = PROJECT_ROOT / "training_data" / "benchmark_questions.csv"

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

SEMANTIC_EVAL_CASES = [
    {
        "question": "Show revenue by tariff association and federal state.",
        "gold_sql": (
            "SELECT t.name AS tarifverbund_name, "
            "rb.bundesland_name, "
            "SUM(tv.umsatz_eur) AS umsatz_eur "
            "FROM ticket_verkaeufe tv "
            "JOIN tarifverbuende t ON tv.tarifverbund_id = t.tarifverbund_id "
            "JOIN postleitzahlen p ON CAST(tv.plz AS VARCHAR) = p.plz "
            "JOIN regionen_bundesland rb ON p.bundesland_code2 = rb.bundesland_code2 "
            "GROUP BY t.name, rb.bundesland_name "
            "ORDER BY t.name, umsatz_eur DESC;"
        ),
    },
    {
        "question": "For each state, show top 3 ticket types by revenue.",
        "gold_sql": (
            "WITH revenue_by_state AS ("
            "SELECT rb.bundesland_name, tp.ticket_name, SUM(tv.umsatz_eur) AS revenue_eur "
            "FROM ticket_verkaeufe tv "
            "JOIN ticket_produkte tp ON tv.ticket_code = tp.ticket_code "
            "JOIN postleitzahlen p ON CAST(tv.plz AS VARCHAR) = p.plz "
            "JOIN regionen_bundesland rb ON p.bundesland_code2 = rb.bundesland_code2 "
            "GROUP BY rb.bundesland_name, tp.ticket_name"
            "), ranked AS ("
            "SELECT bundesland_name, ticket_name, revenue_eur, "
            "ROW_NUMBER() OVER (PARTITION BY bundesland_name ORDER BY revenue_eur DESC) AS rn "
            "FROM revenue_by_state"
            ") "
            "SELECT bundesland_name, ticket_name, revenue_eur "
            "FROM ranked WHERE rn <= 3 "
            "ORDER BY bundesland_name, revenue_eur DESC;"
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
            "WHERE tv.jahr = 2025 "
            "GROUP BY rb.bundesland_name, tp.ticket_name "
            "ORDER BY rb.bundesland_name, revenue_eur DESC;"
        ),
    },
    {
        "question": "Show reporting office revenue with its state and tariff association.",
        "gold_sql": (
            "SELECT m.meldestelle_name, rb.bundesland_name, t.name AS tarifverbund_name, "
            "SUM(tv.umsatz_eur) AS revenue_eur "
            "FROM ticket_verkaeufe tv "
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
            "tp.preis_eur AS catalog_price_eur "
            "FROM ticket_verkaeufe tv "
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
            "WITH revenue_by_state AS ("
            "SELECT rb.bundesland_name, tp.ticket_name, SUM(tv.umsatz_eur) AS revenue_eur "
            "FROM ticket_verkaeufe tv "
            "JOIN ticket_produkte tp ON tv.ticket_code = tp.ticket_code "
            "JOIN postleitzahlen p ON CAST(tv.plz AS VARCHAR) = p.plz "
            "JOIN regionen_bundesland rb ON p.bundesland_code2 = rb.bundesland_code2 "
            "GROUP BY rb.bundesland_name, tp.ticket_name"
            "), ranked AS ("
            "SELECT bundesland_name, ticket_name, revenue_eur, "
            "ROW_NUMBER() OVER (PARTITION BY bundesland_name ORDER BY revenue_eur DESC) AS rn "
            "FROM revenue_by_state"
            ") "
            "SELECT bundesland_name, ticket_name, revenue_eur "
            "FROM ranked WHERE rn <= 5 "
            "ORDER BY bundesland_name, revenue_eur DESC;"
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
            "WHERE t.status = 'aktiv' "
            "GROUP BY rb.bundesland_name "
            "ORDER BY revenue_eur DESC;"
        ),
    },
    {
        "question": "Show revenue by ticket type and month for 2025.",
        "gold_sql": (
            "SELECT tv.monat, tp.ticket_name, SUM(tv.umsatz_eur) AS revenue_eur "
            "FROM ticket_verkaeufe tv "
            "JOIN ticket_produkte tp ON tv.ticket_code = tp.ticket_code "
            "WHERE tv.jahr = 2025 "
            "GROUP BY tv.monat, tp.ticket_name "
            "ORDER BY tv.monat, revenue_eur DESC;"
        ),
    },
    {
        "question": "Show revenue by tariff association and month for 2024.",
        "gold_sql": (
            "SELECT tv.monat, t.name AS tarifverbund_name, SUM(tv.umsatz_eur) AS revenue_eur "
            "FROM ticket_verkaeufe tv "
            "JOIN tarifverbuende t ON tv.tarifverbund_id = t.tarifverbund_id "
            "WHERE tv.jahr = 2024 "
            "GROUP BY tv.monat, t.name "
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


@dataclass
class ConditionResult:
    name: str
    exec_acc: float
    exact_match: float
    compile_rate: float
    gen_median: float
    total_median: float
    per_question: list[dict[str, Any]]
    error: str | None = None


class MEMORYSTATUSEX(ctypes.Structure):
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


def rounded(value: float) -> float:
    return round(float(value), 6)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")


def write_markdown(path: Path, content: str) -> None:
    path.write_text(content.rstrip() + "\n", encoding="utf-8")


def safe_run(command: list[str]) -> tuple[bool, str]:
    try:
        result = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
    except Exception as exc:  # pragma: no cover - environment dependent
        return False, str(exc)

    text = (result.stdout or "").strip() or (result.stderr or "").strip()
    return result.returncode == 0, text


def detect_cpu_model() -> str:
    ok, text = safe_run(
        [
            "powershell",
            "-NoProfile",
            "-Command",
            (
                "(Get-ItemProperty "
                "'HKLM:\\HARDWARE\\DESCRIPTION\\System\\CentralProcessor\\0').ProcessorNameString"
            ),
        ]
    )
    if ok and text:
        return text.strip().splitlines()[0].strip()
    return "not detected"


def detect_memory_bytes() -> int | None:
    memory = MEMORYSTATUSEX()
    memory.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
    try:
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(memory)):
            return int(memory.ullTotalPhys)
    except Exception:  # pragma: no cover - Windows API dependent
        return None
    return None


def bytes_to_gib_string(value: int | None) -> str:
    if value is None:
        return "not detected"
    return f"{value / (1024 ** 3):.2f} GiB"


def package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not detected"


def detect_ollama_details() -> tuple[str, str]:
    version_ok, version_text = safe_run(["ollama", "--version"])
    runtime = version_text if version_ok and version_text else "not detected"

    list_ok, list_text = safe_run(["ollama", "list"])
    if not list_ok or not list_text:
        return runtime, "not detected"

    lines = [line.strip() for line in list_text.splitlines() if line.strip()]
    if len(lines) <= 1:
        return runtime, "not detected"

    model_name = lines[1].split()[0]
    return runtime, model_name


def detect_environment() -> tuple[dict[str, Any], str]:
    import platform

    ram_bytes = detect_memory_bytes()
    runtime_name, model_name = detect_ollama_details()

    environment = {
        "operating_system": platform.platform(),
        "cpu_model": detect_cpu_model(),
        "cpu_cores": "not detected",
        "cpu_threads": os.cpu_count() if os.cpu_count() is not None else "not detected",
        "ram_bytes": ram_bytes if ram_bytes is not None else "not detected",
        "ram_human": bytes_to_gib_string(ram_bytes),
        "python_version": platform.python_version(),
        "streamlit_version": package_version("streamlit"),
        "duckdb_version": package_version("duckdb"),
        "vanna_version": package_version("vanna"),
        "chromadb_version": package_version("chromadb"),
        "local_model_runtime": runtime_name,
        "model_name": model_name,
    }

    markdown = "\n".join(
        [
            "# Evaluation Environment",
            "",
            "- Status: measured",
            f"- Operating system: `{environment['operating_system']}`",
            f"- CPU model: `{environment['cpu_model']}`",
            f"- CPU cores: `{environment['cpu_cores']}`",
            f"- CPU threads: `{environment['cpu_threads']}`",
            f"- RAM: `{environment['ram_human']}`",
            f"- Python version: `{environment['python_version']}`",
            f"- Streamlit version: `{environment['streamlit_version']}`",
            f"- DuckDB version: `{environment['duckdb_version']}`",
            f"- Vanna version: `{environment['vanna_version']}`",
            f"- ChromaDB version: `{environment['chromadb_version']}`",
            f"- Local model runtime: `{environment['local_model_runtime']}`",
            f"- Model name: `{environment['model_name']}`",
        ]
    )
    return environment, markdown


def load_full_training_df() -> pd.DataFrame:
    return vannaagent.load_training_examples()


def lookup_gold_sql(training_df: pd.DataFrame, question: str) -> str:
    gold_sql = vannaagent.get_exact_training_sql(question, training_df)
    if not gold_sql:
        raise RuntimeError(f"Gold SQL missing for question: {question}")
    return gold_sql


def with_vanna_paths(training_csv: Path, chroma_path: Path):
    class PathContext:
        def __enter__(self_inner):
            self_inner.original_training = vannaagent.TRAINING_CSV_PATH
            self_inner.original_chroma = vannaagent.CHROMA_PATH
            vannaagent.TRAINING_CSV_PATH = training_csv
            vannaagent.CHROMA_PATH = str(chroma_path)
            return self_inner

        def __exit__(self_inner, exc_type, exc, tb):
            vannaagent.TRAINING_CSV_PATH = self_inner.original_training
            vannaagent.CHROMA_PATH = self_inner.original_chroma
            return False

    return PathContext()


def prepare_condition_vanna(training_df: pd.DataFrame, training_csv: Path, chroma_path: Path):
    training_csv.parent.mkdir(parents=True, exist_ok=True)
    training_df.to_csv(training_csv, index=False, encoding="utf-8-sig")
    chroma_path.mkdir(parents=True, exist_ok=True)

    with with_vanna_paths(training_csv, chroma_path):
        examples_df = vannaagent.load_training_examples()
        vn = vannaagent.get_vanna()
        vannaagent.train_from_examples(vn, examples_df)
    return examples_df, vn


def run_sql_through_execution_pipeline_local(
    sql: str,
    policy: evaluation_runner.ExecutionPolicy,
) -> tuple[bool, float, str | None, pd.DataFrame]:
    if not sql:
        return False, 0.0, "no_match", pd.DataFrame()

    is_read_only, _ = evaluation_runner.validate_read_only_sql(sql)
    if not is_read_only:
        return False, 0.0, "blocked_read_only", pd.DataFrame()

    complexity_ok, _ = evaluation_runner.validate_sql_complexity(sql, policy)
    if not complexity_ok:
        return False, 0.0, "blocked_complexity", pd.DataFrame()

    compile_error = evaluation_runner.validate_sql_compiles(sql)
    if compile_error:
        return False, 0.0, "compile_fail", pd.DataFrame()

    limited_sql = evaluation_runner.apply_row_limit(sql, policy.max_rows)
    started = time.perf_counter()
    con = duckdb.connect(str(PROJECT_ROOT / evaluation_runner.DUCKDB_PATH), read_only=True)
    try:
        df = con.execute(limited_sql).fetchdf()
        elapsed = time.perf_counter() - started
        return True, float(elapsed), None, df
    except Exception as exc:
        elapsed = time.perf_counter() - started
        category = evaluation_runner.query_logic.classify_execution_failure(str(exc))
        return True, float(elapsed), category, pd.DataFrame()
    finally:
        con.close()


def run_condition(
    name: str,
    questions: list[str],
    gold_sql_by_question: dict[str, str],
    condition_df: pd.DataFrame,
    training_csv: Path,
    chroma_path: Path,
) -> ConditionResult:
    try:
        examples_df, vanna_instance = prepare_condition_vanna(condition_df, training_csv, chroma_path)
        schema_tree = evaluation_runner.get_schema_tree()
        available_years = evaluation_runner.get_available_years()
        policy = evaluation_runner.ExecutionPolicy()
        vanna_cache: dict[str, Any] = {"instance": vanna_instance, "error": None}
    except Exception as exc:
        return ConditionResult(
            name=name,
            exec_acc=0.0,
            exact_match=0.0,
            compile_rate=0.0,
            gen_median=0.0,
            total_median=0.0,
            per_question=[],
            error=str(exc),
        )

    gen_times: list[float] = []
    total_times: list[float] = []
    compiled_total = 0
    exec_correct_total = 0
    exact_total = 0
    per_question: list[dict[str, Any]] = []

    for question in questions:
        gold_sql = gold_sql_by_question[question]
        generated_sql, gen_time, generation_failure = evaluation_runner.generate_sql_app_pipeline(
            question=question,
            schema_tree=schema_tree,
            available_years=available_years,
            training_examples_df=examples_df,
            vanna_cache=vanna_cache,
        )

        compiled = False
        exec_correct = False
        exec_time = 0.0
        failure = generation_failure
        if generated_sql and failure is None:
            compiled, exec_time, exec_failure, generated_df = run_sql_through_execution_pipeline_local(
                generated_sql, policy
            )
            if exec_failure is not None:
                failure = exec_failure
            else:
                gold_compiled, _gold_exec, gold_failure, gold_df = run_sql_through_execution_pipeline_local(
                    gold_sql, policy
                )
                if not gold_compiled or gold_failure is not None:
                    raise RuntimeError(
                        f"Gold SQL failed for question '{question}': {gold_failure or 'compile_fail'}"
                    )
                exec_correct = evaluation_runner.compare_query_results(generated_df, gold_df)
                if not exec_correct:
                    failure = "no_match"

        exact_match = (
            bool(generated_sql)
            and evaluation_runner.normalize_sql_for_exact_match(generated_sql)
            == evaluation_runner.normalize_sql_for_exact_match(gold_sql)
        )

        if compiled:
            compiled_total += 1
        if exec_correct:
            exec_correct_total += 1
        if exact_match:
            exact_total += 1

        total_time = gen_time + exec_time
        gen_times.append(gen_time)
        total_times.append(total_time)
        per_question.append(
            {
                "question": question,
                "generated_sql": generated_sql,
                "gold_sql": gold_sql,
                "compiled": compiled,
                "exec_correct": exec_correct,
                "exact_match": exact_match,
                "gen_time_s": rounded(gen_time),
                "total_time_s": rounded(total_time),
                "failure_category": failure,
            }
        )

    question_count = len(questions)
    return ConditionResult(
        name=name,
        exec_acc=rounded(exec_correct_total / question_count),
        exact_match=rounded(exact_total / question_count),
        compile_rate=rounded(compiled_total / question_count),
        gen_median=rounded(statistics.median(gen_times) if gen_times else 0.0),
        total_median=rounded(statistics.median(total_times) if total_times else 0.0),
        per_question=per_question,
        error=None,
    )


def build_training_impact() -> tuple[dict[str, Any], str]:
    training_df = load_full_training_df()
    gold_sql_by_question = {
        question: lookup_gold_sql(training_df, question) for question in TRAINING_IMPACT_SUBSET
    }

    removed_examples: list[dict[str, str]] = []
    keep_mask = []
    for _, row in training_df.iterrows():
        question = str(row["question"]).strip()
        sql = str(row["sql"]).strip()
        should_keep = question not in TRAINING_IMPACT_SUBSET
        keep_mask.append(should_keep)
        if not should_keep:
            removed_examples.append({"question": question, "sql": sql})

    before_df = training_df.loc[keep_mask].reset_index(drop=True)
    after_df = training_df.copy()

    temp_root = OUTPUT_DIR / "_tmp" / datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    if temp_root.exists():
        shutil.rmtree(temp_root, ignore_errors=True)
    temp_root.mkdir(parents=True, exist_ok=True)

    before_result = run_condition(
        name="before",
        questions=TRAINING_IMPACT_SUBSET,
        gold_sql_by_question=gold_sql_by_question,
        condition_df=before_df,
        training_csv=temp_root / "before" / "training_examples.csv",
        chroma_path=temp_root / "before" / "chroma_store",
    )
    after_result = run_condition(
        name="after",
        questions=TRAINING_IMPACT_SUBSET,
        gold_sql_by_question=gold_sql_by_question,
        condition_df=after_df,
        training_csv=temp_root / "after" / "training_examples.csv",
        chroma_path=temp_root / "after" / "chroma_store",
    )

    report = {
        "question_subset": TRAINING_IMPACT_SUBSET,
        "before": {
            "exec_acc": before_result.exec_acc,
            "exact_match": before_result.exact_match,
            "compile_rate": before_result.compile_rate,
            "gen_median": before_result.gen_median,
            "total_median": before_result.total_median,
            "error": before_result.error,
            "per_question": before_result.per_question,
        },
        "after": {
            "exec_acc": after_result.exec_acc,
            "exact_match": after_result.exact_match,
            "compile_rate": after_result.compile_rate,
            "gen_median": after_result.gen_median,
            "total_median": after_result.total_median,
            "error": after_result.error,
            "per_question": after_result.per_question,
        },
        "training_change_description": (
            "Before condition used a temporary copy of training_examples.csv with the subset-specific "
            "rows removed and retrained an isolated temporary Chroma store. After condition used the "
            "full current training_examples.csv and retrained a separate isolated temporary Chroma store. "
            f"Temporary experiment artifacts were written under {temp_root}."
        ),
        "removed_or_disabled_examples": removed_examples,
    }

    if before_result.error or after_result.error:
        status = "partially measured"
    else:
        status = "measured"

    markdown_lines = [
        "# Training Impact Report",
        "",
        f"- Status: {status}",
        f"- Question subset size: `{len(TRAINING_IMPACT_SUBSET)}`",
        "- Training change: before removed the subset rows from a temporary CSV and isolated Chroma store; after used the full current CSV in a separate isolated Chroma store.",
        f"- Removed/disabled example count: `{len(removed_examples)}`",
        "",
        "## Before",
        f"- ExecAcc: `{report['before']['exec_acc']}`",
        f"- Exact Match: `{report['before']['exact_match']}`",
        f"- Compile Rate: `{report['before']['compile_rate']}`",
        f"- Generation latency median (s): `{report['before']['gen_median']}`",
        f"- Total latency median (s): `{report['before']['total_median']}`",
        f"- Error: `{report['before']['error'] or 'none'}`",
        "",
        "## After",
        f"- ExecAcc: `{report['after']['exec_acc']}`",
        f"- Exact Match: `{report['after']['exact_match']}`",
        f"- Compile Rate: `{report['after']['compile_rate']}`",
        f"- Generation latency median (s): `{report['after']['gen_median']}`",
        f"- Total latency median (s): `{report['after']['total_median']}`",
        f"- Error: `{report['after']['error'] or 'none'}`",
        "",
        "## Removed Or Disabled Examples",
    ]
    for example in removed_examples:
        markdown_lines.append(f"- `{example['question']}`")

    return report, "\n".join(markdown_lines)


def classify_semantic_error(question: str, generated_sql: str) -> str:
    q = question.lower()
    sql = (generated_sql or "").lower()

    if "join" in q and " join " not in sql:
        return "wrong join"
    if "active" in q and "status = 'aktiv'" not in sql:
        return "missing filter"
    if "top 3" in q or "top 5" in q:
        if "row_number()" not in sql and "limit 3" not in sql and "limit 5" not in sql:
            return "wrong aggregation"
    if "ticket type" in q and "ticket_name" not in sql:
        return "wrong column"
    if "tariff association" in q and "tarifverbund" not in sql:
        return "wrong column"
    if "month" in q and "monat" not in sql:
        return "wrong column"
    return "wrong aggregation"


def explain_semantic_error(question: str, generated_sql: str, gold_sql: str) -> str:
    q = question.lower()
    sql = (generated_sql or "").lower()

    if "active" in q and "status = 'aktiv'" not in sql:
        return "The generated SQL ignored the active-only constraint."
    if "ticket type" in q and "ticket_name" not in sql:
        return "The generated SQL dropped the ticket-type dimension from the grouping."
    if "state" in q and "bundesland_name" not in sql:
        return "The generated SQL did not keep the state dimension required by the question."
    if "tariff association" in q and "tarifverbund" not in sql:
        return "The generated SQL omitted the tariff association dimension."
    if "row_number()" in gold_sql.lower() and "row_number()" not in sql:
        return "The generated SQL missed the per-group ranking logic required by the question."
    if "status = 'aktiv'" in gold_sql.lower() and "status = 'aktiv'" not in sql:
        return "The generated SQL omitted a filter that is necessary for the intended semantics."
    return "The generated SQL compiled and ran, but it returned a different grouping or filter than the gold query."


def build_semantic_error_examples() -> tuple[dict[str, Any], str]:
    schema_tree = evaluation_runner.get_schema_tree()
    available_years = evaluation_runner.get_available_years()
    policy = evaluation_runner.ExecutionPolicy()
    training_df = load_full_training_df()
    vanna_cache: dict[str, Any] = {"instance": None, "error": None}

    tested_cases: list[dict[str, Any]] = []
    semantic_errors: list[dict[str, Any]] = []

    for case in SEMANTIC_EVAL_CASES:
        question = case["question"]
        gold_sql = case["gold_sql"]
        generated_sql, gen_time, generation_failure = evaluation_runner.generate_sql_app_pipeline(
            question=question,
            schema_tree=schema_tree,
            available_years=available_years,
            training_examples_df=training_df,
            vanna_cache=vanna_cache,
        )

        generated_compiled = False
        generated_exec_error = generation_failure
        gold_exec_error = None
        result_matches = False

        if generated_sql and generation_failure is None:
            generated_compiled, _gen_exec, generated_exec_error, generated_df = run_sql_through_execution_pipeline_local(
                generated_sql, policy
            )
            gold_compiled, _gold_exec, gold_exec_error, gold_df = run_sql_through_execution_pipeline_local(
                gold_sql, policy
            )
            if gold_compiled and gold_exec_error is None and generated_compiled and generated_exec_error is None:
                result_matches = evaluation_runner.compare_query_results(generated_df, gold_df)

        tested_case = {
            "question": question,
            "generated_sql": generated_sql,
            "corrected_sql": gold_sql,
            "generation_failure": generation_failure,
            "generated_exec_error": generated_exec_error,
            "gold_exec_error": gold_exec_error,
            "compiled_and_ran": generated_compiled and generated_exec_error is None,
            "result_matches_gold": result_matches,
            "gen_time_s": rounded(gen_time),
        }
        tested_cases.append(tested_case)

        if (
            generated_sql
            and generated_compiled
            and generated_exec_error is None
            and gold_exec_error is None
            and not result_matches
            and len(semantic_errors) < 3
        ):
            semantic_errors.append(
                {
                    "question": question,
                    "generated_sql": generated_sql,
                    "corrected_sql": gold_sql,
                    "what_went_wrong": explain_semantic_error(question, generated_sql, gold_sql),
                    "category": classify_semantic_error(question, generated_sql),
                }
            )

    if semantic_errors:
        status = "measured"
        summary = f"Found `{len(semantic_errors)}` semantic error example(s) in `{len(SEMANTIC_EVAL_CASES)}` tested paraphrases."
    else:
        status = "measured"
        summary = "No semantic errors observed in the tested set."

    report = {
        "tested_questions": [case["question"] for case in SEMANTIC_EVAL_CASES],
        "tested_case_count": len(SEMANTIC_EVAL_CASES),
        "semantic_error_examples": semantic_errors,
        "summary": summary,
        "tested_cases": tested_cases,
    }

    markdown_lines = [
        "# Semantic Error Examples",
        "",
        f"- Status: {status}",
        f"- Tested harder in-domain paraphrases: `{len(SEMANTIC_EVAL_CASES)}`",
        f"- Summary: {summary}",
    ]
    if semantic_errors:
        markdown_lines.append("")
        for index, error in enumerate(semantic_errors, start=1):
            markdown_lines.extend(
                [
                    f"## Error {index}",
                    f"- Question: `{error['question']}`",
                    f"- Category: `{error['category']}`",
                    f"- Explanation: {error['what_went_wrong']}",
                    "- Generated SQL:",
                    "```sql",
                    error["generated_sql"],
                    "```",
                    "- Corrected SQL:",
                    "```sql",
                    error["corrected_sql"],
                    "```",
                ]
            )
    return report, "\n".join(markdown_lines)


def detect_benchmark_assets() -> list[str]:
    matches: list[str] = []
    skip_roots = {".git", ".venv", "__pycache__", "outputs", ".pytest_cache"}
    for path in PROJECT_ROOT.rglob("*"):
        rel_parts = path.relative_to(PROJECT_ROOT).parts
        if any(part in skip_roots for part in rel_parts):
            continue
        lowered_parts = [part.lower() for part in rel_parts]
        lowered_name = path.name.lower()
        is_benchmark_dir = any(part in {"spider", "bird"} for part in lowered_parts)
        is_benchmark_file = lowered_name in {
            "tables.json",
            "train_spider.json",
            "train_others.json",
            "dev.json",
        } or lowered_name.endswith(".sqlite") or lowered_name.endswith(".db")
        if is_benchmark_dir or is_benchmark_file:
            matches.append(str(path.relative_to(PROJECT_ROOT)))
    return sorted(set(matches))


def build_benchmark_check() -> tuple[dict[str, Any], str]:
    if LOCAL_BENCHMARK_CSV_PATH.exists():
        cases = run_benchmark.load_benchmark_cases(LOCAL_BENCHMARK_CSV_PATH)
        schema_tree = run_benchmark.get_schema_tree()
        policy = run_benchmark.ExecutionPolicy()
        results = [
            run_benchmark.run_single_question(
                case=case,
                schema_tree=schema_tree,
                policy=policy,
                use_llm=False,
                vanna_instance=None,
            )
            for case in cases
        ]
        summary = run_benchmark.build_summary(results)
        report = {
            "available": True,
            "reason": "local held-out benchmark CSV found",
            "benchmark_type": "local held-out in-domain benchmark",
            "source": str(LOCAL_BENCHMARK_CSV_PATH.relative_to(PROJECT_ROOT)),
            "question_count": summary["total_questions"],
            "metrics": {
                "exact_match": summary["gold_metrics"]["exact_match_rate"],
                "exec_acc": summary["gold_metrics"]["exec_acc"],
                "compile_rate": summary["gold_metrics"]["compile_rate"],
                "success_rate_percent": summary["rates_percent"]["success_rate"],
                "safe_fail_rate_percent": summary["rates_percent"]["safe_fail_rate"],
            },
            "latency_ms": summary["latency_ms"],
        }
        markdown = "\n".join(
            [
                "# Benchmark Availability",
                "",
                "- Status: measured",
                "- Benchmark type: local held-out in-domain benchmark",
                f"- Source: `{report['source']}`",
                f"- Questions: `{report['question_count']}`",
                f"- Exact Match: `{report['metrics']['exact_match']}`",
                f"- ExecAcc: `{report['metrics']['exec_acc']}`",
                f"- Compile Rate: `{report['metrics']['compile_rate']}`",
                f"- Success rate (%): `{report['metrics']['success_rate_percent']}`",
                f"- Safe-fail rate (%): `{report['metrics']['safe_fail_rate_percent']}`",
                "- Note: this is a local held-out benchmark for the thesis project, not Spider/BIRD cross-domain evaluation.",
            ]
        )
        return report, markdown

    matches = detect_benchmark_assets()
    benchmark_like = [match for match in matches if "spider" in match.lower() or "bird" in match.lower()]
    if benchmark_like:
        report = {
            "available": True,
            "reason": "benchmark-like local assets found",
            "matches": benchmark_like,
            "metrics": "not measured automatically",
        }
        markdown = "\n".join(
            [
                "# Benchmark Availability",
                "",
                "- Status: partially measured",
                "- Benchmark-like local assets were found, but no honest small cross-domain run was executed automatically from these paths.",
            ]
            + [f"- Match: `{match}`" for match in benchmark_like]
        )
        return report, markdown

    report = {
        "available": False,
        "reason": "no local benchmark dataset found",
        "matches": [],
    }
    markdown = "\n".join(
        [
            "# Benchmark Availability",
            "",
            "- Status: not measurable automatically",
            "- Available: `false`",
            "- Reason: `no local benchmark dataset found`",
        ]
    )
    return report, markdown


def build_usability_check() -> tuple[dict[str, Any], str]:
    if not METRICS_LOG_PATH.exists():
        report = {
            "status": "not measurable automatically",
            "reason": "no telemetry log found",
            "number_of_sessions": "not measurable automatically",
            "successful_question_run_flows": "not measurable automatically",
            "feedback_signals": "not measurable automatically",
        }
        markdown = "\n".join(
            [
                "# Usability Measurability",
                "",
                "- Status: not measurable automatically",
                "- Reason: no telemetry log found.",
                "- Formal usability outcomes require human participants.",
            ]
        )
        return report, markdown

    events = load_metric_events(METRICS_LOG_PATH)
    summary = summarize_metric_events(events)
    successful_runs = summary["execution"]["success"]
    feedback = summary["feedback"]

    report = {
        "status": "partially measured",
        "reason": "telemetry logs capture events but do not include a reliable session identifier",
        "number_of_sessions": "not measurable automatically",
        "successful_question_run_flows": successful_runs,
        "feedback_signals": {
            "up": feedback["up"],
            "down": feedback["down"],
        },
        "event_totals": summary["totals"],
    }
    markdown = "\n".join(
        [
            "# Usability Measurability",
            "",
            "- Status: partially measured",
            "- Number of sessions: `not measurable automatically`",
            f"- Successful question -> run flows: `{successful_runs}`",
            f"- Feedback up: `{feedback['up']}`",
            f"- Feedback down: `{feedback['down']}`",
            "- Formal usability outcomes still require human participants because the logs do not include participant/session-level identifiers or task-success judgments.",
        ]
    )
    return report, markdown


def build_combined_summary(
    environment: dict[str, Any],
    training_impact: dict[str, Any],
    semantic_errors: dict[str, Any],
    benchmark_check: dict[str, Any],
    usability_check: dict[str, Any],
) -> tuple[dict[str, Any], str]:
    combined = {
        "evaluation_environment": {
            "status": "measured",
            "data": environment,
        },
        "before_after_training_impact": {
            "status": "partially measured"
            if training_impact["before"]["error"] or training_impact["after"]["error"]
            else "measured",
            "data": training_impact,
        },
        "semantic_error_examples": {
            "status": "measured",
            "data": semantic_errors,
        },
        "benchmark_availability_results": {
            "status": "measured" if benchmark_check["available"] else "not measurable automatically",
            "data": benchmark_check,
        },
        "usability_measurability": {
            "status": usability_check["status"],
            "data": usability_check,
        },
    }

    semantic_summary = semantic_errors["summary"]
    summary_lines = [
        "# Chapter 6 Missing Evaluation Summary",
        "",
        "## 1. Evaluation environment",
        "- Status: measured",
        f"- Operating system: `{environment['operating_system']}`",
        f"- CPU model: `{environment['cpu_model']}`",
        f"- RAM: `{environment['ram_human']}`",
        f"- Python / Streamlit / DuckDB / Vanna / ChromaDB: `{environment['python_version']}` / `{environment['streamlit_version']}` / `{environment['duckdb_version']}` / `{environment['vanna_version']}` / `{environment['chromadb_version']}`",
        f"- Local runtime / model: `{environment['local_model_runtime']}` / `{environment['model_name']}`",
        "",
        "## 2. Before/after training impact",
        f"- Status: {combined['before_after_training_impact']['status']}",
        f"- Before ExecAcc / EM / Compile: `{training_impact['before']['exec_acc']}` / `{training_impact['before']['exact_match']}` / `{training_impact['before']['compile_rate']}`",
        f"- After ExecAcc / EM / Compile: `{training_impact['after']['exec_acc']}` / `{training_impact['after']['exact_match']}` / `{training_impact['after']['compile_rate']}`",
        f"- Before generation median / total median (s): `{training_impact['before']['gen_median']}` / `{training_impact['before']['total_median']}`",
        f"- After generation median / total median (s): `{training_impact['after']['gen_median']}` / `{training_impact['after']['total_median']}`",
        f"- Error state: before=`{training_impact['before']['error'] or 'none'}` after=`{training_impact['after']['error'] or 'none'}`",
        "",
        "## 3. Semantic error examples",
        "- Status: measured",
        f"- {semantic_summary}",
        "",
        "## 4. Benchmark availability / results",
        f"- Status: {combined['benchmark_availability_results']['status']}",
        f"- Available: `{benchmark_check['available']}`",
        f"- Reason: `{benchmark_check['reason']}`",
        "",
        "## 5. Usability measurability",
        f"- Status: {usability_check['status']}",
        f"- Number of sessions: `{usability_check['number_of_sessions']}`",
        f"- Successful question -> run flows: `{usability_check['successful_question_run_flows']}`",
        "- Formal usability outcomes require human participants unless participant/session instrumentation is added.",
    ]
    return combined, "\n".join(summary_lines)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    environment, environment_md = detect_environment()
    training_impact, training_impact_md = build_training_impact()
    semantic_errors, semantic_errors_md = build_semantic_error_examples()
    benchmark_check, benchmark_check_md = build_benchmark_check()
    usability_check, usability_check_md = build_usability_check()
    combined_report, combined_summary_md = build_combined_summary(
        environment=environment,
        training_impact=training_impact,
        semantic_errors=semantic_errors,
        benchmark_check=benchmark_check,
        usability_check=usability_check,
    )

    write_json(OUTPUT_DIR / "eval_environment.json", environment)
    write_markdown(OUTPUT_DIR / "eval_environment.md", environment_md)

    write_json(OUTPUT_DIR / "training_impact_report.json", training_impact)
    write_markdown(OUTPUT_DIR / "training_impact_report.md", training_impact_md)

    write_json(OUTPUT_DIR / "semantic_error_examples.json", semantic_errors)
    write_markdown(OUTPUT_DIR / "semantic_error_examples.md", semantic_errors_md)

    write_json(OUTPUT_DIR / "benchmark_check.json", benchmark_check)
    write_markdown(OUTPUT_DIR / "benchmark_check.md", benchmark_check_md)

    write_json(OUTPUT_DIR / "usability_check.json", usability_check)
    write_markdown(OUTPUT_DIR / "usability_check.md", usability_check_md)

    write_json(OUTPUT_DIR / "chapter6_missing_eval_report.json", combined_report)
    write_markdown(OUTPUT_DIR / "chapter6_missing_eval_summary.md", combined_summary_md)

    print(
        json.dumps(
            {
                "output_dir": str(OUTPUT_DIR),
                "environment_status": "measured",
                "training_before_error": training_impact["before"]["error"],
                "training_after_error": training_impact["after"]["error"],
                "semantic_error_count": len(semantic_errors["semantic_error_examples"]),
                "benchmark_available": benchmark_check["available"],
                "usability_status": usability_check["status"],
            },
            ensure_ascii=True,
        )
    )


if __name__ == "__main__":
    main()
