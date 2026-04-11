"""Tests for parsing and summary math helpers in tools/run_benchmark.py."""

import shutil
import uuid
from pathlib import Path

from tools.run_benchmark import (
    build_summary,
    load_benchmark_cases,
    parse_questions_from_markdown,
)


def _make_local_temp_dir() -> Path:
    path = Path("outputs") / "test_tmp" / f"benchmark_{uuid.uuid4().hex}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def test_parse_questions_from_markdown_supports_en_and_q() -> None:
    temp_dir = _make_local_temp_dir()
    file_path = temp_dir / "questions.md"
    file_path.write_text(
        "\n".join(
            [
                "1. EN: Show revenue by month.",
                "2. DE: Ignored line.",
                "Q: Show top 10 states.",
            ]
        ),
        encoding="utf-8",
    )
    try:
        questions = parse_questions_from_markdown(file_path)
        assert questions == ["Show revenue by month.", "Show top 10 states."]
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_build_summary_calculates_counts_and_rates() -> None:
    results = [
        {
            "outcome": "success",
            "generation_ms": 10,
            "execution_ms": 20,
            "compiled": True,
            "exact_match": True,
            "exec_correct": True,
            "gold_sql": "SELECT 1",
        },
        {
            "outcome": "compile_fail",
            "generation_ms": 5,
            "execution_ms": 0,
            "compiled": False,
            "exact_match": False,
            "exec_correct": False,
            "gold_sql": "SELECT 2",
        },
        {"outcome": "safe_fail", "generation_ms": 7, "execution_ms": 0},
        {"outcome": "runtime_fail", "generation_ms": 8, "execution_ms": 12},
    ]
    summary = build_summary(results)
    assert summary["total_questions"] == 4
    assert summary["counts"]["success"] == 1
    assert summary["counts"]["compile_fail"] == 1
    assert summary["counts"]["safe_fail"] == 1
    assert summary["counts"]["runtime_fail"] == 1
    assert summary["rates_percent"]["success_rate"] == 25.0
    assert "median_generation_ms" in summary["latency_ms"]
    assert "median_execution_ms" in summary["latency_ms"]
    assert "prd_kpis" in summary
    assert "compile_rate" in summary["prd_kpis"]
    assert "safe_fail_rate" in summary["prd_kpis"]
    assert summary["gold_metrics"]["gold_question_count"] == 2
    assert summary["gold_metrics"]["exact_match_rate"] == 0.5
    assert summary["gold_metrics"]["exec_acc"] == 0.5


def test_load_benchmark_cases_from_csv() -> None:
    temp_dir = _make_local_temp_dir()
    csv_path = temp_dir / "benchmark.csv"
    csv_path.write_text(
        "\n".join(
            [
                "question,gold_sql,difficulty,category",
                '"Show revenue by month.","SELECT 1",easy,time',
                '"Show top states.","SELECT 2",hard,ranking',
            ]
        ),
        encoding="utf-8",
    )
    try:
        cases = load_benchmark_cases(csv_path)
        assert len(cases) == 2
        assert cases[0]["question"] == "Show revenue by month."
        assert cases[0]["gold_sql"] == "SELECT 1"
        assert cases[1]["difficulty"] == "hard"
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


