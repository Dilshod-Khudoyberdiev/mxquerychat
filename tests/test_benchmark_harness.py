from pathlib import Path

from tools.run_benchmark import build_summary, parse_questions_from_markdown


def test_parse_questions_from_markdown_supports_en_and_q(tmp_path: Path) -> None:
    file_path = tmp_path / "questions.md"
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
    questions = parse_questions_from_markdown(file_path)
    assert questions == ["Show revenue by month.", "Show top 10 states."]


def test_build_summary_calculates_counts_and_rates() -> None:
    results = [
        {"outcome": "success", "generation_ms": 10, "execution_ms": 20},
        {"outcome": "compile_fail", "generation_ms": 5, "execution_ms": 0},
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
