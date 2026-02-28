from pathlib import Path

from tools.summarize_metrics import (
    load_metric_events,
    summarize_metric_events,
    write_summary,
)


def test_summarize_metric_events_schema_and_kpis() -> None:
    events = [
        {"event": "sql_generation", "success": True, "duration_ms": 100},
        {"event": "sql_generation", "success": False, "duration_ms": 200},
        {
            "event": "query_execution",
            "success": False,
            "failure_category": "blocked_read_only",
            "failure_reason": "Blocked",
        },
        {"event": "query_execution", "success": True, "execution_ms": 30},
        {"event": "user_feedback", "rating": "up", "has_result": True},
        {"event": "user_feedback", "rating": "down", "has_result": True},
    ]

    summary = summarize_metric_events(events)

    assert "totals" in summary
    assert "generation" in summary
    assert "execution" in summary
    assert "feedback" in summary
    assert "prd_kpis" in summary
    assert summary["generation"]["success"] == 1
    assert summary["generation"]["failed"] == 1
    assert summary["execution"]["failure_breakdown"]["blocked_read_only"] == 1
    assert summary["feedback"]["up"] == 1
    assert summary["feedback"]["down"] == 1
    assert isinstance(summary["prd_kpis"]["median_generation_ms"], float)


def test_load_and_write_summary_roundtrip(tmp_path: Path) -> None:
    metrics_path = tmp_path / "metrics.jsonl"
    metrics_path.write_text(
        "\n".join(
            [
                '{"event":"sql_generation","success":true,"duration_ms":10}',
                '{"event":"query_execution","success":true,"execution_ms":25}',
            ]
        ),
        encoding="utf-8",
    )

    events = load_metric_events(metrics_path)
    assert len(events) == 2

    summary = summarize_metric_events(events)
    output_path = tmp_path / "summary.json"
    write_summary(summary, output_path)

    assert output_path.exists()
    written = output_path.read_text(encoding="utf-8")
    assert "generation" in written
    assert "execution" in written
