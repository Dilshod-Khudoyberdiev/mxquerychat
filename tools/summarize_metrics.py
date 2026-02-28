"""Summarize mxquerychat metrics JSONL into PRD-facing KPI aggregates."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.query_logic import classify_execution_failure


def load_metric_events(metrics_path: Path) -> list[dict]:
    """Read JSONL metrics log and return parsed events."""
    if not metrics_path.exists():
        return []

    events: list[dict] = []
    for line in metrics_path.read_text(encoding="utf-8", errors="replace").splitlines():
        text = line.strip()
        if not text:
            continue
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            events.append(payload)
    return events


def _safe_percent(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return round((numerator / denominator) * 100.0, 2)


def _median(values: list[float]) -> float:
    if not values:
        return 0.0
    return round(float(statistics.median(values)), 2)


def summarize_metric_events(events: list[dict]) -> dict:
    """Build summary structure for generation, execution, and feedback."""
    generation_events = [e for e in events if e.get("event") == "sql_generation"]
    execution_events = [e for e in events if e.get("event") == "query_execution"]
    feedback_events = [e for e in events if e.get("event") == "user_feedback"]

    generation_success = sum(1 for e in generation_events if bool(e.get("success")))
    generation_failed = len(generation_events) - generation_success
    generation_durations = [
        float(e.get("duration_ms", 0))
        for e in generation_events
        if isinstance(e.get("duration_ms"), (int, float))
    ]

    execution_success = sum(1 for e in execution_events if bool(e.get("success")))
    execution_failed = len(execution_events) - execution_success
    execution_durations = [
        float(e.get("execution_ms", 0))
        for e in execution_events
        if isinstance(e.get("execution_ms"), (int, float)) and float(e.get("execution_ms", 0)) > 0
    ]

    failure_breakdown: dict[str, int] = {}
    for event in execution_events:
        if bool(event.get("success")):
            continue
        category = str(event.get("failure_category") or "").strip()
        if not category:
            category = classify_execution_failure(str(event.get("failure_reason", "")))
        failure_breakdown[category] = failure_breakdown.get(category, 0) + 1

    feedback_up = sum(
        1 for event in feedback_events if str(event.get("rating", "")).lower() == "up"
    )
    feedback_down = sum(
        1 for event in feedback_events if str(event.get("rating", "")).lower() == "down"
    )

    return {
        "totals": {
            "events": len(events),
            "generation_events": len(generation_events),
            "execution_events": len(execution_events),
            "feedback_events": len(feedback_events),
        },
        "generation": {
            "success": generation_success,
            "failed": generation_failed,
            "success_rate": _safe_percent(generation_success, len(generation_events)),
            "fail_rate": _safe_percent(generation_failed, len(generation_events)),
            "median_generation_ms": _median(generation_durations),
        },
        "execution": {
            "success": execution_success,
            "failed": execution_failed,
            "success_rate": _safe_percent(execution_success, len(execution_events)),
            "fail_rate": _safe_percent(execution_failed, len(execution_events)),
            "median_execution_ms": _median(execution_durations),
            "failure_breakdown": failure_breakdown,
        },
        "feedback": {
            "up": feedback_up,
            "down": feedback_down,
        },
        "prd_kpis": {
            "generation_success_rate": _safe_percent(
                generation_success, len(generation_events)
            ),
            "execution_fail_breakdown": failure_breakdown,
            "median_generation_ms": _median(generation_durations),
            "median_execution_ms": _median(execution_durations),
            "feedback_up": feedback_up,
            "feedback_down": feedback_down,
        },
    }


def write_summary(summary: dict, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2, ensure_ascii=True), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize metrics JSONL into KPI JSON.")
    parser.add_argument(
        "--metrics-path",
        default="logs/metrics.jsonl",
        help="Path to metrics JSONL file.",
    )
    parser.add_argument(
        "--output",
        default="reports/metrics_summary.json",
        help="Output JSON path.",
    )
    args = parser.parse_args()

    metrics_path = Path(args.metrics_path)
    events = load_metric_events(metrics_path)
    summary = summarize_metric_events(events)
    output_path = Path(args.output)
    write_summary(summary, output_path)

    print(f"Loaded events: {summary['totals']['events']}")
    print(f"Generation success rate: {summary['generation']['success_rate']}%")
    print(f"Execution fail rate: {summary['execution']['fail_rate']}%")
    print(f"Metrics summary: {output_path}")


if __name__ == "__main__":
    main()
