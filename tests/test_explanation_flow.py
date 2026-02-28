import json

from src.core.query_logic import run_query_if_read_only
from src.llm.sql_explainer import build_explanation_cache_key, maybe_generate_explanation
from src.utils.telemetry import record_metric_event
from sql_guard import validate_read_only_sql


def test_on_demand_explanation_does_not_call_model_without_trigger() -> None:
    called = {"count": 0}

    def _fake_generate(**kwargs):
        called["count"] += 1
        return "text", ""

    text, status, cache_hit = maybe_generate_explanation(
        triggered=False,
        question="Q",
        sql="SELECT 1",
        cache={},
        model="mistral",
        ollama_url="http://127.0.0.1:11434",
        timeout_seconds=8,
        generate_fn=_fake_generate,
    )
    assert text == ""
    assert status == "idle"
    assert cache_hit is False
    assert called["count"] == 0


def test_explanation_cache_hit_avoids_model_call() -> None:
    called = {"count": 0}

    def _fake_generate(**kwargs):
        called["count"] += 1
        return "generated", ""

    cache = {build_explanation_cache_key("Q", "SELECT 1"): "cached"}
    text, status, cache_hit = maybe_generate_explanation(
        triggered=True,
        question="Q",
        sql="SELECT 1",
        cache=cache,
        model="mistral",
        ollama_url="http://127.0.0.1:11434",
        timeout_seconds=8,
        generate_fn=_fake_generate,
    )
    assert text == "cached"
    assert status == "ready"
    assert cache_hit is True
    assert called["count"] == 0


def test_explanation_failure_does_not_block_sql_execution() -> None:
    def _fake_generate(**kwargs):
        return "", "timeout"

    text, status, cache_hit = maybe_generate_explanation(
        triggered=True,
        question="Q",
        sql="SELECT 1",
        cache={},
        model="mistral",
        ollama_url="http://127.0.0.1:11434",
        timeout_seconds=8,
        generate_fn=_fake_generate,
    )
    assert text == ""
    assert status == "timeout"
    assert cache_hit is False

    called = {"count": 0}

    def _run(sql: str):
        called["count"] += 1
        return {"ok": sql == "SELECT 1"}

    is_allowed, result, message = run_query_if_read_only("SELECT 1", validate_read_only_sql, _run)
    assert is_allowed is True
    assert result == {"ok": True}
    assert message == "OK"
    assert called["count"] == 1


def test_sql_explanation_telemetry_includes_failure_category(tmp_path, monkeypatch) -> None:
    metrics_path = tmp_path / "metrics.jsonl"
    monkeypatch.setenv("APP_METRICS_LOG_PATH", str(metrics_path))

    record_metric_event(
        "sql_explanation",
        success=False,
        path="on_demand_ollama",
        failure_category="timeout",
        duration_ms=123,
    )

    lines = metrics_path.read_text(encoding="utf-8").strip().splitlines()
    payload = json.loads(lines[-1])
    assert payload["event"] == "sql_explanation"
    assert payload["failure_category"] == "timeout"
    assert payload["path"] == "on_demand_ollama"
