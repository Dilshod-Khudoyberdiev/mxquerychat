"""

Purpose:
This module provides centralized logging and structured metric-event writing for mxQueryChat. It keeps
operational observability simple, local, and consistent.

What This File Contains:
- Logger configuration with console and file handlers.
- Path-resolution helpers for app log and metrics JSONL locations.
- A metric-event writer that appends normalized JSON payloads with UTC timestamps.

Key Invariants and Safety Guarantees:
- Logging setup is idempotent to avoid duplicate handlers.
- Metric events are written as one JSON object per line for easy downstream processing.
- Failures during metrics write are logged as warnings without crashing the app flow.

How Other Modules Use This File:
app.py and test utilities import this module to record generation, execution, explanation, and user
feedback events. tools/summarize_metrics.py later consumes these events for KPI reporting.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _build_paths() -> tuple[Path, Path]:
    log_dir = Path(os.getenv("APP_LOG_DIR", "logs"))
    app_log_path = Path(os.getenv("APP_LOG_PATH", str(log_dir / "app.log")))
    metrics_log_path = Path(
        os.getenv("APP_METRICS_LOG_PATH", str(log_dir / "metrics.jsonl"))
    )
    return app_log_path, metrics_log_path


def configure_app_logging() -> logging.Logger:
    """Configure app logger with console + file handlers (idempotent)."""
    logger = logging.getLogger("mxquerychat")
    if logger.handlers:
        return logger

    app_log_path, _ = _build_paths()
    app_log_path.parent.mkdir(parents=True, exist_ok=True)

    log_level = os.getenv("APP_LOG_LEVEL", "INFO").upper()
    logger.setLevel(log_level)
    logger.propagate = False

    formatter = logging.Formatter(
        "%(asctime)s %(levelname)s %(name)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    file_handler = logging.FileHandler(app_log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    logger.info("Logging initialized.")
    return logger


def record_metric_event(event: str, **fields: Any) -> None:
    """
    Write one structured metric event to logs/metrics.jsonl and app logs.
    For user feedback events, pass:
      - question_hash: short stable hash
      - rating: "up" or "down"
      - has_result: bool
    For failure events, pass:
      - failure_category: blocked_read_only | blocked_complexity | timeout | compile_fail | runtime_fail | no_match
    """
    logger = configure_app_logging()
    _, metrics_log_path = _build_paths()
    metrics_log_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "event": event,
        **fields,
    }
    try:
        with metrics_log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=True) + "\n")
        logger.info("metric=%s", json.dumps(payload, ensure_ascii=True))
    except Exception as exc:
        logger.warning("Could not write metric event '%s': %s", event, exc)


