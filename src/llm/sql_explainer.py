"""On-demand SQL explanation helpers using local Ollama."""

from __future__ import annotations

import hashlib
import json
import os
import socket
from typing import Callable
from urllib import error, request


def _read_int_env(name: str, default: int) -> int:
    raw = os.getenv(name, str(default))
    try:
        return int(raw)
    except ValueError:
        return default


def build_explanation_prompt(question: str, sql: str) -> str:
    """Build a concise prompt for local LLM SQL explanation."""
    question_text = (question or "").strip()
    sql_text = " ".join((sql or "").strip().split())
    if len(sql_text) > 700:
        sql_text = sql_text[:700].rstrip() + " ..."
    return (
        "Explain SQL intent for a business user.\n"
        "Return exactly one sentence, max 25 words.\n"
        "Mention metric, main filter, and grouping when present.\n"
        "No markdown.\n\n"
        f"Question:\n{question_text}\n\n"
        f"SQL:\n{sql_text}\n"
    )


def build_explanation_cache_key(question: str, sql: str) -> str:
    """Stable cache key from question + SQL."""
    joined = f"{(question or '').strip()}||{(sql or '').strip()}"
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()


def _extract_response_text(payload: dict) -> str:
    """Extract textual model response from Ollama payload variants."""
    if not isinstance(payload, dict):
        return ""
    if isinstance(payload.get("response"), str):
        return payload["response"].strip()
    message = payload.get("message")
    if isinstance(message, dict) and isinstance(message.get("content"), str):
        return message["content"].strip()
    return ""


def generate_sql_explanation(
    model: str,
    prompt: str,
    ollama_url: str,
    timeout_seconds: int,
) -> tuple[str, str]:
    """
    Generate explanation text via local Ollama.
    Returns (explanation_text, error_code):
      - error_code: "", "timeout", "model_error", "empty_response"
    """
    endpoint = (ollama_url or "").rstrip("/") + "/api/generate"
    body = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "keep_alive": os.getenv("OLLAMA_KEEP_ALIVE", "30m"),
        "options": {
            "temperature": 0,
            "num_predict": _read_int_env("EXPLANATION_NUM_PREDICT", 32),
        },
    }
    data = json.dumps(body).encode("utf-8")
    req = request.Request(
        endpoint,
        data=data,
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    try:
        with request.urlopen(req, timeout=timeout_seconds) as response:
            raw = response.read().decode("utf-8", errors="replace")
            parsed = json.loads(raw)
            text = _extract_response_text(parsed)
            if not text:
                return "", "empty_response"
            return text, ""
    except (TimeoutError, socket.timeout):
        return "", "timeout"
    except error.URLError as exc:
        if isinstance(getattr(exc, "reason", None), socket.timeout):
            return "", "timeout"
        return "", "model_error"
    except Exception:
        return "", "model_error"


def maybe_generate_explanation(
    triggered: bool,
    question: str,
    sql: str,
    cache: dict[str, str],
    model: str,
    ollama_url: str,
    timeout_seconds: int,
    generate_fn: Callable[[str, str, str, int], tuple[str, str]] = generate_sql_explanation,
) -> tuple[str, str, bool]:
    """
    On-demand explanation workflow with cache.
    Returns (text, status, cache_hit)
    status: idle | ready | timeout | model_error | empty_response
    """
    if not triggered:
        return "", "idle", False

    cache_key = build_explanation_cache_key(question, sql)
    cached = cache.get(cache_key, "")
    if cached:
        return cached, "ready", True

    prompt = build_explanation_prompt(question, sql)
    text, error_code = generate_fn(
        model=model,
        prompt=prompt,
        ollama_url=ollama_url,
        timeout_seconds=timeout_seconds,
    )
    if error_code:
        return "", error_code, False

    cache[cache_key] = text
    return text, "ready", False
