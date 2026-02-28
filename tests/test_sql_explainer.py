from urllib import error

from src.llm.sql_explainer import (
    build_explanation_cache_key,
    build_explanation_prompt,
    generate_sql_explanation,
)


def test_build_explanation_prompt_contains_question_and_sql() -> None:
    prompt = build_explanation_prompt(
        "Show revenue by month",
        "SELECT monat, SUM(umsatz_eur) FROM ticket_verkaeufe GROUP BY monat",
    )
    assert "Question:" in prompt
    assert "SQL:" in prompt
    assert "Show revenue by month" in prompt
    assert "SUM(umsatz_eur)" in prompt


def test_build_explanation_cache_key_is_stable() -> None:
    key_1 = build_explanation_cache_key("Q1", "SELECT 1")
    key_2 = build_explanation_cache_key("Q1", "SELECT 1")
    key_3 = build_explanation_cache_key("Q1", "SELECT 2")
    assert key_1 == key_2
    assert key_1 != key_3


def test_generate_sql_explanation_success(monkeypatch) -> None:
    class _Resp:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self):
            return b'{"response":"This query groups revenue by month."}'

    monkeypatch.setattr("src.llm.sql_explainer.request.urlopen", lambda *args, **kwargs: _Resp())

    text, err = generate_sql_explanation(
        model="mistral",
        prompt="p",
        ollama_url="http://127.0.0.1:11434",
        timeout_seconds=8,
    )
    assert err == ""
    assert "groups revenue by month" in text


def test_generate_sql_explanation_timeout(monkeypatch) -> None:
    def _raise_timeout(*args, **kwargs):
        raise TimeoutError("timed out")

    monkeypatch.setattr("src.llm.sql_explainer.request.urlopen", _raise_timeout)
    text, err = generate_sql_explanation(
        model="mistral",
        prompt="p",
        ollama_url="http://127.0.0.1:11434",
        timeout_seconds=1,
    )
    assert text == ""
    assert err == "timeout"


def test_generate_sql_explanation_model_error(monkeypatch) -> None:
    def _raise_error(*args, **kwargs):
        raise error.URLError("connection refused")

    monkeypatch.setattr("src.llm.sql_explainer.request.urlopen", _raise_error)
    text, err = generate_sql_explanation(
        model="mistral",
        prompt="p",
        ollama_url="http://127.0.0.1:11434",
        timeout_seconds=8,
    )
    assert text == ""
    assert err == "model_error"
