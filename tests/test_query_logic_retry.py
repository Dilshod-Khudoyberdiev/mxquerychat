from src.core.query_logic import generate_sql_with_retry


SCHEMA_TREE = {
    "ticket_verkaeufe": [
        ("jahr", "INTEGER"),
        ("monat", "INTEGER"),
        ("umsatz_eur", "DOUBLE"),
        ("plz", "VARCHAR"),
        ("ticket_code", "VARCHAR"),
        ("tarifverbund_id", "INTEGER"),
    ],
    "postleitzahlen": [("plz", "VARCHAR"), ("bundesland_code2", "VARCHAR")],
    "regionen_bundesland": [
        ("bundesland_code2", "VARCHAR"),
        ("bundesland_name", "VARCHAR"),
    ],
    "ticket_produkte": [("ticket_code", "VARCHAR"), ("ticket_name", "VARCHAR")],
    "tarifverbuende": [("tarifverbund_id", "INTEGER"), ("name", "VARCHAR")],
}


def ok_timeout_runner(fn, timeout_seconds):
    return fn(), None


def test_retry_success_on_first_attempt() -> None:
    def generate_sql_fn(_prompt: str) -> str:
        return "SELECT 1"

    def compile_sql_fn(_sql: str) -> str:
        return ""

    sql, notes, error = generate_sql_with_retry(
        generate_sql_fn=generate_sql_fn,
        question_text="What is total revenue?",
        schema_tree=SCHEMA_TREE,
        compile_sql_fn=compile_sql_fn,
        timeout_seconds=30,
        run_with_timeout_fn=ok_timeout_runner,
    )

    assert sql == "SELECT 1"
    assert error == ""
    assert any("attempt 1" in note.lower() and "valid sql" in note.lower() for note in notes)


def test_retry_compile_error_then_success_on_second_attempt() -> None:
    outputs = ["SELECT * FROM bad_table", "SELECT * FROM ticket_verkaeufe"]

    def generate_sql_fn(_prompt: str) -> str:
        return outputs.pop(0)

    def compile_sql_fn(sql: str) -> str:
        return "table not found" if "bad_table" in sql else ""

    sql, notes, error = generate_sql_with_retry(
        generate_sql_fn=generate_sql_fn,
        question_text="Show revenue by state",
        schema_tree=SCHEMA_TREE,
        compile_sql_fn=compile_sql_fn,
        timeout_seconds=30,
        run_with_timeout_fn=ok_timeout_runner,
    )

    assert sql == "SELECT * FROM ticket_verkaeufe"
    assert error == ""
    assert any("schema-aware retry" in note.lower() and "valid sql" in note.lower() for note in notes)
    assert any("resolved using related tables" in note.lower() for note in notes)


def test_retry_non_sql_then_final_attempt_success() -> None:
    outputs = ["This is not SQL", "still not sql", "SELECT * FROM ticket_verkaeufe"]

    def generate_sql_fn(_prompt: str) -> str:
        return outputs.pop(0)

    def compile_sql_fn(_sql: str) -> str:
        return ""

    sql, notes, error = generate_sql_with_retry(
        generate_sql_fn=generate_sql_fn,
        question_text="Show revenue by month",
        schema_tree=SCHEMA_TREE,
        compile_sql_fn=compile_sql_fn,
        timeout_seconds=30,
        run_with_timeout_fn=ok_timeout_runner,
    )

    assert sql == "SELECT * FROM ticket_verkaeufe"
    assert error == ""
    assert any("final strict attempt" in note.lower() and "valid sql" in note.lower() for note in notes)
    assert any("resolved on final strict attempt" in note.lower() for note in notes)


def test_retry_timeout_returns_timeout_error_code() -> None:
    def generate_sql_fn(_prompt: str) -> str:
        return "SELECT 1"

    def compile_sql_fn(_sql: str) -> str:
        return ""

    def timeout_runner(_fn, _timeout_seconds):
        return None, "Model timeout after 30 seconds."

    sql, notes, error = generate_sql_with_retry(
        generate_sql_fn=generate_sql_fn,
        question_text="Show revenue",
        schema_tree=SCHEMA_TREE,
        compile_sql_fn=compile_sql_fn,
        timeout_seconds=30,
        run_with_timeout_fn=timeout_runner,
    )

    assert sql == ""
    assert error == "timeout"
    assert any("model timeout" in note.lower() for note in notes)


def test_retry_model_error_returns_model_error_code() -> None:
    def generate_sql_fn(_prompt: str) -> str:
        return "SELECT 1"

    def compile_sql_fn(_sql: str) -> str:
        return ""

    def error_runner(_fn, _timeout_seconds):
        return None, "Connection refused."

    sql, notes, error = generate_sql_with_retry(
        generate_sql_fn=generate_sql_fn,
        question_text="Show revenue",
        schema_tree=SCHEMA_TREE,
        compile_sql_fn=compile_sql_fn,
        timeout_seconds=30,
        run_with_timeout_fn=error_runner,
    )

    assert sql == ""
    assert error == "model_error"
    assert any("model error" in note.lower() for note in notes)


def test_retry_no_match_after_all_attempts() -> None:
    outputs = ["NO_MATCH", "NO_MATCH", "NO_MATCH"]

    def generate_sql_fn(_prompt: str) -> str:
        return outputs.pop(0)

    def compile_sql_fn(_sql: str) -> str:
        return ""

    sql, notes, error = generate_sql_with_retry(
        generate_sql_fn=generate_sql_fn,
        question_text="Unknown request",
        schema_tree=SCHEMA_TREE,
        compile_sql_fn=compile_sql_fn,
        timeout_seconds=30,
        run_with_timeout_fn=ok_timeout_runner,
    )

    assert sql == ""
    assert error == "no_match"
    assert any("all generation strategies failed" in note.lower() for note in notes)

