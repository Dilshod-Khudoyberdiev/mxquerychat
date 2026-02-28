import time

import pandas as pd
import pytest

vannaagent = pytest.importorskip("vannaagent")


def test_ensure_training_schema_adds_timestamp_columns() -> None:
    df = pd.DataFrame([{"question": "q1", "sql": "SELECT 1", "description": "d"}])
    normalized = vannaagent.ensure_training_schema(df)
    assert list(normalized.columns) == [
        "question",
        "sql",
        "description",
        "created_at",
        "updated_at",
    ]


def test_save_training_examples_preserves_created_at_and_updates_changed_rows(
    tmp_path, monkeypatch
) -> None:
    csv_path = tmp_path / "training_examples.csv"
    monkeypatch.setattr(vannaagent, "TRAINING_CSV_PATH", csv_path)

    first_df = pd.DataFrame(
        [{"question": "Q1", "sql": "SELECT 1", "description": "desc"}]
    )
    vannaagent.save_training_examples(first_df)
    loaded_1 = vannaagent.load_training_examples()
    created_1 = loaded_1.loc[0, "created_at"]
    updated_1 = loaded_1.loc[0, "updated_at"]

    # Unchanged row should keep created_at and updated_at.
    vannaagent.save_training_examples(first_df)
    loaded_2 = vannaagent.load_training_examples()
    assert loaded_2.loc[0, "created_at"] == created_1
    assert loaded_2.loc[0, "updated_at"] == updated_1

    # Changed content should preserve created_at but refresh updated_at.
    time.sleep(1.1)
    changed_df = pd.DataFrame(
        [{"question": "Q1", "sql": "SELECT 1", "description": "changed"}]
    )
    vannaagent.save_training_examples(changed_df)
    loaded_3 = vannaagent.load_training_examples()
    assert loaded_3.loc[0, "created_at"] == created_1
    assert loaded_3.loc[0, "updated_at"] != updated_1


def test_normalize_training_for_save_drops_empty_and_counts_duplicates() -> None:
    df = pd.DataFrame(
        [
            {"question": "Q1", "sql": "SELECT 1", "description": "a"},
            {"question": "Q1", "sql": "SELECT 1", "description": "b"},
            {"question": "", "sql": "SELECT 2", "description": "missing question"},
            {"question": "Q3", "sql": "", "description": "missing sql"},
        ]
    )

    cleaned, stats = vannaagent.normalize_training_for_save(df)
    assert len(cleaned) == 2
    assert stats["rows_before"] == 4
    assert stats["rows_after"] == 2
    assert stats["dropped_missing_question_or_sql"] == 2
    assert stats["duplicate_question_sql_rows"] == 1
