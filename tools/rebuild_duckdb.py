import sys
from pathlib import Path

try:
    import duckdb
except ImportError as exc:
    raise SystemExit("duckdb is required. Install with: pip install duckdb") from exc


def main() -> int:
    no_delete = "--no-delete" in sys.argv
    repo_root = Path(__file__).resolve().parents[1]
    db_path = repo_root / "mxquerychat.duckdb"
    sql_dir = repo_root / "sql"

    sql_files = sorted(sql_dir.glob("ingest_data_*.sql"))
    if not sql_files:
        raise SystemExit("No ingest scripts found in sql/.")

    if no_delete and not db_path.exists():
        raise SystemExit(f"{db_path} does not exist. Run without --no-delete to create it.")

    if db_path.exists() and not no_delete:
        db_path.unlink()

    conn = duckdb.connect(str(db_path))
    try:
        for path in sql_files:
            conn.execute(path.read_text(encoding="ascii"))
    finally:
        conn.close()

    print(f"Rebuilt {db_path}")
    print("Loaded tables:")
    for path in sql_files:
        print(f"- {path.stem.replace('ingest_data_', '')}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
