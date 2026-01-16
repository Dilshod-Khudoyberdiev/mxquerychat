# tools/export_duckdb_ddl.py
# This script reads your DuckDB database and exports CREATE TABLE statements
# into docs/schema_ddl.sql so you can use them for training and your thesis report.

from pathlib import Path
import duckdb

DUCKDB_PATH = "mxquerychat.duckdb"
OUTPUT_PATH = Path("docs/schema_ddl.sql")


def main():
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect(DUCKDB_PATH, read_only=True)

    tables = con.execute("""
        SELECT table_name
        FROM information_schema.tables
        WHERE table_schema = 'main'
        ORDER BY table_name
    """).fetchall()

    ddl_parts = []
    ddl_parts.append("-- Auto-generated DDL from DuckDB (synthetic dataset)\n")

    for (table_name,) in tables:
        # DuckDB usually supports SHOW CREATE TABLE
        try:
            ddl_row = con.execute(f"SHOW CREATE TABLE {table_name}").fetchone()
            if ddl_row and ddl_row[0]:
                ddl_parts.append(ddl_row[0] + ";\n")
                continue
        except Exception:
            pass

        # Fallback: build a basic CREATE TABLE from columns
        cols = con.execute(f"""
            SELECT column_name, data_type
            FROM information_schema.columns
            WHERE table_name = '{table_name}'
            ORDER BY ordinal_position
        """).fetchall()

        cols_sql = ",\n  ".join([f"{c} {t}" for c, t in cols])
        ddl_parts.append(f"CREATE TABLE {table_name} (\n  {cols_sql}\n);\n")

    con.close()

    OUTPUT_PATH.write_text("\n".join(ddl_parts), encoding="utf-8")
    print(f"Done! Wrote DDL to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
