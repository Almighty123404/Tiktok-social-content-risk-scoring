"""
02_run_eda.py
Runs every query in sql/02_exploratory_analysis.sql against the SQLite
DB, prints results, and saves them to outputs/eda_results.md for the
project write-up / GitHub README.
"""
import sqlite3
import pandas as pd
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
DB = BASE / "trust_safety.db"
SQL_FILE = BASE / "sql" / "02_exploratory_analysis.sql"
OUT = BASE / "outputs" / "eda_results.md"

def split_queries(sql_text):
    """Split the .sql file into (comment_title, query) blocks on blank lines."""
    blocks = [b.strip() for b in sql_text.split("\n\n\n") if b.strip()]
    parsed = []
    for b in blocks:
        lines = b.splitlines()
        title_lines = [l.lstrip("- ").strip() for l in lines if l.strip().startswith("--")]
        title = " ".join(title_lines).replace("--", "").strip()
        query = "\n".join(l for l in lines if not l.strip().startswith("--"))
        if query.strip():
            parsed.append((title, query.strip()))
    return parsed

def main():
    conn = sqlite3.connect(DB)
    sql_text = SQL_FILE.read_text()
    queries = split_queries(sql_text)

    md_lines = ["# Exploratory SQL Analysis Results\n"]
    for i, (title, query) in enumerate(queries, 1):
        print(f"\n{'='*70}\nQ{i}: {title}\n{'='*70}")
        df = pd.read_sql_query(query, conn)
        print(df.to_string(index=False))

        md_lines.append(f"## Q{i}. {title}\n")
        md_lines.append(df.to_markdown(index=False))
        md_lines.append("\n")

    OUT.write_text("\n".join(md_lines))
    print(f"\nSaved results to {OUT}")
    conn.close()

if __name__ == "__main__":
    main()
