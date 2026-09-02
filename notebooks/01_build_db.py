"""
01_build_db.py
Loads the raw TikTok CSV, applies light cleaning, and writes it into a
SQLite database so we can do all exploratory analysis in real SQL.

Why SQLite: keeps the project fully portable/reproducible on GitHub
without requiring a Postgres/MySQL server. Same SQL (CTEs, window
functions, aggregates) still applies — swap the connection string for
Postgres/MySQL in production without changing query logic.
"""
import sqlite3
import pandas as pd
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
DATA = BASE / "data" / "tiktok_dataset.csv"
DB = BASE / "trust_safety.db"
SCHEMA = BASE / "sql" / "01_schema.sql"

def main():
    df = pd.read_csv(DATA)

    print("Raw shape:", df.shape)
    print("Missing values:\n", df.isna().sum())

    # --- Cleaning ---
    # 298 rows have nulls across claim_status + all engagement metrics
    # together (same rows) -- these are unlabeled/incomplete records with
    # no analytical value for either the SQL layer or the ML target.
    before = len(df)
    df = df.dropna(subset=["claim_status", "video_view_count"]).copy()
    print(f"Dropped {before - len(df)} incomplete rows -> {len(df)} remain")

    # Normalize text categoricals (defensive lowercasing/stripping)
    for col in ["claim_status", "verified_status", "author_ban_status"]:
        df[col] = df[col].str.strip().str.lower()

    # Rename the index column for clarity
    df = df.rename(columns={"#": "row_id"})

    # --- Load into SQLite ---
    conn = sqlite3.connect(DB)
    with open(SCHEMA) as f:
        conn.executescript(f.read())

    df.to_sql("videos", conn, if_exists="append", index=False)
    conn.commit()

    n = conn.execute("SELECT COUNT(*) FROM videos").fetchone()[0]
    print(f"Loaded {n} rows into {DB.name} (table: videos)")
    conn.close()

if __name__ == "__main__":
    main()
