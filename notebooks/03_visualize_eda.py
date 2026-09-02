"""
03_visualize_eda.py
Generates the key charts for the write-up, pulling straight from SQL.
"""
import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
DB = BASE / "trust_safety.db"
OUT = BASE / "outputs"

plt.rcParams["figure.dpi"] = 130
COLORS = {"active": "#4C9F70", "under review": "#E8A33D", "banned": "#C1443C"}

def main():
    conn = sqlite3.connect(DB)

    # --- Chart 1: claim vs opinion share within each ban status ---
    df1 = pd.read_sql_query("""
        SELECT author_ban_status, claim_status, COUNT(*) as n
        FROM videos GROUP BY author_ban_status, claim_status
    """, conn)
    pivot1 = df1.pivot(index="author_ban_status", columns="claim_status", values="n")
    pivot1 = pivot1.loc[["active", "under review", "banned"]]
    pivot1_pct = pivot1.div(pivot1.sum(axis=1), axis=0) * 100

    fig, ax = plt.subplots(figsize=(6, 4))
    pivot1_pct.plot(kind="barh", stacked=True, ax=ax, color=["#3E6FA8", "#B7C9DC"])
    ax.set_xlabel("% of videos")
    ax.set_ylabel("")
    ax.set_title("Claim vs. Opinion Content Share by Author Ban Status")
    ax.legend(title="Content type", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(OUT / "01_claim_share_by_ban_status.png")
    plt.close()

    # --- Chart 2: engagement ratios by ban status ---
    df2 = pd.read_sql_query("""
        SELECT author_ban_status,
               AVG(1.0*video_share_count/NULLIF(video_view_count,0)) AS share_rate,
               AVG(1.0*video_comment_count/NULLIF(video_view_count,0)) AS comment_rate,
               AVG(1.0*video_like_count/NULLIF(video_view_count,0)) AS like_rate
        FROM videos GROUP BY author_ban_status
    """, conn).set_index("author_ban_status").loc[["active", "under review", "banned"]]

    fig, ax = plt.subplots(figsize=(6, 4))
    df2.plot(kind="bar", ax=ax, color=["#3E6FA8", "#E8A33D", "#4C9F70"])
    ax.set_ylabel("Avg. rate per view")
    ax.set_title("Engagement Ratios by Author Ban Status")
    ax.set_xlabel("")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(OUT / "02_engagement_ratios_by_ban_status.png")
    plt.close()

    # --- Chart 3: % of platform views/shares from at-risk authors ---
    df3 = pd.read_sql_query("""
        WITH totals AS (SELECT SUM(video_view_count) tv, SUM(video_share_count) ts FROM videos)
        SELECT
          100.0*SUM(CASE WHEN author_ban_status IN ('banned','under review') THEN video_view_count ELSE 0 END)/t.tv AS pct_views,
          100.0*SUM(CASE WHEN author_ban_status IN ('banned','under review') THEN video_share_count ELSE 0 END)/t.ts AS pct_shares
        FROM videos, totals t
    """, conn)

    fig, ax = plt.subplots(figsize=(5, 4))
    vals = [df3["pct_views"][0], 100 - df3["pct_views"][0]]
    ax.pie(vals, labels=["At-risk authors\n(banned + under review)", "Active authors"],
           autopct="%1.1f%%", colors=["#C1443C", "#4C9F70"], startangle=90)
    ax.set_title("Share of Total Platform Views\nby Author Risk Status")
    plt.tight_layout()
    plt.savefig(OUT / "03_at_risk_view_share.png")
    plt.close()

    conn.close()
    print("Saved 3 charts to", OUT)

if __name__ == "__main__":
    main()
