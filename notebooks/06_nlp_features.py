"""
06_nlp_features.py

NLP FEATURE ENGINEERING FROM video_transcription_text
=====================================================
The transcription text column was completely unused in V1. This script
extracts both TF-IDF and hand-engineered text features, measures their
correlation with the existing is_claim flag, and tests whether they
provide incremental predictive lift.

Key finding from data exploration:
  - Claims use prefixes like "someone shared", "a friend read in the media"
  - Opinions use prefixes like "i think", "my colleagues' opinion is"
  - This means many NLP features will be proxies for claim_status

We document this correlation explicitly -- an honest analysis of what
NLP features do and don't add is more valuable than pretending they're
independent signals.

Run after: 01_build_db.py
Outputs:   outputs/nlp_feature_correlation.png
           Enriched feature matrix saved to models/feature_matrix.joblib
"""
import sqlite3
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from pathlib import Path
import joblib

BASE = Path(__file__).resolve().parent.parent
DB = BASE / "trust_safety.db"
OUT = BASE / "outputs"
MODELS = BASE / "models"

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)


def load_data():
    conn = sqlite3.connect(DB)
    df = pd.read_sql_query("""
        SELECT
            video_duration_sec,
            video_view_count,
            video_like_count,
            video_share_count,
            video_download_count,
            video_comment_count,
            claim_status,
            verified_status,
            author_ban_status,
            video_transcription_text
        FROM videos
    """, conn)
    conn.close()
    return df


def engineer_text_features(df):
    """Hand-engineered text features from video_transcription_text."""
    text = df["video_transcription_text"].fillna("")

    features = pd.DataFrame(index=df.index)

    # --- Basic text statistics ---
    features["text_length"]      = text.str.len()
    features["word_count"]       = text.str.split().str.len().fillna(0).astype(int)
    features["avg_word_length"]  = (features["text_length"] /
                                    features["word_count"].replace(0, np.nan)).fillna(0)

    # --- Punctuation & formatting signals ---
    features["exclamation_count"] = text.str.count("!")
    features["question_count"]    = text.str.count(r"\?")
    features["exclamation_density"] = (features["exclamation_count"] /
                                       features["word_count"].replace(0, np.nan)).fillna(0)

    # --- Numeric claims (numbers in text suggest factual claims) ---
    features["number_count"] = text.apply(lambda t: len(re.findall(r'\d+', t)))
    features["has_dollar"]   = text.str.contains(r'\$', regex=True).astype(int)
    features["has_percent"]  = text.str.contains(r'%', regex=False).astype(int)

    # --- Superlatives (often in sensational claims) ---
    superlatives = r'\b(most|largest|biggest|greatest|highest|fastest|' \
                   r'smallest|lowest|worst|best|deepest|longest|oldest|' \
                   r'richest|poorest|tallest|shortest)\b'
    features["superlative_count"] = text.str.lower().str.count(superlatives)
    features["has_superlative"]   = (features["superlative_count"] > 0).astype(int)

    # --- Hedging phrases (opinion markers) ---
    hedging = r'\b(i think|i believe|in my opinion|my opinion|my view|' \
              r'my understanding|i feel|i reckon|i suppose|my hypothesis|' \
              r'willing to wager|my colleagues)\b'
    features["hedging_count"]  = text.str.lower().str.count(hedging)
    features["has_hedging"]    = (features["hedging_count"] > 0).astype(int)

    # --- Claim-triggering phrases ---
    claim_triggers = r'\b(someone shared|a friend|shared with me|' \
                     r'read in the media|learned from the media|' \
                     r'a colleague|i learned|someone read|i read)\b'
    features["claim_trigger_count"] = text.str.lower().str.count(claim_triggers)
    features["has_claim_trigger"]   = (features["claim_trigger_count"] > 0).astype(int)

    # --- Named entity proxies (capitalized words, often in factual claims) ---
    features["capitalized_word_count"] = text.apply(
        lambda t: len([w for w in t.split() if w and w[0].isupper()])
    )

    return features


def engineer_tfidf_features(df, n_components=20):
    """TF-IDF + SVD dimensionality reduction for text."""
    text = df["video_transcription_text"].fillna("")

    tfidf = TfidfVectorizer(
        max_features=200,
        stop_words="english",
        ngram_range=(1, 2),
        min_df=5,
        max_df=0.95,
    )
    tfidf_matrix = tfidf.fit_transform(text)

    svd = TruncatedSVD(n_components=n_components, random_state=RANDOM_STATE)
    tfidf_reduced = svd.fit_transform(tfidf_matrix)

    tfidf_df = pd.DataFrame(
        tfidf_reduced,
        columns=[f"tfidf_svd_{i}" for i in range(n_components)],
        index=df.index
    )

    print(f"TF-IDF: {tfidf_matrix.shape[1]} terms -> SVD {n_components} components")
    print(f"SVD explained variance: {svd.explained_variance_ratio_.sum():.3f}")
    return tfidf_df, tfidf, svd


def build_base_features(df):
    """Reproduce the V1 feature set for comparison."""
    features = pd.DataFrame(index=df.index)
    features["video_duration_sec"]   = df["video_duration_sec"]
    features["video_view_count"]     = df["video_view_count"]
    features["video_like_count"]     = df["video_like_count"]
    features["video_share_count"]    = df["video_share_count"]
    features["video_download_count"] = df["video_download_count"]
    features["video_comment_count"]  = df["video_comment_count"]

    features["share_rate"]    = df["video_share_count"] / df["video_view_count"].replace(0, np.nan)
    features["comment_rate"]  = df["video_comment_count"] / df["video_view_count"].replace(0, np.nan)
    features["like_rate"]     = df["video_like_count"] / df["video_view_count"].replace(0, np.nan)
    features["download_rate"] = df["video_download_count"] / df["video_view_count"].replace(0, np.nan)
    features[["share_rate", "comment_rate", "like_rate", "download_rate"]] = \
        features[["share_rate", "comment_rate", "like_rate", "download_rate"]].fillna(0)

    features["is_claim"]    = (df["claim_status"] == "claim").astype(int)
    features["is_verified"] = (df["verified_status"] == "verified").astype(int)

    return features


def measure_correlation_with_claim(text_features, is_claim):
    """Measure Pearson correlation between each NLP feature and is_claim."""
    correlations = text_features.corrwith(is_claim).sort_values(key=abs, ascending=False)
    print("\n" + "=" * 70)
    print("NLP Feature Correlation with is_claim")
    print("(High |r| = feature is a proxy for claim_status, not independent signal)")
    print("=" * 70)
    for feat, corr in correlations.items():
        flag = " [!] PROXY" if abs(corr) > 0.5 else ""
        print(f"  {feat:30s}  r = {corr:+.4f}{flag}")
    return correlations


def compare_feature_sets(base_features, text_features, tfidf_features, y):
    """Compare cross-validated ROC-AUC across feature set configurations."""
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    rf = RandomForestClassifier(n_estimators=200, max_depth=10,
                                class_weight="balanced",
                                random_state=RANDOM_STATE, n_jobs=-1)

    configs = {
        "V1 baseline (engagement + is_claim)": base_features,
        "+ hand-engineered NLP features": pd.concat([base_features, text_features], axis=1),
        "+ TF-IDF SVD features": pd.concat([base_features, tfidf_features], axis=1),
        "+ all NLP features (hand + TF-IDF)": pd.concat([base_features, text_features, tfidf_features], axis=1),
        "NLP only (no engagement)": pd.concat([text_features, tfidf_features], axis=1),
    }

    print("\n" + "=" * 70)
    print("FEATURE SET COMPARISON -- 5-Fold Stratified CV ROC-AUC")
    print("=" * 70)
    results = {}
    for name, X in configs.items():
        scores = cross_val_score(rf, X, y, cv=cv, scoring="roc_auc", n_jobs=-1)
        results[name] = {"mean": scores.mean(), "std": scores.std(), "scores": scores}
        print(f"  {name:45s}  AUC = {scores.mean():.4f} +/- {scores.std():.4f}")

    return results


def plot_correlation_heatmap(text_features, is_claim):
    """Plot NLP feature correlation with is_claim."""
    correlations = text_features.corrwith(is_claim).sort_values(key=abs, ascending=False)

    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ["#C1443C" if abs(c) > 0.5 else "#E8A33D" if abs(c) > 0.2 else "#4C9F70"
              for c in correlations.values]
    bars = ax.barh(range(len(correlations)), correlations.values, color=colors)
    ax.set_yticks(range(len(correlations)))
    ax.set_yticklabels(correlations.index, fontsize=8)
    ax.set_xlabel("Pearson correlation with is_claim")
    ax.set_title("NLP Feature Correlation with Claim Status\n"
                 "(Red = strong proxy, Orange = moderate, Green = independent)",
                 fontsize=11)
    ax.axvline(x=0, color="black", linewidth=0.5)
    ax.axvline(x=0.5, color="red", linewidth=0.5, linestyle="--", alpha=0.5)
    ax.axvline(x=-0.5, color="red", linewidth=0.5, linestyle="--", alpha=0.5)
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(OUT / "08_nlp_feature_correlation.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nSaved correlation plot to {OUT / '08_nlp_feature_correlation.png'}")


def main():
    print("Loading data...")
    df = load_data()
    print(f"Loaded {len(df)} rows\n")

    # --- Build feature sets ---
    base_features = build_base_features(df)
    text_features = engineer_text_features(df)
    tfidf_features, tfidf_vec, svd_model = engineer_tfidf_features(df)

    is_claim = (df["claim_status"] == "claim").astype(int)
    y = df["author_ban_status"].isin(["banned", "under review"]).astype(int)

    # --- Correlation analysis ---
    correlations = measure_correlation_with_claim(text_features, is_claim)
    plot_correlation_heatmap(text_features, is_claim)

    # --- Compare feature sets ---
    results = compare_feature_sets(base_features, text_features, tfidf_features, y)

    # --- Save enriched feature matrix for advanced modeling ---
    full_features = pd.concat([base_features, text_features, tfidf_features], axis=1)
    feature_data = {
        "X": full_features,
        "y_binary": y,
        "y_multiclass": df["author_ban_status"],
        "base_feature_cols": list(base_features.columns),
        "text_feature_cols": list(text_features.columns),
        "tfidf_feature_cols": list(tfidf_features.columns),
        "all_feature_cols": list(full_features.columns),
        "tfidf_vectorizer": tfidf_vec,
        "svd_model": svd_model,
    }
    joblib.dump(feature_data, MODELS / "feature_matrix.joblib")
    print(f"\nSaved enriched feature matrix to {MODELS / 'feature_matrix.joblib'}")

    # --- Summary ---
    print("\n" + "=" * 70)
    print("NLP FEATURE ENGINEERING -- SUMMARY")
    print("=" * 70)
    baseline_auc = results["V1 baseline (engagement + is_claim)"]["mean"]
    best_name = max(results, key=lambda k: results[k]["mean"])
    best_auc = results[best_name]["mean"]
    lift = best_auc - baseline_auc

    print(f"""
Baseline ROC-AUC (V1 features):           {baseline_auc:.4f}
Best configuration:                        {best_name}
Best ROC-AUC:                              {best_auc:.4f}
Incremental lift from NLP features:        {lift:+.4f}

Key insight: Many hand-engineered NLP features (hedging phrases, claim triggers)
are strong proxies for claim_status (|r| > 0.5), confirming that the text
structure is highly predictable from the claim/opinion label. However, TF-IDF
SVD components may capture subtler linguistic patterns not fully encoded in
the binary is_claim flag.

{"NLP features provide meaningful lift (" + f"+{lift:.4f}" + " AUC)." if lift > 0.005
 else "NLP features provide minimal incremental lift over is_claim alone -- the text "
      "structure is largely redundant with the claim_status label. This is an honest "
      "negative result: the text column's predictive value is already captured by is_claim."}
""")
    print("[OK] NLP feature engineering complete.")


if __name__ == "__main__":
    main()
