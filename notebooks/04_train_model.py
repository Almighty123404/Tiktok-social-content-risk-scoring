"""
04_train_model.py

Goal: predict whether a video's author is "at risk" (banned OR under
review) vs "active", using engagement + content features pulled
straight from the SQL layer. Mirrors the structure of a credit-default
risk model: structured features -> binary risk flag -> ensemble
classifier, evaluated on recall for the risk class (the costly error
here is missing an at-risk author, not flagging a safe one).
"""
import sqlite3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    classification_report, roc_auc_score, roc_curve,
    confusion_matrix, ConfusionMatrixDisplay
)
from xgboost import XGBClassifier
import joblib

BASE = Path(__file__).resolve().parent.parent
DB = BASE / "trust_safety.db"
OUT = BASE / "outputs"
MODELS = BASE / "models"

RANDOM_STATE = 42

def load_features():
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
            author_ban_status
        FROM videos
    """, conn)
    conn.close()

    # --- Feature engineering (ratios computed in-DB style, done here for
    # clarity but equivalent to the SQL in 02_exploratory_analysis.sql) ---
    df["share_rate"]    = df["video_share_count"]    / df["video_view_count"].replace(0, np.nan)
    df["comment_rate"]  = df["video_comment_count"]  / df["video_view_count"].replace(0, np.nan)
    df["like_rate"]     = df["video_like_count"]     / df["video_view_count"].replace(0, np.nan)
    df["download_rate"] = df["video_download_count"] / df["video_view_count"].replace(0, np.nan)
    df[["share_rate", "comment_rate", "like_rate", "download_rate"]] = \
        df[["share_rate", "comment_rate", "like_rate", "download_rate"]].fillna(0)

    df["is_claim"]     = (df["claim_status"] == "claim").astype(int)
    df["is_verified"]  = (df["verified_status"] == "verified").astype(int)

    # Binary target: at_risk = banned OR under review
    df["at_risk"] = df["author_ban_status"].isin(["banned", "under review"]).astype(int)

    feature_cols = [
        "video_duration_sec", "video_view_count", "video_like_count",
        "video_share_count", "video_download_count", "video_comment_count",
        "share_rate", "comment_rate", "like_rate", "download_rate",
        "is_claim", "is_verified"
    ]
    return df[feature_cols], df["at_risk"], feature_cols


def main():
    X, y, feature_cols = load_features()
    print("Class balance:\n", y.value_counts(normalize=True).round(3))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, max_depth=10, class_weight="balanced",
            random_state=RANDOM_STATE, n_jobs=-1
        ),
        "XGBoost": XGBClassifier(
            n_estimators=300, max_depth=5, learning_rate=0.05,
            eval_metric="logloss", random_state=RANDOM_STATE,
            scale_pos_weight=(y_train == 0).sum() / (y_train == 1).sum()
        ),
        "MLP": MLPClassifier(
            hidden_layer_sizes=(64, 32), max_iter=500, random_state=RANDOM_STATE
        ),
    }

    results = {}
    fitted = {}
    for name, model in models.items():
        use_scaled = name in ("Logistic Regression", "MLP")
        Xtr, Xte = (X_train_s, X_test_s) if use_scaled else (X_train, X_test)
        model.fit(Xtr, y_train)
        proba = model.predict_proba(Xte)[:, 1]
        preds = model.predict(Xte)
        auc = roc_auc_score(y_test, proba)
        results[name] = auc
        fitted[name] = model
        print(f"\n--- {name} (ROC-AUC = {auc:.4f}) ---")
        print(classification_report(y_test, preds, target_names=["active", "at_risk"]))

    # --- Soft-voting ensemble: RF + XGBoost + MLP (mirrors a
    # 2-stage ensemble approach) ---
    ensemble = VotingClassifier(
        estimators=[
            ("rf", fitted["Random Forest"]),
            ("xgb", fitted["XGBoost"]),
        ],
        voting="soft"
    )
    # VotingClassifier refit needed since fitted models were trained separately;
    # for a clean ensemble we refit lightweight on unscaled data.
    ensemble.fit(X_train, y_train)
    ens_proba = ensemble.predict_proba(X_test)[:, 1]
    ens_preds = ensemble.predict(X_test)
    ens_auc = roc_auc_score(y_test, ens_proba)
    results["RF + XGBoost Ensemble"] = ens_auc
    print(f"\n--- RF + XGBoost Ensemble (ROC-AUC = {ens_auc:.4f}) ---")
    print(classification_report(y_test, ens_preds, target_names=["active", "at_risk"]))

    # --- Save best model ---
    best_name = max(results, key=results.get)
    print(f"\nBest model: {best_name} (ROC-AUC = {results[best_name]:.4f})")
    best_model = ensemble if best_name == "RF + XGBoost Ensemble" else fitted[best_name]
    joblib.dump(best_model, MODELS / "at_risk_classifier.joblib")
    joblib.dump(scaler, MODELS / "scaler.joblib")

    # --- Charts: ROC comparison ---
    fig, ax = plt.subplots(figsize=(6, 5))
    for name, model in list(fitted.items()) + [("RF+XGB Ensemble", ensemble)]:
        use_scaled = name in ("Logistic Regression", "MLP")
        Xte = X_test_s if use_scaled else X_test
        proba = model.predict_proba(Xte)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, proba)
        ax.plot(fpr, tpr, label=f"{name} (AUC={roc_auc_score(y_test, proba):.3f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves — At-Risk Author Classification")
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(OUT / "04_roc_curves.png")
    plt.close()

    # --- Chart: feature importance (XGBoost) ---
    fig, ax = plt.subplots(figsize=(6, 5))
    importances = fitted["XGBoost"].feature_importances_
    order = np.argsort(importances)
    ax.barh(np.array(feature_cols)[order], importances[order], color="#3E6FA8")
    ax.set_title("Feature Importance — XGBoost")
    plt.tight_layout()
    plt.savefig(OUT / "05_feature_importance.png")
    plt.close()

    # --- Chart: confusion matrix for best model ---
    fig, ax = plt.subplots(figsize=(5, 5))
    cm_preds = ens_preds if best_name == "RF + XGBoost Ensemble" else fitted[best_name].predict(
        X_test_s if best_name in ("Logistic Regression", "MLP") else X_test
    )
    cm = confusion_matrix(y_test, cm_preds)
    ConfusionMatrixDisplay(cm, display_labels=["active", "at_risk"]).plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(f"Confusion Matrix — {best_name}")
    plt.tight_layout()
    plt.savefig(OUT / "06_confusion_matrix.png")
    plt.close()

    print("\nSaved model + charts to models/ and outputs/")

if __name__ == "__main__":
    main()
