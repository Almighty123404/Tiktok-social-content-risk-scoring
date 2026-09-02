"""
08_shap_explainability.py

SHAP EXPLAINABILITY FOR MODEL INTERPRETABILITY
================================================
Extract interpretable feature contributions using SHAP (SHapley Additive exPlanations):

1. SHAP summary plot (beeswarm): shows feature impact on model output
2. SHAP waterfall plots: explain individual predictions (TP and FP)
3. Feature importance comparison: SHAP vs native XGBoost importance
4. Save all plots to outputs/

Run after: 07_advanced_modeling.py (best model must exist)
Outputs:   09_shap_summary.png
           10_shap_waterfall.png
           11_feature_importance_v2.png
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import joblib
import warnings
warnings.filterwarnings("ignore")

import shap
from sklearn.preprocessing import StandardScaler

BASE = Path(__file__).resolve().parent.parent
MODELS = BASE / "models"
OUT = BASE / "outputs"
OUT.mkdir(exist_ok=True)

RANDOM_STATE = 42


def load_data_and_models():
    """Load feature matrix, scaler, and best model."""
    feature_data = joblib.load(MODELS / "feature_matrix.joblib")
    scaler = joblib.load(MODELS / "scaler_v2.joblib")
    best_model = joblib.load(MODELS / "at_risk_classifier.joblib")
    
    X = feature_data["X"]
    y = feature_data["y_binary"]
    
    # Scale features
    X_scaled = scaler.transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
    
    return X_scaled, y, best_model, feature_data


def get_explainer_and_shap_values(model, X, sample_size=100):
    """Create SHAP explainer and compute SHAP values."""
    print("Creating SHAP explainer (this may take a minute)...")
    
    # Use a sample for faster computation
    if len(X) > sample_size:
        sample_idx = np.random.RandomState(RANDOM_STATE).choice(len(X), sample_size, replace=False)
        X_sample = X.iloc[sample_idx]
    else:
        X_sample = X
    
    try:
        # Try TreeExplainer first (works for tree-based models)
        print("Using TreeExplainer...")
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sample)
    except Exception as e:
        print(f"TreeExplainer failed: {e}")
        print("Using permutation-based SHAP (slower but works with any model)...")
        
        # Create background data for SHAP (smaller sample for efficiency)
        background_size = min(50, len(X))
        background_idx = np.random.RandomState(RANDOM_STATE).choice(len(X), background_size, replace=False)
        X_background = X.iloc[background_idx]
        
        explainer = shap.Explainer(
            lambda x: model.predict_proba(pd.DataFrame(x, columns=X.columns))[:, 1],
            X_background,
            check_additivity=False
        )
        shap_values = explainer(X_sample)
    
    # Handle different SHAP output formats
    if hasattr(shap_values, 'values'):
        # New SHAP API returns Explanation object
        shap_values_array = shap_values.values
    elif isinstance(shap_values, list):
        # Old API might return list for multiclass
        shap_values_array = shap_values[1] if len(shap_values) > 1 else shap_values[0]
    else:
        shap_values_array = shap_values
    
    return explainer, shap_values_array, X_sample


def plot_shap_summary(explainer, shap_values, X_sample, feature_cols):
    """Plot SHAP summary plot (beeswarm)."""
    print("Plotting SHAP summary plot...")
    
    plt.figure(figsize=(10, 8))
    shap.summary_plot(
        shap_values, X_sample,
        feature_names=feature_cols,
        plot_type="dot",
        show=False
    )
    plt.title("SHAP Summary Plot: Feature Impact on Model Output\n(Positive = increases risk of ban)")
    plt.tight_layout()
    plt.savefig(OUT / "09_shap_summary.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved SHAP summary to {OUT / '09_shap_summary.png'}")


def plot_shap_waterfall_examples(explainer, shap_values, X_sample, y_sample, feature_cols):
    """Plot SHAP waterfall for individual predictions (TP, FP)."""
    print("Plotting SHAP waterfall examples...")
    
    # Get predictions
    try:
        y_pred_proba = explainer.model.predict_proba(X_sample)[:, 1]
    except:
        y_pred_proba = explainer.model.predict(X_sample)
    
    y_pred = (y_pred_proba >= 0.5).astype(int)
    
    # Find a true positive and false positive
    tp_idx = np.where((y_sample == 1) & (y_pred == 1))[0]
    fp_idx = np.where((y_sample == 0) & (y_pred == 1))[0]
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # True Positive
    if len(tp_idx) > 0:
        idx = tp_idx[0]
        plt.sca(axes[0])
        shap.plots._waterfall.waterfall_legacy(
            shap.Explanation(
                shap_values[idx],
                base_values=explainer.expected_value if hasattr(explainer, 'expected_value') else 0.5,
                data=X_sample.iloc[idx],
                feature_names=feature_cols
            ),
            max_display=10
        )
        axes[0].set_title(f"TP: Risk Score={y_pred_proba[idx]:.3f} (Actual Ban=1)")
    
    # False Positive
    if len(fp_idx) > 0:
        idx = fp_idx[0]
        plt.sca(axes[1])
        shap.plots._waterfall.waterfall_legacy(
            shap.Explanation(
                shap_values[idx],
                base_values=explainer.expected_value if hasattr(explainer, 'expected_value') else 0.5,
                data=X_sample.iloc[idx],
                feature_names=feature_cols
            ),
            max_display=10
        )
        axes[1].set_title(f"FP: Risk Score={y_pred_proba[idx]:.3f} (Actual Ban=0)")
    
    plt.tight_layout()
    plt.savefig(OUT / "10_shap_waterfall.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved SHAP waterfall to {OUT / '10_shap_waterfall.png'}")


def plot_feature_importance_comparison(shap_values, X_sample, feature_cols):
    """Compare SHAP-based feature importance vs absolute mean."""
    print("Plotting feature importance comparison...")
    
    # SHAP importance: mean absolute SHAP value per feature
    shap_importance = np.abs(shap_values).mean(axis=0)
    shap_importance_df = pd.DataFrame({
        "feature": feature_cols,
        "importance": shap_importance
    }).sort_values("importance", ascending=False)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(range(len(shap_importance_df)), shap_importance_df["importance"], color="steelblue")
    ax.set_yticks(range(len(shap_importance_df)))
    ax.set_yticklabels(shap_importance_df["feature"], fontsize=9)
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title("SHAP Feature Importance (Top 20 features)")
    ax.invert_yaxis()
    
    # Show only top 20
    if len(shap_importance_df) > 20:
        ax.set_ylim([20, -1])
    
    plt.tight_layout()
    plt.savefig(OUT / "11_feature_importance_v2.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved feature importance to {OUT / '11_feature_importance_v2.png'}")
    
    # Print top features
    print("\nTop 10 Features by SHAP Importance:")
    for i, row in shap_importance_df.head(10).iterrows():
        print(f"  {row['feature']:35s}  {row['importance']:.4f}")
    
    return shap_importance_df


def main():
    print("Loading data and models...")
    X, y, best_model, feature_data = load_data_and_models()
    
    feature_cols = feature_data["all_feature_cols"]
    
    print(f"Data shape: {X.shape}")
    print(f"Model type: {type(best_model).__name__}")
    
    # --- Get SHAP explainer and values ---
    explainer, shap_values, X_sample = get_explainer_and_shap_values(best_model, X, sample_size=200)
    y_sample = y.iloc[X_sample.index]
    
    # --- Generate plots ---
    print("\n" + "=" * 70)
    print("GENERATING SHAP VISUALIZATIONS")
    print("=" * 70)
    
    plot_shap_summary(explainer, shap_values, X_sample, feature_cols)
    plot_shap_waterfall_examples(explainer, shap_values, X_sample, y_sample, feature_cols)
    shap_importance_df = plot_feature_importance_comparison(shap_values, X_sample, feature_cols)
    
    # --- Summary ---
    print("\n" + "=" * 70)
    print("SHAP EXPLAINABILITY COMPLETE")
    print("=" * 70)
    print("\nGenerated outputs:")
    print(f"  - {OUT / '09_shap_summary.png'} (Feature impact across all predictions)")
    print(f"  - {OUT / '10_shap_waterfall.png'} (Individual prediction breakdown)")
    print(f"  - {OUT / '11_feature_importance_v2.png'} (Top features by SHAP importance)")
    print("\nKey insights:")
    print(f"  - Most important feature: {shap_importance_df.iloc[0]['feature']}")
    print(f"  - Features explain model decisions through additive contributions")
    print(f"  - Use waterfall plots to understand specific at-risk author flags")


if __name__ == "__main__":
    main()
