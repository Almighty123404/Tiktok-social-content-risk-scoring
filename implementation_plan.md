# Trust & Safety Project V2 — Upgrade Implementation Plan

## Background

The existing project has a solid SQL-first analytical foundation but a weak ML layer: the classifier achieves ROC-AUC = 0.704 with feature importance dominated by a single binary feature (`is_claim ≈ 0.9`), no cross-validation, no hyperparameter tuning, and no use of the `video_transcription_text` column. This plan upgrades the project across five areas to make it resume-differentiating.

---

## Key Research Finding: Text Column Structure

> [!IMPORTANT]
> The `video_transcription_text` column has **highly structured prefixes** that almost perfectly predict `claim_status`:
> - **Claims** start with phrases like "someone read in the media", "i learned from the media", "a friend read in the media"
> - **Opinions** start with phrases like "my colleagues' understanding is", "my family is willing to wager", "i think that"
>
> This means NLP features extracted from transcription text will likely be **highly correlated with `is_claim`** and may not provide orthogonal signal for predicting ban status. The implementation will explicitly test and document this — an honest negative result showing "NLP features are largely proxies for claim_status" is itself a valuable analytical insight.

---

## Proposed Changes

### Phase 1: Data Leakage / Validity Audit

#### [NEW] [05_leakage_audit.py](file:///c:/Users/at727/Downloads/frnd_trust_safety_project/frnd_trust_safety_project/notebooks/05_leakage_audit.py)

Statistical investigation of whether engagement metrics are plausible pre-ban signals or post-ban artifacts:
- Compare engagement distributions (views, likes, shares) across ban statuses using Mann-Whitney U tests
- Check for suspicious truncation patterns (e.g., many banned videos with exactly 0 views/shares suggesting post-ban suppression)
- Analyze whether engagement metric distributions for banned authors look "natural" or "censored"
- Test for engagement metric value overlap between classes
- Document findings as a limitation section in the README regardless of outcome

**Key question**: If engagement is post-ban (frozen/suppressed), the model would be learning from the *outcome* rather than predicting it — making the recall numbers misleading.

---

### Phase 2: NLP Feature Engineering

#### [NEW] [06_nlp_features.py](file:///c:/Users/at727/Downloads/frnd_trust_safety_project/frnd_trust_safety_project/notebooks/06_nlp_features.py)

Extract features from `video_transcription_text` and test whether they add predictive value beyond `is_claim`:
1. **TF-IDF features**: Top 100-200 unigrams/bigrams → SVD/truncated dimensionality reduction to ~20 components
2. **Hand-engineered text signals**:
   - Text length (character count, word count)
   - Exclamation/question mark density
   - Numeric claim density (count of numbers in text)
   - Superlative count ("most", "largest", "best", "greatest")
   - Hedging phrase presence ("i think", "i believe", "my opinion")
   - Claim-triggering phrase presence ("someone shared", "a friend", "in the media")
   - Sentiment-related word counts
3. **Correlation analysis**: measure correlation between NLP features and `is_claim` to quantify redundancy
4. Save enriched feature matrix for use in advanced modeling

---

### Phase 3: Advanced Modeling

#### [NEW] [07_advanced_modeling.py](file:///c:/Users/at727/Downloads/frnd_trust_safety_project/frnd_trust_safety_project/notebooks/07_advanced_modeling.py)

Complete modeling overhaul:

**Cross-validation**: 5-fold stratified CV with mean ± std for ROC-AUC, recall, and precision across folds (not just a single split)

**Hyperparameter tuning**: RandomizedSearchCV for Random Forest and XGBoost with defined parameter grids

**Class imbalance techniques**:
- Current: `class_weight="balanced"` (baseline)
- SMOTE oversampling
- SMOTE-Tomek combined
- Compare all three and report honestly if SMOTE doesn't help

**Ensemble methods**:
- Current soft-voting ensemble (baseline)
- Stacking ensemble with Logistic Regression meta-learner on top of RF + XGBoost base models

**Multiclass model**: Active vs Under Review vs Banned as a secondary model — compare business usefulness (e.g., can the model distinguish "under review" from "banned"?)

**Threshold optimization**:
- Plot precision-recall curve
- Define business cost assumption: cost of missing an at-risk author (false negative) vs cost of a false flag (false positive) — use a 5:1 cost ratio as a default
- Select optimal threshold based on F-beta or cost-minimization
- Report metrics at both default (0.5) and optimized thresholds

**Probability calibration**:
- Apply Platt scaling (sigmoid) and isotonic regression
- Plot reliability diagrams (calibration curves)
- Compare calibrated vs uncalibrated Brier scores

Outputs: All charts saved to `outputs/`; updated model saved to `models/`

---

### Phase 4: SHAP Explainability

#### [NEW] [08_shap_explainability.py](file:///c:/Users/at727/Downloads/frnd_trust_safety_project/frnd_trust_safety_project/notebooks/08_shap_explainability.py)

- SHAP summary plot (beeswarm) showing feature contributions across all predictions
- SHAP waterfall/force plot for 1-2 individual predictions (one true positive, one false positive)
- SHAP feature importance comparison vs native XGBoost importance
- Save all plots to `outputs/`

---

### Phase 5: Charts & Deliverables

#### Output Charts (all to `outputs/`)
| File | Content |
|------|---------|
| `07_cv_roc_auc_comparison.png` | Cross-validated ROC-AUC comparison (bar chart with error bars) |
| `08_precision_recall_curve.png` | PR curve with chosen threshold marked |
| `09_shap_summary.png` | SHAP beeswarm summary plot |
| `10_shap_waterfall.png` | SHAP waterfall for individual prediction |
| `11_feature_importance_v2.png` | Updated feature importance with NLP features |
| `12_calibration_curve.png` | Reliability diagram |
| `13_threshold_analysis.png` | Cost-based threshold selection plot |
| `14_multiclass_confusion.png` | Multiclass confusion matrix |

#### [MODIFY] [README.md](file:///c:/Users/at727/Downloads/frnd_trust_safety_project/frnd_trust_safety_project/README.md)

Complete rewrite with:
- Version 2 changelog (old vs new metrics side by side)
- Data leakage audit findings
- NLP feature analysis results
- Business cost threshold reasoning
- Tightened "Business Takeaway" section
- Resume bullet (under 30 words)

---

## Verification Plan

### Automated Tests
- Run all scripts in sequence: `05_leakage_audit.py` → `06_nlp_features.py` → `07_advanced_modeling.py` → `08_shap_explainability.py`
- Verify all output PNGs and updated model are generated
- Verify all claimed metrics are reproducible (pinned random seeds)

### Manual Verification
- Review all output charts for visual quality
- Confirm README metrics match actual script output
- Verify resume bullet accurately reflects measured performance

---

## Dependencies to Install

```
pip install shap imbalanced-learn
```

(scikit-learn, xgboost, pandas, matplotlib, numpy, joblib, scipy, optuna, tabulate are already installed)

---

## Open Questions

> [!NOTE]
> **NLP feature redundancy**: Given the text prefixes almost perfectly predict `claim_status`, NLP features may provide minimal incremental lift. The plan accounts for this by explicitly testing and documenting the outcome either way — an honest negative result is more resume-credible than omitting the analysis.

> [!NOTE]
> **Cost ratio for threshold optimization**: I'll use a default 5:1 ratio (cost of missing at-risk : cost of false flag) as a reasonable trust & safety assumption. If you have a different ratio in mind, let me know.

> [!NOTE]
> **SMOTE with text features**: If TF-IDF SVD components are included, SMOTE will operate in a high-dimensional mixed space. I'll test both with and without NLP features for SMOTE comparisons to ensure fair evaluation.
