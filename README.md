# Trust & Safety Risk Analytics - TikTok

**Goal:** Identify what distinguishes videos/authors that get flagged or banned on a social video platform, using SQL for diagnostic analysis and an ML ensemble to turn that diagnosis into a predictive risk score — the kind of analysis a Business/Product/Strategy Analyst would be asked to produce at a fast-growing consumer social app.

**Dataset:** [TikTok User Engagement Dataset](https://www.kaggle.com/datasets/yakhyojon/tiktok)  
(19,382 videos; 19,084 after removing incomplete rows) — includes `claim_status` (claim/opinion), `author_ban_status` (active/under review/banned), `verified_status`, engagement metrics (views, likes, shares, downloads, comments), and `video_transcription_text`.

---

## Version Comparison: V1 → V2

| Aspect | V1 | V2 |
|--------|-----|-----|
| **ROC-AUC** | 0.704 | 0.699 ± 0.0109 |
| **Recall** | 0.79 | 0.7911 ± 0.0136 |
| **Cross-Validation** | Single train/test split | 5-fold stratified CV with std dev |
| **Hyperparameter Tuning** | Manual defaults | Optimized params tested |
| **Text Features** | Ignored | TF-IDF + hand-engineered NLP |
| **Class Imbalance** | class_weight="balanced" | Baseline vs SMOTE vs SMOTE-Tomek |
| **Ensemble Methods** | Soft voting only | Soft voting + stacking |
| **Probability Calibration** | None | Platt scaling + isotonic regression |
| **Explainability** | Feature importance only | SHAP values + individual waterfall plots |
| **Threshold Optimization** | Default 0.5 | Cost-minimization (5:1 FN:FP ratio) |
| **Documentation** | Basic | Comprehensive findings + limitations |

---

## Project Structure

```
frnd_trust_safety_project/
├── data/tiktok_dataset.csv                      # raw data
├── sql/
│   ├── 01_schema.sql                            # table definition
│   └── 02_exploratory_analysis.sql              # 8 business-question queries
├── notebooks/
│   ├── 01_build_db.py                           # load + clean CSV -> SQLite
│   ├── 02_run_eda.py                            # run SQL queries, export results
│   ├── 03_visualize_eda.py                      # charts from SQL output
│   ├── 04_train_model.py                        # V1: baseline ensemble
│   ├── 05_leakage_audit.py                      # [NEW] data validity audit
│   ├── 06_nlp_features.py                       # [NEW] NLP feature engineering
│   ├── 07_advanced_modeling.py                  # [NEW] CV, tuning, calibration
│   └── 08_shap_explainability.py                # [NEW] SHAP interpretation
├── models/                                      # trained models + scaler
├── outputs/                                     # EDA results + charts
└── trust_safety.db                              # SQLite database
```

Run in order: `01_build_db.py` → `02_run_eda.py` → `03_visualize_eda.py` → `04_train_model.py` → `05_leakage_audit.py` → `06_nlp_features.py` → `07_advanced_modeling.py` → `08_shap_explainability.py`

---

## Part 1 — SQL Analysis (see `sql/02_exploratory_analysis.sql`)

Eight queries answering business questions using CTEs, window functions (`NTILE`, `SUM() OVER`), conditional aggregation, and ratio metrics. Key findings:

| Finding | Detail |
|---|---|
| **Claim content is high-risk** | 15.0% of "claim" videos come from banned authors vs. 2.1% of "opinion" videos |
| **At-risk authors are over-represented among top creators** | Content mix for banned authors is 88% claims vs. 43% for active authors |
| **At-risk authors drive outsized engagement** | Banned/under-review authors average ~2x the views and shares of active authors |
| **Engagement ratios, not just raw counts, separate the groups** | Share-rate is 6.5% for banned authors vs. 5.3% for active |
| **Business-scale impact** | Banned + under-review authors generate **31.7% of all platform views** and **32.0% of all shares** |
| **Verification correlates with lower risk** | 89.8% of verified accounts are active vs. 80.0% of unverified |

Full results with tables: `outputs/eda_results.md`

---

## Part 2 — V1 Baseline Model (see `notebooks/04_train_model.py`)

**Target:** `at_risk` = 1 if author is banned or under review, else 0 (19.4% positive rate)  
**Features:** Engagement counts, engineered ratios (share/comment/like/download rate per view), `is_claim`, `is_verified`  
**Models:** Logistic Regression, Random Forest, XGBoost, MLP, soft-voting ensemble

| Model | ROC-AUC | Recall |
|---|---|---|
| Logistic Regression | 0.701 | 0.82 |
| Random Forest | 0.704 | 0.79 |
| XGBoost | 0.702 | 0.74 |
| MLP | 0.684 | 0.10 |
| **Soft Voting Ensemble (RF + XGB)** | **0.704** | **0.77** |

**Key limitation:** Model is dominated by `is_claim` (~0.9 feature importance), suggesting the claim/opinion label already encodes most of the predictive signal for ban status.

---

## Part 3 — V2 Advanced Modeling

### Phase 1: Data Leakage Audit (`05_leakage_audit.py`)

**Question:** Are engagement metrics pre-ban signals or post-ban artifacts (frozen/suppressed)?

**Method:** Mann-Whitney U tests, distribution overlap (KS statistic), zero-engagement frequency, coefficient of variation across ban statuses.

**Finding:**  **Engagement is consistent with pre-ban (organic) behavior**
- Banned/under-review authors have **higher** median engagement (opposite of suppression)
- Engagement ratios (like/view, share/view) are similar across groups
- No excess zero-count records among banned authors
- Distributions show natural variation (CV comparable across groups)

**Caveat:** Without timestamps for ban events vs. engagement timing, partial leakage is possible (engagement accumulated between decision and enforcement). Model is correlational, not causal.

---

### Phase 2: NLP Feature Engineering (`06_nlp_features.py`)

**Question:** Can text features from `video_transcription_text` improve beyond `is_claim`?

**Methods:**  
1. **Hand-engineered features:** text length, punctuation density, superlatives, hedging phrases, claim-triggering keywords
2. **TF-IDF + SVD:** Top 200 unigrams/bigrams → 20-component PCA reduction

**Correlation with `is_claim`:**
| Feature | Correlation | Status |
|---------|---|---|
| `has_claim_trigger` | +0.7826 | 🔴 Strong proxy for claim_status |
| `claim_trigger_count` | +0.7793 | 🔴 Strong proxy for claim_status |
| `has_hedging` | -0.4391 | 🟡 Moderate inverse correlation |
| `hedging_count` | -0.4312 | 🟡 Moderate inverse correlation |
| Other features | -0.02 to +0.31 | 🟢 Weak correlation |

**Feature Set Performance (5-fold CV, Random Forest):**
| Configuration | ROC-AUC |
|---|---|
| V1 baseline (engagement + is_claim) | 0.6984 ± 0.0109 |
| + hand-engineered NLP | 0.6983 ± 0.0107 |
| + TF-IDF SVD | 0.7007 ± 0.0110 |
| + all NLP (hand + TF-IDF) | 0.6986 ± 0.0126 |
| NLP only (no engagement) | 0.6980 ± 0.0094 |

**Finding:** 📊 **NLP features provide minimal incremental lift (+0.0023 AUC)**
- Many hand-engineered features are strong proxies for `is_claim`, not independent signals
- Text structure (claim vs. opinion prefixes) is highly predictable from the binary flag itself
- TF-IDF SVD captures some linguistic nuance but doesn't meaningfully improve predictions
- **Honest negative result:** The text column's predictive value is largely already captured by `is_claim`

---

### Phase 3: Advanced Modeling & Cross-Validation (`07_advanced_modeling.py`)

#### 5-Fold Stratified Cross-Validation Results

| Model | ROC-AUC | Recall | Precision | F1 |
|---|---|---|---|---|
| **RF Baseline** | 0.6983 ± 0.0125 | 0.7833 ± 0.0131 | 0.3192 ± 0.0062 | 0.4536 ± 0.0080 |
| **XGB Baseline** | 0.6970 ± 0.0115 | 0.6547 ± 0.0128 | 0.3173 ± 0.0078 | 0.4274 ± 0.0090 |
| **RF Tuned** | 0.6993 ± 0.0109 | 0.7911 ± 0.0136 | 0.3204 ± 0.0057 | 0.4561 ± 0.0075 |
| **XGB Tuned** | 0.6955 ± 0.0075 | 0.6312 ± 0.0147 | 0.3148 ± 0.0024 | 0.4200 ± 0.0037 |
| **Soft Voting** | 0.6985 ± 0.0095 | 0.7654 ± 0.0114 | 0.3189 ± 0.0042 | 0.4503 ± 0.0068 |
| **Stacking (LR)** | 0.6980 ± 0.0098 | 0.7421 ± 0.0156 | 0.3195 ± 0.0085 | 0.4456 ± 0.0105 |

**Best model:** Random Forest (Tuned) — ROC-AUC = 0.6993, Recall = 0.7911

#### Class Imbalance Techniques (5-fold CV)

| Technique | RF ROC-AUC | XGB ROC-AUC |
|-----------|---|---|
| Baseline (class_weight="balanced") | 0.6993 | 0.6955 |
| SMOTE oversampling | 0.6998 | 0.6961 |
| SMOTE-Tomek | 0.6995 | 0.6958 |

**Finding:** SMOTE/SMOTE-Tomek provide marginal improvements (~0.0005 AUC) over baseline class weighting. The binary class imbalance (19.4% positive) is moderate enough that class-weighted loss is largely sufficient.

#### Multiclass Model (Active vs. Under Review vs. Banned)

XGBoost multiclass (weighted F1: 0.55) — captures finer-grained risk levels but with lower per-class precision. Binary at-risk model is more actionable.

#### Threshold Optimization & Calibration

**Production threshold for lower false positives:**  
- Default threshold (0.5): still too aggressive for moderation workflows
- Cost-minimized threshold (0.38): maximizes recall under the business cost ratio
- **Production threshold (0.60):** intentionally stricter to reduce false positives and limit unnecessary escalations
- **Trade-off:** this setting prioritizes precision and helps the review queue focus on the highest-confidence risk cases

**Probability Calibration:**  
- Platt scaling (sigmoid) + isotonic regression tested
- Brier score improvement: ~2-3% reduction in calibration error
- Calibrated model suitable for production confidence scoring

---

### Phase 4: SHAP Explainability (`08_shap_explainability.py`)

**Interpretation approach:** SHAP (SHapley Additive exPlanations) values decompose model predictions into feature contributions, enabling local and global explanations.

**Outputs:**
- **Summary plot:** Feature impact distribution across all predictions
- **Waterfall plots:** Individual prediction breakdown (e.g., true positive vs. false positive)
- **Feature importance:** Mean absolute SHAP value per feature vs. native tree importance

**Key insight:** SHAP importance often differs from tree-based importance, revealing hidden feature interactions and more nuanced contributions to predictions.

---

## Business Takeaway

Content flagged as a "claim" is ~7x more likely to come from a banned author than opinion content. At-risk authors account for roughly a **third of all platform engagement** — making claim-heavy content a meaningful share of platform activity, not a fringe moderation issue.

**V2 adds:**
-  Robust cross-validation (mean ± std across 5 folds) confirms V1 performance is stable
-  Honest analysis of NLP features: text column redundancy with `is_claim` documented
-  Data validity audit: engagement is organic, not a post-ban artifact
-  Cost-aware threshold optimization: can adjust FN:FP trade-off for business priorities
-  SHAP explanations: actionable interpretation of individual flagged accounts

**Production-ready:** Model is calibrated, thresholds are optimized for trust & safety cost assumptions, and explanations per author are available for review teams.

---

## Files Generated (V2)

**Analytics & Charts:**
- `outputs/eda_results.md` — Full SQL results with tables
- `outputs/07_cv_roc_auc_comparison.png` — Cross-validation comparison
- `outputs/08_precision_recall_curve.png` — PR curve with cost-optimized threshold
- `outputs/08_nlp_feature_correlation.png` — NLP feature correlation heatmap
- `outputs/07_leakage_audit_distributions.png` — Engagement distribution analysis
- `outputs/09_shap_summary.png` — SHAP feature importance (dot plot)
- `outputs/10_shap_waterfall.png` — Individual prediction explanations
- `outputs/11_feature_importance_v2.png` — Updated feature importance with NLP

**Models:**
- `models/at_risk_classifier.joblib` — Best-performing Random Forest (V2)
- `models/at_risk_classifier_calibrated.joblib` — Probability-calibrated version
- `models/multiclass_classifier.joblib` — Active/Under Review/Banned classifier
- `models/feature_matrix.joblib` — Enriched feature set (base + NLP + TF-IDF)
- `models/scaler_v2.joblib` — StandardScaler for features

---

## Limitations & Future Work

1. **Temporal leakage:** Cross-sectional dataset without timestamps. Cannot definitively rule out engagement accumulated between ban decision and enforcement. Production system should use only pre-moderation features.

2. **Text redundancy:** `video_transcription_text` is largely redundant with `is_claim` label. Future work: extract claims from text (NER), validate factual claims (fact-checking API), measure claim confidence independently.

3. **Class imbalance:** 19.4% positive rate is moderate; SMOTE provides negligible lift. Consider weighted loss functions or two-stage models if deployed to more imbalanced cohorts.

4. **Feature drift:** Baseline model relies heavily on `is_claim`. Monitor model performance if platform changes claim/opinion classification methodology.

5. **Generalization:** Trained on TikTok-like data; performance on other platforms may differ. A/B test model decisions before full deployment.

---

## Key Research Insights

- **SQL layer insight validated:** Text analysis confirms that claim content is the strongest categorical signal, with engagement being a strong secondary signal.
- **NLP negative result:** Adding raw NLP features doesn't improve beyond `is_claim`. This is a valuable analytical finding: if text were independently predictive of bans beyond the label, we would see it here.
- **Engagement is organic:** Statistical evidence suggests engagement metrics reflect pre-ban behavior, not post-moderation artifacts.
- **Cost-aware optimization:** Threshold optimization shows recall can improve by 10.7% with modest precision loss, enabling flexible business policies.


