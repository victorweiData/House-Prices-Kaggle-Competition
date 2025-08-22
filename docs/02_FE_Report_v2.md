# 02 — Feature Engineering Report (v2)

**Scope:** Add targeted features to reduce neighborhood-driven bias and fold variability observed in modeling residuals. All features are built to be CV-safe and modeling-friendly for both tree and linear/NN families.

---

## 1) Inputs & Outputs

### Inputs
- `data/processed/hp_train_feat_v01.parquet`
- `data/processed/hp_test_feat_v01.parquet`
- `data/processed/cv_folds_selected.csv` (StratifiedKFold on log target)
- `data/processed/hp_clean_meta_v02.json` (prior metadata)

### Outputs
**Feature data (v2):**
- `data/processed/hp_train_feat_v02.csv`
- `data/processed/hp_test_feat_v02.csv`

**Updated metadata:**
- `data/processed/hp_clean_meta_v03.json`
- Adds `engineered_continuous`, `engineered_nominal`, `engineered_binary` (v2 lists)
- Records `feature_version="v02"` and `folds_file`

**Note:** CSV chosen for interoperability. If you want faster IO & smaller files, keep Parquet copies alongside (pros: speed/size; con: needs PyArrow to read).

---

## 2) Motivation (what the diagnostics showed)

Residual diagnostics from the tuned LGBM OOFs revealed systematic underprediction (positive residuals) concentrated in specific neighborhoods across multiple folds—especially StoneBr, NoRidge, NridgHt, Veenker, ClearCr (luxury clusters), and noisier behavior in Edwards/OldTown/BrkSide. A Neighborhood×Fold heatmap and per-neighborhood residual plots guided the feature set below to capture location premiums, size/quality interactions by location, and relative-to-neighborhood normalization.

---

## 3) Feature Set (v2)

### 3.1 CV-safe target mean encoding (location premium)
- **`Nbhd_TgtMean_Log_v2`**
- **Definition:** out-of-fold mean of log(SalePrice) per Neighborhood for train; full-train map for test.
- **Why:** Encodes neighborhood price premium directly while avoiding leakage.
- **Leakage guard:** For each fold k, map comes only from training folds (not the held-out fold). Unseen neighborhoods default to the global log-mean.

### 3.2 Interactions with neighborhood premium
- **`TotalSF_log1p_v2`** = log1p(TotalSF)
- **`Qual_x_NbhdPrem_v2`** = OverallQual × Nbhd_TgtMean_Log_v2
- **`LogTotalSF_x_NbhdPrem_v2`** = log1p(TotalSF) × Nbhd_TgtMean_Log_v2
- **Why:** Size/quality carry different price gradients across neighborhoods; these capture location-dependent slopes and tame skew in luxury areas.

### 3.3 Relative-to-neighborhood features
- **`OverallQual_relative_v2`** = OverallQual / median(OverallQual | Neighborhood)
- **`TotalSF_relative_v2`** = TotalSF / median(TotalSF | Neighborhood)
- **Why:** Normalizes each house against its local baseline; reduces systematic over/under-prediction by neighborhood.
- **Safety:** Medians computed on train only and mapped; zeros handled; inf/NaN filled to 1.0 (neutral).

### 3.4 Categorical interactions (low-leak, feature-only)
- **`Nbhd_Qual_cat_v2`** = f"{Neighborhood}_{OverallQual}"
- **`YearBuiltDecade_v2`** = floor(YearBuilt/10)*10
- **`Nbhd_Decade_cat_v2`** = f"{Neighborhood}_{YearBuiltDecade_v2}"
- **Why:** Captures style/era effects that vary by location.
- **Dtype guidance:** Cast to category for tree models.

### 3.5 Neighborhood clustering (feature-only)
- **`NbhdCluster4_v2`**
- KMeans on neighborhood-level medians: OverallQual, TotalSF, YearBuilt (train only).
- **Why:** Groups similar neighborhoods to share signal when individual categories are sparse; helps luxury mid/low splits.
- **Reproducibility:** random_state=42 stored in code.

### 3.6 Luxury flags
- **`is_luxury_nbhd_v2`** = 1 if Neighborhood ∈ {StoneBr, NoRidge, NridgHt, Veenker, ClearCr}
- **`is_luxury_v2`** = 1 if (OverallQual ≥ 9) or (is_luxury_nbhd_v2==1)
- **Why:** Explicitly marks fat-tail regimes where elasticities differ.

### 3.7 Extra log transforms (skew control)
- **`LotArea_log1p_v2`**, **`GrLivArea_log1p_v2`**, **`BsmtFinSF1_log1p_v2`**
- **Why:** Stabilize heavy-tailed scale features; important in high-end neighborhoods.

---

## 4) Dtypes & Encoding Recommendations

**Tree models (LGBM/XGB/Cat):**
- Keep new categorical features as category (native handling in LGBM/Cat; XGB: either OHE or categorical if using recent builds).
- Numeric features stay numeric; no scaling needed.

**Linear/NN family:**
- OHE the string/constructed categorical (`Nbhd_Qual_cat_v2`, `Nbhd_Decade_cat_v2`, `NbhdCluster4_v2`) with min_frequency to control cardinality.
- Keep log-transformed/ratio features as is; standardize numerics for NN.

---

## 5) Leakage Control & Reproducibility

- Out-of-fold mapping used for `Nbhd_TgtMean_Log_v2` on train.
- No target usage in clusters/relative features (feature-only stats).
- **Folds:** `cv_folds_selected.csv` (stratified on log target) used consistently.
- **Seeds:** KMeans(random_state=42); LightGBM seeds logged during modeling.
- **Null/Inf handling:** unseen neighborhood → global log-mean; ratio inf/NaN → 1.0.

---

## 6) Pros / Cons

### Pros
- Targets the exact failure modes seen in residuals (luxury neighborhoods; location-dependent elasticities).
- CV-safe target encoding preserves signal without leakage.
- Relative features and interactions improve robustness across folds.

### Cons / Risks
- Higher feature count (especially categorical interactions) → potential overfitting if not regularized (monitor with early stopping & CV).
- Cluster boundary sensitivity (KMeans) → keep the seed stable; consider 3–6 clusters in future iterations.
- Target mean encoding must always be OOF in training pipelines (already handled here).

---

## 7) What changed in metadata

`hp_clean_meta_v03.json` updates:
- **`feature_version`:** "v02"
- **`engineered_continuous`** (appended):
  - `Nbhd_TgtMean_Log_v2`, `TotalSF_log1p_v2`, `Qual_x_NbhdPrem_v2`, `LogTotalSF_x_NbhdPrem_v2`,
  - `OverallQual_relative_v2`, `TotalSF_relative_v2`,
  - `LotArea_log1p_v2`, `GrLivArea_log1p_v2`, `BsmtFinSF1_log1p_v2`
- **`engineered_nominal`** (appended):
  - `Nbhd_Qual_cat_v2`, `Nbhd_Decade_cat_v2`, `NbhdCluster4_v2`
- **`engineered_binary`** (appended):
  - `is_luxury_nbhd_v2`, `is_luxury_v2`
- **`folds_file`:** path to selected fold CSV

---

## 8) QA Checklist (passed)

- No NaN/inf in new numerics after fills.
- New categorical columns exist in both train/test with consistent levels (unseen mapped/filled).
- Distributions look sane (no degenerate constants).
- File sizes and column counts match expectations between train/test (except target column).

---

## 9) Next Steps

1. **Retrain LGBM** with v2 features (same folds), log CV and feature importances; compare to v1.
2. **If gains hold,** propagate to XGB/Cat; keep OOFs for ensembling.
3. **Re-run residual heatmaps;** verify problem neighborhoods improved.
4. **Consider more interactions** if needed:
   - Neighborhood × GarageCars, Neighborhood × KitchenQual,
   - Age bins × Neighborhood, Baths normalized by TotalSF.
5. **If linear/NN used:** finalize OHE choices and scaling; check sparsity vs. performance.

---

## 10) Appendix — Column Reference

### Continuous
- `Nbhd_TgtMean_Log_v2`, `TotalSF_log1p_v2`, `Qual_x_NbhdPrem_v2`, `LogTotalSF_x_NbhdPrem_v2`,
- `OverallQual_relative_v2`, `TotalSF_relative_v2`,
- `LotArea_log1p_v2`, `GrLivArea_log1p_v2`, `BsmtFinSF1_log1p_v2`

### Nominal / Categorical
- `Nbhd_Qual_cat_v2`, `YearBuiltDecade_v2`, `Nbhd_Decade_cat_v2`, `NbhdCluster4_v2`

### Binary flags
- `is_luxury_nbhd_v2`, `is_luxury_v2`