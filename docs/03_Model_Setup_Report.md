# 03 — Modeling Setup Report

**Scope:** Build a reproducible modeling scaffold for the House Prices (Kaggle) project: fixed CV folds, shared preprocessing recipes, and baseline models. Also compare CV schemes and pick a project-default.

---

## 1) Inputs & Versions

**Feature data**
- `data/processed/hp_train_feat_v01.parquet`
- `data/processed/hp_test_feat_v01.parquet`

**Metadata**
- `data/processed/hp_clean_meta_v02.json`
  (column groups, engineered features, transform plans)

**Artifacts directories**
- `artifacts/oof/`, `artifacts/submissions/`

**Notebook(s) covered**
- `03_modeling_setup.ipynb` (baselines, folds)
- `03b_cv_comparison.ipynb` (CV strategy comparison)

---

## 2) Target & Metric

- **Training target:** log(SalePrice)
- **Metric:** RMSE on log(SalePrice)

This matches Kaggle's evaluation: "RMSE between log(pred) and log(actual)".
We also compute RMSE on the original SalePrice scale for intuition (not used for model selection).

---

## 3) Cross-Validation Design

### 3.1 Primary CV (selected)
- **Splitter:** StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
- **Stratification:** deciles of log(SalePrice)
- **Saved as:** `data/processed/cv_folds_v01.csv`
- **Project default (frozen):** `data/processed/cv_folds_selected.csv` (copy of v01)

**Why selected:** best performance and stable behavior in our baseline experiments (see §4.3).

### 3.2 Stress-test CV
- **GroupKFold** by Neighborhood (simulate unseen neighborhoods).
- Used occasionally to assess robustness to location shift.

---

## 4) Baseline Models & Results

### 4.1 Preprocessing recipes

**Linear / NN family (for ElasticNet baseline)**
- ColumnTransformer:
  - log1p on planned continuous features (from metadata)
  - Yeo–Johnson on selected continuous features
  - Pass-through remaining continuous + ordinal-int columns
  - One-Hot Encoding for string categoricals (ignore unknowns, min_frequency=10)
- StandardScaler(with_mean=False) post-OHE
- ElasticNet (alpha=0.1, l1_ratio=0.5, max_iter=2000, random_state=42)

**Tree family (LightGBM baseline)**
- Raw numerics; no scaling
- Pandas category dtypes for categoricals (native handling)
- Early stopping on validation fold
- Baseline params: learning_rate=0.05, n_estimators=5000, num_leaves=31, subsample=0.8, colsample_bytree=0.8, reg_lambda=0.0, random_state=42

### 4.2 5-Fold CV (log-RMSE)

| model | cv_rmse_mean | cv_rmse_std |
|-------|--------------|-------------|
| LGBM_baseline_v01 | 0.09998 | 0.00426 |
| ElasticNet_baseline_v01 | 0.13041 | 0.00426 |

### 4.3 CV strategy comparison (original-scale RMSE, lower is better)

| model | cv_scheme | cv_rmse_original |
|-------|-----------|------------------|
| LGBM | stratlog | 21,298.66 |
| LGBM | random | 21,753.51 |
| LGBM | group_nbhd | 25,706.37 |
| ElasticNet | random | 25,613.71 |
| ElasticNet | stratlog | 25,692.98 |
| ElasticNet | group_nbhd | 27,175.76 |

**Decision:** Use StratifiedKFold on log(SalePrice) as the project-default CV, and keep GroupKFold by Neighborhood as a robustness check.

---

## 5) Saved Artifacts

**Folds**
- `data/processed/cv_folds_v01.csv` (stratified on log target)
- `data/processed/cv_folds_selected.csv` (project default)
- (Optional) `cv_folds_v02_alt_neigh.csv` (experimental group variant)

**OOF predictions (log target)**
- `artifacts/oof/elasticnet_baseline_v01_oof.csv`
- `artifacts/oof/lgbm_baseline_v01_oof.csv`

**Submissions (expm1 back-transform)**
- `artifacts/submissions/elasticnet_baseline_v01.csv`
- `artifacts/submissions/lgbm_baseline_v01.csv`

---

## 6) MLflow (recommended)

If enabled, each run logs:
- **Params:** model family, seed, CV scheme, key hyperparams, data/meta versions
- **Metrics:** fold scores, mean±std CV (log-RMSE)
- **Artifacts:** OOF CSV, submission CSV, config YAML
- **Tags:** notebook path, git commit

This makes comparing families and tuning runs straightforward.

---

## 7) Reproducibility & Guardrails

- One fixed fold file used across models for fair comparisons.
- All preprocessing happens inside folds (fit on train fold only).
- No target leakage (e.g., target encodings deferred to modeling and done fold-wise).
- Report both log-RMSE (selection metric) and original-scale RMSE (intuition).
- Keep GroupKFold checks for robustness to location shift.

---

## 8) Next Steps

1. **Tune LightGBM** (`04b_lgbm.ipynb`) on the selected folds:
   sweep num_leaves, min_data_in_leaf, feature_fraction, bagging_fraction, lambda_l1/l2, learning_rate; keep early stopping.
2. **Add XGBoost and CatBoost** baselines; save OOFs with the same folds.
3. **Blend/Stack** using OOF predictions (start with weighted average; then try Ridge/ElasticNet stacker).
4. **Re-check finalists** under GroupKFold to ensure robustness.

---

## Appendix — Design Choices

- Log target for all families (aligns with Kaggle).
- String-only OHE; numeric-coded categoricals remain numeric/ordinal.
- Early stopping for boosters.
- Stratified CV chosen by evidence; group CV used as a stress test.