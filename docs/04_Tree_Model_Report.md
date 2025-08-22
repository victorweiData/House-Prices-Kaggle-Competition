# 04 — Tree Models Report (LGBM / XGBoost / CatBoost)

## Scope
We train three tree ensembles on **Feature Set v02** using a **neighborhood×price stratified 5-fold** split to maximize stability and leaderboard transfer. Models predict on the **log price** scale and submissions are back-transformed.

---

## Data & Splits
- **Train/Test**: `hp_train_feat_v02.csv`, `hp_test_feat_v02.csv`
- **Folds**: `cv_folds_strat_nbhd_price_v01.csv`  
  - Stratification key = `nbhd_tier` (quartile of neighborhood median log price) × `price_bin` (decile of `SalePrice_log`)
  - Motivation: reduce fold volatility observed with random KFold; ensure each fold covers both location tiers and price ranges.

---

## Features (v02 highlights)
- **Raw numerics**: `GrLivArea`, `TotalBsmtSF`, `GarageCars`, `YearBuilt`, `...`
- **Log transforms** for skewed counts: `LotArea_log`, `WoodDeckSF_log`, `...`
- **Neighborhood-aware**:  
  - `Nbhd_TgtMean_Log_v2` (KFold target mean of `SalePrice_log` per `Neighborhood` — leakage-safe in train; global mean for test)
  - Relative scalers: `OverallQual_relative_v2`, `TotalSF_relative_v2` (value / neighborhood median)
- **Flags**: zero/has-feature indicators for porches, pools, etc.
- **Categoricals**: kept as categories (LGBM/XGB native handling), and as strings without NaN for CatBoost (“Missing” token).

> Note: numeric-coded categories (e.g., `MSSubClass`) remain numeric per our earlier decision.

---

## Evaluation
- **CV metric**: RMSE on `log(SalePrice)`
- **LB metric**: Kaggle RMSE on `log(SalePrice)`  
- **OOF logging**: `../artifacts/oof/*.csv`  
- **Submissions**: `../artifacts/submissions/*.csv`  
- **Tracking**: MLflow (`file:../mlruns`), Optuna studies (`sqlite:///../optuna_studies.db`)

---

## Models & Search Spaces

### LightGBM
- **Search**: Optuna Stage-A (≈60–80 trials), small-data bounds:
  - `num_leaves ∈ [16, 256] (log)`, `max_depth ∈ {-1,6…10}`, `min_data_in_leaf ∈ [10,200] (log)`
  - `feature_fraction, bagging_fraction ∈ [0.6,1.0]`, `bagging_freq ∈ {0,1,2}`
  - `lambda_l1, lambda_l2 ∈ [1e-4,30] (log)`, `min_gain_to_split ∈ [1e-4,1] (log)`
  - Stabilizers: `min_sum_hessian_in_leaf ∈ [1e-3,1e-1] (log)`, `cat_smooth ∈ [1,100] (log)`
- **Notes**: early stopping 200; native categorical via `category` dtype.

### XGBoost
- **Search**: depthwise + native categorical (≈60–80 trials):
  - `max_depth ∈ [4,10]`, `min_child_weight ∈ [0.5,32] (log)`
  - `learning_rate ∈ [1e-3,2e-1] (log)`
  - `subsample, colsample_bytree, colsample_bylevel ∈ [0.6,1.0]`
  - `reg_alpha, reg_lambda ∈ [1e-4,50] (log)`, `gamma ∈ [1e-4,10] (log)`
  - Categorical: `max_cat_to_onehot ∈ {16,32,64,128}`, `max_cat_threshold ∈ {32,64,128}`

### CatBoost
- **Search**: small-data tuned ranges (≈60–80 trials):
  - `learning_rate ∈ [0.02,0.12] (log)`, `depth ∈ [5,9]`, `l2_leaf_reg ∈ [1,30] (log)`
  - `subsample, rsm ∈ [0.6,1.0]`
  - Extras: `random_strength ∈ [0,2]`, `bagging_temperature ∈ [0,10]`
- **Preproc**: categorical columns converted to **strings** with `"Missing"` token; passed by **column index**.

---

## Results (current snapshot)

| Model run tag                                | CV RMSE (log) | LB RMSE (log) | Notes |
|---|---:|---:|---|
| `cat_v02_baseline_nbhdstrat`                 | *(from MLflow)* | **0.12442** | Best single so far |
| `lgbm_v02_optuna_nbhdstrat` *(Stage-A)*      | *(from MLflow)* | ~0.125–0.1255 | Similar to prior LGBM; stable folds |
| `xgb_v02_optuna_nbhdstrat`                   | *(from MLflow)* | *(submit)* | Baseline diversity |
| `stack_convex_nbhdstrat_lgbm_xgb_cat`        | *(oof-cv)*      | 0.12473 | Cat dominates; convex blend not helpful |
| Earlier LGBM refs (`v01/v02` mixes)          | —               | 0.12499–0.12548 | Older folds; not directly comparable |

> Interpretation: CatBoost carries most of the tree-based signal on v02 with the new folds. Stacks help only if added models bring **uncorrelated** signal.

---

## Diagnostics & Fixes
- **Fold volatility by neighborhood**: StoneBr / OldTown / Somerst showed higher residuals on older splits.  
  **Fix**: neighborhood×price stratification reduced extremes and stabilized CV.
- **CatBoost NaNs in cats**: raised error; fixed by converting cats to strings with `"Missing"`.
- **Stack degradation**: traced to mixed fold schemes & high model correlation; standardized to new folds for all tree models.

---

## Artifacts
- **OOF (log)**: `../artifacts/oof/<run_name>_oof.csv`
- **Submissions**: `../artifacts/submissions/<run_name>.csv`
- **Feature importance**:
  - CSV: `../artifacts/<run_name>_feat_importance*.csv`
  - PNG: `../artifacts/<run_name>_feat_importance_top30.png`, `..._all.png`
- **Tuning**:
  - Best params: `../artifacts/configs/<run_name>_best_params.yaml`
  - Trials: `../artifacts/<run_name>_optuna_trials.json`
- **Tracking**: MLflow (`file:../mlruns`), Optuna storage (`sqlite:///../optuna_studies.db`)

---

## Repro: one-liner
```bash
# sequential, safe defaults
./scripts/run_trees.sh 80 nbhdstrat