# Feature Engineering Report — House Prices (Kaggle)

**Scope:** Deterministic, leak-free feature engineering on the canonical cleaned dataset produced by EDA.  
**Principles:** domain-aware features, no target usage, reproducible artifacts, model-agnostic outputs (no scaling/OHE/power transforms here).

---

## 1) Objectives

- Enrich the dataset with size, composition, age, amenities, location/materials, and compact interactions.
- Preserve modeling flexibility (linear/NN vs. tree ensembles) by deferring model-specific transforms to pipelines.
- Produce versioned feature artifacts + metadata that are reproducible and CV-safe.

---

## 2) Inputs & Assumptions

**From EDA:**
- `data/processed/hp_train_clean_v01.parquet`, `hp_test_clean_v01.parquet`
- `data/processed/hp_clean_meta_v01.json` (column groups, ordinal maps, rare-bucketing settings)

**Canonical rules (from EDA):**
- String categoricals rare-bucketed; numeric-coded categoricals (e.g., MSSubClass, MoSold, YrSold) left as-is.
- MoSold_sin, MoSold_cos already created for seasonality.
- No rows dropped beyond surgical train-only outliers handled in EDA.
- **No target leakage:** SalePrice is not used in any computation here.

---

## 3) Guardrails & Non-Goals

- **No model-specific transforms:** no scaling, winsorization, one-hot, power/log transforms in this notebook. Those happen inside CV pipelines later.
- **No target encodings here;** if used, they'll be fold-wise in modeling.
- **Keep originals** when adding composites/ratios to let regularization and model choice decide.

---

## 4) Feature Families & Additions

### 4.1 Structural Size & Composition

**Why:** overall scale and layout ratios strongly correlate with price.

**Totals:**
- `fe_TotalSF = 1stFlrSF + 2ndFlrSF + TotalBsmtSF`
- `fe_TotalPorchSF = OpenPorchSF + EnclosedPorch + 3SsnPorch + ScreenPorch`
- `fe_TotalBaths = FullBath + 0.5·HalfBath + BsmtFullBath + 0.5·BsmtHalfBath`

**Ratios / Densities (with safe denominators):**
- `fe_AboveVsBasement = GrLivArea / (TotalBsmtSF + 1)`
- `fe_FirstVsSecond = 1stFlrSF / (2ndFlrSF + 1)`
- `fe_GarageCarsPer100SF = GarageCars / (GarageArea/100 + 1)`

**Notes:** Keep all base parts; ratios capture composition beyond raw totals.

### 4.2 Age & Temporal Features

**Why:** depreciation and recency effects; seasonality already prepped in EDA.

- `fe_AgeAtSale = (YrSold − YearBuilt) ⩾ 0`
- `fe_YearsSinceRemod = (YrSold − YearRemodAdd) ⩾ 0`
- `fe_GarageAgeAtSale = (YrSold − GarageYrBlt) ⩾ 0` (0 if no garage)
- **Season categorical:** `comb_Season ∈ {Winter, Spring, Summer, Fall}` derived from MoSold
- **Keep** MoSold_sin, MoSold_cos (cyclic) from EDA

### 4.3 Amenity Presence Flags (Binary)

**Why:** robust signals that help both trees and linear/NN.

- `bin_HasPool`, `bin_Has2ndFlr`, `bin_HasBasement`, `bin_HasPorch`, `bin_HasDeck`, `bin_HasFireplace`, `bin_HasGarage`
- **Simple rule:** indicator = 1 if the corresponding area/count > 0, else 0.

### 4.4 Quality/Condition Composites (Ordinal-Aware)

**Why:** summarize craftsmanship succinctly while keeping component signals.

- `fe_OverallGrade = 2·OverallQual + OverallCond`
- `fe_ExteriorGrade = ExterQual + ExterCond`
- `fe_BasementGrade = BsmtQual + BsmtCond + BsmtExposure`
- `fe_GarageGrade = GarageQual + GarageCond`
- `fe_KitchenGrade = KitchenQual` (kept simple; counts available separately)

**Ordinal columns** use the EDA mappings (e.g., Ex=5 … None=0).

### 4.5 Location & Materials (Target-Free Derivations)

**Condition proximity flags** (from Condition1, Condition2):
- `bin_NearArtery` if either ∈ {Artery, Feedr}
- `bin_NearRR` if either ∈ {RRNn, RRAn, RRNe, RRAe}
- `bin_NearPositive` if either ∈ {PosN, PosA}

**Exterior combination** (order-invariant material pair):
- `comb_ExteriorSet = "+".join(sorted({Exterior1st, Exterior2nd}))` as a single categorical label.

**Neighborhood feature aggregates** (train-only):
- For each Neighborhood, compute train medians of feature columns (no target):
  - e.g., `fe_NbhdMed_LotArea`, `fe_NbhdMed_YearBuilt`, `fe_NbhdMed_GrLivArea`, `fe_NbhdMed_fe_TotalSF`
- Map medians to both train/test.

**Rationale:** capture localized scale/age norms without target leakage.

### 4.6 Compact Interactions (Shortlist)

**Why:** capture high-value synergies with minimal dimensional blow-up.

- `int_OQual_x_GrLiv = OverallQual · GrLivArea`
- `int_OQual_x_TotalSF = OverallQual · fe_TotalSF`
- `int_TotalBaths_x_Beds = fe_TotalBaths · BedroomAbvGr`
- `int_GarCars_x_GarArea = GarageCars · GarageArea`

Keep list short; additional polynomial interactions can be explored in modeling if needed.

---

## 5) Safety & QA

- **Finite checks:** replace inf/−inf from ratios with NaN → 0.
- **Dtypes:** flags int{0,1}; ratios/totals float; combos category.
- **Cardinality:** review comb_ExteriorSet and comb_Season; apply string-only rare bucketing rules if cardinality spikes (consistent with EDA).
- **Row count:** unchanged from inputs; no row drops.
- **Leakage audit:** no feature used SalePrice or any target-derived statistic.

---

## 6) Artifacts & Versioning

**Saved outputs** (unscaled, unencoded, no power/log transforms):
- `data/processed/hp_train_feat_v01.parquet`
- `data/processed/hp_test_feat_v01.parquet`

**Updated metadata:**
- `data/processed/hp_clean_meta_v02.json` updated with:
  - engineered_binary, engineered_continuous, engineered_nominal, engineered_interactions
  - (Carried from EDA) ordinal_int_cols, cont_num_cols, nominal_cols_final, ordinal maps, rare-bucketing label/threshold, cyclic fields.
- **Feature Registry** (recommended): `docs/feature_registry_v01.json` (or alongside metadata)
  - For each engineered feature: name, dtype, source_columns, logic, notes.

**Versioning:** This doc reflects v01 engineered features with meta v02 (since metadata was extended). Bump versions as you iterate.

---

## 7) What's Deferred to Modeling (by design)

- **Scaling / Standardization** (linear/NN).
- **Winsorization and power/log transforms** on selected continuous features (from EDA skew analysis).
- **One-hot encoding** of string categoricals; pass ordinal ints through.
- **Target encodings** (if any), applied fold-wise only.
- **Target training** on log(SalePrice) via wrapped regressors; inverse at prediction.

---

## 8) Handoff & Next Steps

**Build pipelines per family** using the metadata lists:
- **Linear/NN:** winsorize → log1p/Yeo-Johnson (per lists) → scale → OHE string cats → pass ordinals → regressor wrapped with log/exp.
- **LGBM:** raw numerics + pandas category; specify categorical features directly.
- **XGB:** raw numerics + OHE string cats (or native cat support if stable).
- **CatBoost:** raw numerics + raw string cats with cat indices.

**Establish fixed CV folds** (stratified by log target) and train/evaluate:
- Save OOF predictions for blending/stacking.
- Compare models; blend/stack; prepare submission.

---

## Appendix A — Engineered Feature List (by family)

### Structural
- `fe_TotalSF`, `fe_TotalPorchSF`, `fe_TotalBaths`
- `fe_AboveVsBasement`, `fe_FirstVsSecond`, `fe_GarageCarsPer100SF`

### Age/Temporal
- `fe_AgeAtSale`, `fe_YearsSinceRemod`, `fe_GarageAgeAtSale`
- `comb_Season`, `MoSold_sin`, `MoSold_cos` (from EDA)

### Amenities (Binary)
- `bin_HasPool`, `bin_Has2ndFlr`, `bin_HasBasement`, `bin_HasPorch`, `bin_HasDeck`, `bin_HasFireplace`, `bin_HasGarage`

### Quality/Condition Composites
- `fe_OverallGrade`, `fe_ExteriorGrade`, `fe_BasementGrade`, `fe_GarageGrade`, `fe_KitchenGrade`

### Location/Materials
- `bin_NearArtery`, `bin_NearRR`, `bin_NearPositive`
- `comb_ExteriorSet`
- `fe_NbhdMed_LotArea`, `fe_NbhdMed_YearBuilt`, `fe_NbhdMed_GrLivArea`, `fe_NbhdMed_fe_TotalSF`

### Interactions
- `int_OQual_x_GrLiv`, `int_OQual_x_TotalSF`, `int_TotalBaths_x_Beds`, `int_GarCars_x_GarArea`

---

## Appendix B — Reproducibility Checklist

- ✅ Inputs loaded from versioned parquet/JSON.
- ✅ No target usage.
- ✅ Deterministic operations only.
- ✅ Versioned outputs saved.
- ✅ Metadata lists updated to reflect all new columns.