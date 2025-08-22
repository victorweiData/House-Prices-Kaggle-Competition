# EDA Report — House Prices (Kaggle)

**Scope:** Data understanding and cleaning for the "House Prices – Advanced Regression Techniques" dataset, producing a single canonical, modeling-ready table and metadata.

**Key principles:** goal-driven EDA, leak-free preprocessing, reproducibility, and model-agnostic outputs.

## 1. Objectives

- Validate and repair data integrity (shape, IDs, dtypes, consistency)
- Handle missingness with domain logic (distinguish "not present" vs missing)
- Normalize semantics: ordinal encodings, categorical hygiene
- Identify and mitigate outliers without destroying signal
- Prepare stable artifacts (parquet + metadata) for downstream pipelines

## 2. Data Sources

- `train.csv`, `test.csv` (Kaggle)
- Canonical combined table produced during EDA (with train/test flag)

## 3. Environment & Conventions

- Python + pandas/NumPy; seaborn/matplotlib for optional visuals; missingno for NA diagnostics
- Paths centralized via `Path("../data/")`
- All model-specific transforms (scaling, one-hot, power/log transforms) are deferred to modeling pipelines to avoid leakage

---

## 4. What We Did (Step-by-Step)

### 4.1 General Integrity Checks

- Shapes and column counts (train vs test)
- Id presence, uniqueness, and no overlap between train/test
- Full-row duplicates
- Feature alignment (excluding SalePrice): set difference + order check
- Dtype mismatches between shared columns
- Zero-variance columns flagged

**Outcome:** dataset structure validated; early warnings surfaced before modeling.

### 4.2 Missing Data Strategy

- **"NA means not present" categories** filled with literal "None": Alley, basement quals/finishes, garage quals/finishes/types, PoolQC, Fence, MiscFeature, MasVnrType
- **Basement numeric areas/baths** → 0 where missing
- **Garage logic:**
  - If no garage: GarageCars/GarageArea/GarageYrBlt = 0
  - Else, missing GarageYrBlt → YearBuilt proxy
- **MasVnrType/MasVnrArea consistency:** "None" ↔ 0
- **LotFrontage** imputed by Neighborhood median, then global median fallback
- **Low-missingness categoricals** (e.g., MSZoning, Electrical, KitchenQual, Exterior1st/2nd, SaleType, Utilities) → mode; Functional → "Typ"

**Outcome:** coherent NA handling that respects domain meaning and reduces bias.

### 4.3 Data Type Corrections

- **Categorical codes:** MSSubClass, MoSold, YrSold set to category (treated as categorical, not numeric)
- **Ordinal mappings to integers:**
  - Qualities/conditions (Ex/Gd/TA/Fa/Po/None → 5..0)
  - Basement exposure (Gd/Av/Mn/No/None → 4..0)
  - Basement finish types (GLQ/ALQ/BLQ/Rec/LwQ/Unf/None → 6..0)
  - Functional (Typ/Min1/Min2/Mod/Maj1/Maj2/Sev/Sal → 7..0)
- **CentralAir** → binary (Y=1, N=0)

**Outcome:** models can learn monotonic relationships; clearer semantics.

### 4.4 Outlier Detection & Treatment

- Distribution/boxplot checks for heavy-tailed numerics
- IQR bounds on size features (GrLivArea, LotArea, TotalBsmtSF, GarageArea, LotFrontage) to flag candidates
- Competition-specific heuristic: extremely large GrLivArea paired with unusually low price
- Cook's distance from a simple OLS on log(SalePrice) to identify high-influence points
- Surgical drops: a small set of train rows only; no test rows removed
- Optional winsorization (train-quantile-based) staged for linear/NN models; not applied globally (trees don't need it)

**Outcome:** remove clear leverage points while preserving realistic extremes.

### 4.5 Consistency Repairs

- **Additive identities enforced:**
  - TotalBsmtSF = BsmtFinSF1 + BsmtFinSF2 + BsmtUnfSF
  - GrLivArea = 1stFlrSF + 2ndFlrSF + LowQualFinSF
- **Garage consistency:** zeros when no garage; year sanity; counts cast to int
- **Fireplace/Pool:** quality aligned with presence (0 when absent); where present but missing, filled via Neighborhood medians (feature-only)
- **Masonry veneer:** type/area mutually consistent
- **Basement:** if no area, ordinal basement descriptors set to 0; otherwise filled via Neighborhood medians (feature-only)
- **Years:** YearRemodAdd ≥ YearBuilt; garage year floor rules

**Outcome:** removes contradictory signals that especially hurt linear/NN fits.

### 4.6 Standardization & Encoding Prep (no transforms applied yet)

- All object → category
- **Rare-category bucketing** only for string-like categoricals (train-driven ≥1% or ≥10 samples) with label "Other"
- Numeric-coded categoricals (e.g., MSSubClass, MoSold, YrSold) left untouched to avoid Arrow/dtype conflicts and to keep semantics
- **Cyclic month features:** MoSold_sin, MoSold_cos (keep raw MoSold too)
- **Typed column lists for pipelines:**
  - ordinal_int_cols (mapped ordinals)
  - cont_num_cols (continuous numerics)
  - nominal_cols_final (categoricals)
- **Skewness assessment** (train-only) recorded to metadata; actual log/Yeo-Johnson transforms will happen in model pipelines

**Outcome:** stable, consistent category spaces and explicit feature groups for downstream pipelines.

---

## 5. Artifacts Saved

- **Canonical cleaned table (Parquet):** `data/processed/hp_clean_v01.parquet`
- **Plus convenience splits:** `hp_train_clean_v01.parquet`, `hp_test_clean_v01.parquet`
- **Metadata JSON:** `data/processed/hp_clean_meta_v01.json` containing:
  - Ordinal maps; rare-bucketing label/threshold; column lists (ordinal_int_cols, cont_num_cols, nominal_cols_final)
  - (Optional additions): string_cat_cols, numeric_coded_cat_cols, planned transform lists (log1p_cols, yeojohnson_cols)

**Note:** If you adopted the "bucket only string categoricals" rebuild as a separate version, the canonical files are `..._v02.parquet/json`. Either way, there is one canonical dataset at a time.

---

## 6. Deferred to Modeling (by design)

- Scaling, winsorization, power/log transforms on features
- One-hot encoding / embedding of categoricals
- Target encodings (if any) — applied fold-wise only
- Target transform: train on log(SalePrice), inverse on prediction

---

## 7. Reproducibility

- Single source of truth (one parquet + one metadata JSON)
- All transforms for modeling wrapped in sklearn pipelines and fit within CV folds
- Versioned files (v01, v02, …) and optional commit hash stored in metadata

## 8. Next Steps

- **Feature engineering notebook:** totals/ratios, amenities flags, age features, small set of interactions, neighborhood-safe derivations
- **Modeling notebooks:** linear/NN vs. GBDTs; CV; tuning; ensembling; submission

---

## Appendix A — Ordinal Mappings (Summary)

- **Qualities/Conditions:** Ex=5, Gd=4, TA=3, Fa=2, Po=1, None=0
- **Basement Exposure:** Gd=4, Av=3, Mn=2, No=1, None=0
- **Basement Finish:** GLQ=6, ALQ=5, BLQ=4, Rec=3, LwQ=2, Unf=1, None=0
- **Functional:** Typ=7, Min1=6, Min2=5, Mod=4, Maj1=3, Maj2=2, Sev=1, Sal=0
- **CentralAir:** Y=1, N=0