# 🏡 House Prices – Advanced Regression Techniques (Kaggle)

This repository contains my end-to-end pipeline for the [Kaggle House Prices: Advanced Regression Techniques](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/overview) competition.

## 🚀 Results

* **Leaderboard placement:** Top **3%** (Public LB) (minus the cheaters who used leaked predictations)
* **Final score:** ~0.120 RMSE (log scale)
* **Modeling strategy:** A **single CatBoost model** (v04 feature set) – no heavy stacking or complex blending.

![alt text](image.png)

Despite experimenting with linear models (Ridge, Lasso, ElasticNet), kernel methods, XGBoost, LightGBM, and stacking/blending approaches, the **best performance came from a single, well-tuned CatBoost model** with careful feature engineering and robust cross-validation.

## 🔑 Highlights of My Approach

* **Thorough EDA & cleaning:** Consistent treatment of missing values, rare-category bucketing, and ordinal encodings.
* **Feature engineering:**
   * Neighborhood-aware target encodings.
   * Relative quality/size ratios.
   * Selected interaction terms and domain-specific ratios.
* **Cross-validation:** Stratified by neighborhood × log-price to stabilize fold variance and mimic LB distribution.
* **Modeling:**
   * **CatBoost Regressor** with tuned parameters via Optuna.
   * Used categorical handling natively (no one-hot explosion).
* **Why single model:** Stacking and blending did not improve LB stability – CatBoost alone generalized best.

## 🛠️ Tech Stack

**Languages & Environment**

* Python 3.12
* Jupyter Notebooks for exploration and prototyping
* Bash scripting for automation

**Core Libraries**

* **Data & Preprocessing:** pandas, numpy, scikit-learn
* **Visualization:** matplotlib, seaborn, plotly, missingno
* **Modeling:**
   * Gradient boosting: catboost, lightgbm, xgboost, sklearn.ensemble
   * Linear models: Ridge, Lasso, ElasticNet, KernelRidge
   * Neural networks: MLPRegressor (sklearn)
* **Optimization & Logging:**
   * Hyperparameter tuning: Optuna
   * Experiment tracking: MLflow

**Workflow**

* Cross-validation folds saved in data/processed/
* Model artifacts (OOF predictions, submissions, feature importances) saved in artifacts/
* Scripts parameterized with argparse for reproducibility
* Experiments logged with **MLflow**, search spaces managed with **Optuna**

## 📊 Lessons Learned

* Linear models achieved low CV error (~0.11 RMSE) but failed to transfer to LB (scores ~0.127).
* Tree ensembles (CatBoost, LGBM, XGB) were more stable, but CatBoost consistently outperformed blends.
* Sometimes **the simplest strong baseline outperforms complex ensembles** – quality features + robust CV matter more.

## 📂 Repository Structure

```
.
├── notebooks/           # EDA, Feature Engineering, Modeling experiments
├── training/            # Training scripts (argparse + MLflow logging)
├── data/                # Raw and processed datasets
├── artifacts/           # OOF predictions, submissions, feature importances
├── docs/                # Project reports (EDA, FE, Modeling)
└── README.md            # This file
```
