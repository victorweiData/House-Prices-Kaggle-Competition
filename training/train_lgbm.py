#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LightGBM trainer for Kaggle House Prices with:
- New neighborhood×price stratified folds
- Optuna tuning (Stage-A style)
- Refit best -> OOF (log), submission, feature importance (CSV + PNG)
- Full MLflow logging (params, metrics, artifacts)

Target: trains on log(SalePrice); submissions are expm1 back-transformed.
"""

import os
import json
import math
import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import mlflow

warnings.filterwarnings("ignore")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=str, default="../data")
    p.add_argument("--processed-dir", type=str, default="../data/processed")
    p.add_argument("--train-file", type=str, default="hp_train_feat_v02.csv")
    p.add_argument("--test-file", type=str, default="hp_test_feat_v02.csv")
    p.add_argument("--folds-file", type=str, default="cv_folds_strat_nbhd_price_v01.csv")
    p.add_argument("--meta-file", type=str, default="hp_clean_meta_v03.json")

    p.add_argument("--artifacts-dir", type=str, default="../artifacts")
    p.add_argument("--run-name", type=str, default="lgbm_optuna_run")
    p.add_argument("--study-name", type=str, default=None)
    p.add_argument("--optuna-storage", type=str, default="sqlite:///../optuna_studies.db")
    p.add_argument("--n-trials", type=int, default=30)
    p.add_argument("--early-stopping", type=int, default=200)
    p.add_argument("--random-state", type=int, default=42)

    # MLflow
    p.add_argument("--mlflow-tracking-uri", type=str, default="file:../mlruns")
    p.add_argument("--mlflow-experiment", type=str, default="houseprices_lgbm")

    # Plotting
    p.add_argument("--topk", type=int, default=30)

    # Fixed base params (you can override here if you like)
    p.add_argument("--n-estimators", type=int, default=10000)
    return p.parse_args()


def load_data(args):
    DATA_DIR = Path(args.data_dir)
    PROCESSED = Path(args.processed_dir)

    df_tr = pd.read_csv(PROCESSED / args.train_file)
    df_te = pd.read_csv(PROCESSED / args.test_file)
    folds_df = pd.read_csv(PROCESSED / args.folds_file)

    with open(PROCESSED / args.meta_file, "r") as f:
        meta = json.load(f)

    id_col = "Id"
    target_col = "SalePrice"

    # Merge folds
    df_tr = df_tr.merge(folds_df, on=id_col, how="left")
    if "fold" not in df_tr.columns:
        raise ValueError("Folds file must include a 'fold' column keyed by Id.")

    # Feature list
    feature_cols = [c for c in df_tr.columns if c not in [id_col, target_col, "fold"]]

    # Categorical features: from meta + any engineered cats you keep
    nominal_cols = set(meta.get("nominal_cols_final", [])) | set(meta.get("engineered_nominal", []))
    # If you still keep these engineered cats, they’ll be picked up automatically if present:
    maybe_engineered = {"Nbhd_Qual_cat_v2", "Nbhd_Decade_cat_v2", "NbhdCluster4_v2"}
    nominal_cols |= {c for c in maybe_engineered if c in df_tr.columns}

    cat_features = [c for c in nominal_cols if c in df_tr.columns]
    for c in cat_features:
        df_tr[c] = df_tr[c].astype("category")
        df_te[c] = df_te[c].astype("category")

    X = df_tr[feature_cols].copy()
    T = df_te[feature_cols].copy()
    y_log = np.log1p(df_tr[target_col]).astype(float)
    fold = df_tr["fold"].values

    return df_tr, df_te, X, T, y_log, fold, feature_cols, cat_features, meta


def cv_train_eval(params, X, y_log, T, fold, cat_features, n_estimators=10000, early_stopping=200, seed=42):
    """Train LGBM with fixed folds; return OOF, test_preds_per_fold, fold_scores, feature_importance list."""
    oof = np.zeros(len(X), dtype=float)
    test_folds = []
    fold_scores = []
    feat_imps = []

    for k in sorted(np.unique(fold)):
        tr_idx = np.where(fold != k)[0]
        va_idx = np.where(fold == k)[0]

        model = lgb.LGBMRegressor(**params, n_estimators=n_estimators, random_state=seed, n_jobs=-1)
        _ = model.fit(
            X.iloc[tr_idx], y_log.iloc[tr_idx],
            eval_set=[(X.iloc[va_idx], y_log.iloc[va_idx])],
            eval_metric="rmse",
            callbacks=[lgb.early_stopping(early_stopping, verbose=False)],
            categorical_feature=cat_features,
        )
        pred_va = model.predict(X.iloc[va_idx], num_iteration=model.best_iteration_)
        oof[va_idx] = pred_va
        rmse_k = float(np.sqrt(mean_squared_error(y_log.iloc[va_idx], pred_va)))
        fold_scores.append(rmse_k)

        test_folds.append(model.predict(T, num_iteration=model.best_iteration_))

        booster = model.booster_
        feat_imps.append(pd.DataFrame({
            "feature": booster.feature_name(),
            "gain": booster.feature_importance(importance_type="gain"),
            "split": booster.feature_importance(importance_type="split"),
            "fold": int(k),
        }))

    return oof, test_folds, fold_scores, feat_imps


def make_importance_plots(feat_imps_df, out_top_png, out_all_png, topk=30, title_prefix="LGBM Importance"):
    stats = (
        feat_imps_df.groupby("feature", as_index=False)[["gain"]]
        .mean()
        .sort_values("gain", ascending=False)
    )

    # Top-K
    top = stats.head(topk).sort_values("gain", ascending=True)
    plt.figure(figsize=(9, 10))
    plt.barh(top["feature"], top["gain"])
    plt.xlabel("Gain importance (mean across folds)")
    plt.ylabel("Feature")
    plt.title(f"{title_prefix} — Top {topk}")
    plt.tight_layout()
    plt.savefig(out_top_png, dpi=200, bbox_inches="tight")
    plt.close()

    # All
    nfeat = len(stats)
    height_in = float(np.clip(nfeat * 0.25, 6.0, 60.0))
    all_stats = stats.sort_values("gain", ascending=True)
    plt.figure(figsize=(10, height_in))
    plt.barh(all_stats["feature"], all_stats["gain"])
    plt.xlabel("Gain importance (mean across folds)")
    plt.ylabel("Feature")
    plt.title(f"{title_prefix} — All {nfeat}")
    plt.tight_layout()
    plt.savefig(out_all_png, dpi=200, bbox_inches="tight")
    plt.close()

    return stats



def build_search_space(trial, anchor=None):
    """
    Narrowed Stage-A for ~1.5k rows:
    - smaller capacity, stronger regularization options
    - categorical stabilizers: cat_smooth, min_sum_hessian_in_leaf
    """
    params = dict(
        objective="regression",
        metric="rmse",
        verbosity=-1,

        # learning
        learning_rate=trial.suggest_float("learning_rate", 1e-2, 2e-1, log=True),

        # capacity
        num_leaves=trial.suggest_int("num_leaves", 16, 256, log=True),
        max_depth=trial.suggest_categorical("max_depth", [-1, 6, 7, 8, 9, 10]),
        min_data_in_leaf=trial.suggest_int("min_data_in_leaf", 10, 200, log=True),

        # sampling
        feature_fraction=trial.suggest_float("feature_fraction", 0.6, 1.0),
        bagging_fraction=trial.suggest_float("bagging_fraction", 0.6, 1.0),
        bagging_freq=trial.suggest_categorical("bagging_freq", [0, 1, 2]),

        # regularization
        lambda_l1=trial.suggest_float("lambda_l1", 1e-4, 30.0, log=True),
        lambda_l2=trial.suggest_float("lambda_l2", 1e-4, 30.0, log=True),
        min_gain_to_split=trial.suggest_float("min_gain_to_split", 1e-4, 1.0, log=True),

        # stabilizers for small data / high-card cats
        min_sum_hessian_in_leaf=trial.suggest_float("min_sum_hessian_in_leaf", 1e-3, 1e-1, log=True),
        cat_smooth=trial.suggest_float("cat_smooth", 1.0, 100.0, log=True),
    )

    # keep leaves consistent with depth if depth is set
    if params["max_depth"] != -1:
        params["num_leaves"] = int(min(params["num_leaves"], 2 ** params["max_depth"]))

    return params


def main():
    args = parse_args()

    # MLflow setup
    os.environ["MLFLOW_TRACKING_URI"] = args.mlflow_tracking_uri
    mlflow.set_tracking_uri(args.mlflow_tracking_uri)
    mlflow.set_experiment(args.mlflow_experiment)

    # Artifacts
    ART = Path(args.artifacts_dir); ART.mkdir(parents=True, exist_ok=True)
    OOF_DIR = ART / "oof"; OOF_DIR.mkdir(exist_ok=True)
    SUB_DIR = ART / "submissions"; SUB_DIR.mkdir(exist_ok=True)
    CFG_DIR = ART / "configs"; CFG_DIR.mkdir(exist_ok=True)

    # Load data
    df_tr, df_te, X, T, y_log, fold, feature_cols, cat_features, meta = load_data(args)

    # Optuna study
    study_name = args.study_name or f"{args.run_name}_study"
    study = optuna.create_study(
        study_name=study_name,
        direction="minimize",
        storage=args.optuna_storage,
        load_if_exists=True,
        sampler=TPESampler(seed=args.random_state),
        pruner=MedianPruner(n_warmup_steps=2),
    )

    def objective(trial):
        params = build_search_space(trial)

        # CV loop
        scores = []
        for k in sorted(np.unique(fold)):
            tr_idx = np.where(fold != k)[0]
            va_idx = np.where(fold == k)[0]

            model = lgb.LGBMRegressor(**params, n_estimators=args.n_estimators,
                                      random_state=args.random_state, n_jobs=-1)
            _ = model.fit(
                X.iloc[tr_idx], y_log.iloc[tr_idx],
                eval_set=[(X.iloc[va_idx], y_log.iloc[va_idx])],
                eval_metric="rmse",
                callbacks=[lgb.early_stopping(args.early_stopping, verbose=False)],
                categorical_feature=cat_features,
            )
            pred = model.predict(X.iloc[va_idx], num_iteration=model.best_iteration_)
            rmse = float(np.sqrt(mean_squared_error(y_log.iloc[va_idx], pred)))
            scores.append(rmse)
            trial.report(rmse, step=int(k))
            if trial.should_prune():
                raise optuna.TrialPruned()

        return float(np.mean(scores))

    # Optimize
    study.optimize(objective, n_trials=args.n_trials, gc_after_trial=True)
    best = study.best_trial
    best_params = dict(study.best_params)
    # Fill in fixed items
    best_params.update(dict(objective="regression", metric="rmse"))

    # Refit with best params -> OOF/sub/importance
    oof, test_folds, fold_scores, feat_imps = cv_train_eval(
        params=best_params,
        X=X, y_log=y_log, T=T, fold=fold,
        cat_features=cat_features,
        n_estimators=args.n_estimators,
        early_stopping=args.early_stopping,
        seed=args.random_state
    )

    # Aggregate results
    cv_mean = float(np.mean(fold_scores)); cv_std = float(np.std(fold_scores))
    test_mean = np.column_stack(test_folds).mean(axis=1)

    run_tag = args.run_name
    id_col = "Id"

    # Save CSV artifacts
    oof_path = (OOF_DIR / f"{run_tag}_oof.csv")
    sub_path = (SUB_DIR / f"{run_tag}.csv")
    imp_csv_path = (ART / f"{run_tag}_feat_importance.csv")

    pd.DataFrame({id_col: df_tr[id_col].values, "pred_log": oof}).to_csv(oof_path, index=False)
    pd.DataFrame({id_col: df_te[id_col].values, "SalePrice": np.expm1(test_mean)}).to_csv(sub_path, index=False)

    feat_imps_df = pd.concat(feat_imps, ignore_index=True)
    feat_imps_agg = (
        feat_imps_df.groupby("feature", as_index=False)[["gain", "split"]]
        .mean()
        .sort_values("gain", ascending=False)
    )
    feat_imps_agg.to_csv(imp_csv_path, index=False)

    # Save best params YAML-ish
    cfg_text = "\n".join(f"{k}: {v}" for k, v in sorted(best_params.items()))
    cfg_path = CFG_DIR / f"{run_tag}_best_params.yaml"
    cfg_path.write_text(cfg_text)

    # Save plots
    top_png = ART / f"{run_tag}_feat_importance_top{args.topk}.png"
    all_png = ART / f"{run_tag}_feat_importance_all.png"
    _ = make_importance_plots(
        feat_imps_df[["feature", "gain", "fold"]],
        out_top_png=top_png,
        out_all_png=all_png,
        topk=args.topk,
        title_prefix=f"LGBM Importance ({run_tag})"
    )

    # MLflow logging
    mlflow.set_tracking_uri(args.mlflow_tracking_uri)
    mlflow.set_experiment(args.mlflow_experiment)
    with mlflow.start_run(run_name=run_tag):
        # params
        mlflow.log_params({k: (float(v) if isinstance(v, (np.floating,)) else v) for k, v in best_params.items()})
        mlflow.log_param("feature_version", args.train_file)
        mlflow.log_param("folds_file", args.folds_file)
        mlflow.log_param("n_estimators", args.n_estimators)
        mlflow.log_param("early_stopping", args.early_stopping)
        mlflow.log_param("random_state", args.random_state)
        mlflow.log_param("n_trials", args.n_trials)
        mlflow.log_param("study_name", study.study_name)

        # metrics
        mlflow.log_metric("cv_rmse_mean_log", cv_mean)
        mlflow.log_metric("cv_rmse_std_log", cv_std)
        for i, s in enumerate(fold_scores):
            mlflow.log_metric(f"fold{i}_rmse_log", float(s))

        # artifacts
        mlflow.log_artifact(str(oof_path))
        mlflow.log_artifact(str(sub_path))
        mlflow.log_artifact(str(imp_csv_path))
        mlflow.log_artifact(str(cfg_path))
        mlflow.log_artifact(str(top_png))
        mlflow.log_artifact(str(all_png))

        # Optional: log all trials as JSON for later inspection
        trials_summary = [
            {
                "number": t.number,
                "state": str(t.state),
                "value": t.value,
                "params": t.params,
            } for t in study.trials
        ]
        trials_path = ART / f"{run_tag}_optuna_trials.json"
        with open(trials_path, "w") as f:
            json.dump(trials_summary, f, indent=2)
        mlflow.log_artifact(str(trials_path))

    # Minimal textual summary to stdout (safe for scripts)
    print(json.dumps({
        "run_name": run_tag,
        "cv_rmse_mean_log": cv_mean,
        "cv_rmse_std_log": cv_std,
        "oof_path": str(oof_path),
        "submission_path": str(sub_path),
        "importance_csv": str(imp_csv_path),
        "best_params_file": str(cfg_path)
    }, indent=2))


if __name__ == "__main__":
    main()