#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CatBoost trainer for Kaggle House Prices with:
- Neighborhood×price stratified folds
- Optuna tuning (Stage-A style)
- Refit best -> OOF (log), submission, feature importance (CSV + PNG)
- MLflow logging (params, metrics, artifacts)

Notes:
- CatBoost requires categorical features be strings/ints with NO NaN.
- This script converts categorical columns to strings and fills NaN with "Missing".
"""

import os, json, argparse, warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error

import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner

import mlflow
from catboost import CatBoostRegressor, Pool

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
    p.add_argument("--run-name", type=str, default="cat_optuna_run")
    p.add_argument("--study-name", type=str, default=None)
    p.add_argument("--optuna-storage", type=str, default="sqlite:///../optuna_studies.db")
    p.add_argument("--n-trials", type=int, default=30)
    p.add_argument("--early-stopping", type=int, default=200)  # handled by od_wait
    p.add_argument("--random-state", type=int, default=42)

    # MLflow
    p.add_argument("--mlflow-tracking-uri", type=str, default="file:../mlruns")
    p.add_argument("--mlflow-experiment", type=str, default="houseprices_cat")

    # Plotting
    p.add_argument("--topk", type=int, default=30)

    # Fixed base params
    p.add_argument("--iterations", type=int, default=20000)
    return p.parse_args()


def to_cat_strings(df, cols, token="Missing"):
    for c in cols:
        df[c] = df[c].astype("object").where(df[c].notna(), token).astype(str)
    return df


def load_data(args):
    PROCESSED = Path(args.processed_dir)
    df_tr = pd.read_csv(PROCESSED / args.train_file)
    df_te = pd.read_csv(PROCESSED / args.test_file)
    folds_df = pd.read_csv(PROCESSED / args.folds_file)

    with open(PROCESSED / args.meta_file, "r") as f:
        meta = json.load(f)

    id_col, target_col = "Id", "SalePrice"
    df_tr = df_tr.merge(folds_df, on=id_col, how="left")
    feature_cols = [c for c in df_tr.columns if c not in [id_col, target_col, "fold"]]

    # Cat columns (names)
    nominal_cols = set(meta.get("nominal_cols_final", [])) | set(meta.get("engineered_nominal", []))
    maybe_engineered = {"Nbhd_Qual_cat_v2", "Nbhd_Decade_cat_v2", "NbhdCluster4_v2"}
    nominal_cols |= {c for c in maybe_engineered if c in df_tr.columns}
    cat_features = [c for c in nominal_cols if c in df_tr.columns]

    X = df_tr[feature_cols].copy()
    T = df_te[feature_cols].copy()
    y_log = np.log1p(df_tr[target_col]).astype(float).values
    fold = df_tr["fold"].values

    # Ensure cat columns are strings and have no NaN
    X = to_cat_strings(X, cat_features)
    T = to_cat_strings(T, cat_features)

    # Indices for CatBoost
    cat_idx = [X.columns.get_loc(c) for c in cat_features]

    return df_tr, df_te, X, T, y_log, fold, feature_cols, cat_features, cat_idx


def build_search_space(trial, iterations_default):
    """
    Small-data tuned ranges:
    - cooler learning rate band
    - trimmed depth and l2 ranges
    """
    params = dict(
        loss_function="RMSE",
        iterations=iterations_default,
        learning_rate=trial.suggest_float("learning_rate", 0.02, 0.12, log=True),
        depth=trial.suggest_int("depth", 5, 9),
        l2_leaf_reg=trial.suggest_float("l2_leaf_reg", 1.0, 30.0, log=True),

        # sampling
        subsample=trial.suggest_float("subsample", 0.6, 1.0),
        rsm=trial.suggest_float("rsm", 0.6, 1.0),

        # extra regularizers
        random_strength=trial.suggest_float("random_strength", 0.0, 2.0),
        bagging_temperature=trial.suggest_float("bagging_temperature", 0.0, 10.0),

        # fixed
        random_seed=42,
        eval_metric="RMSE",
        od_type="Iter",
        od_wait=200,
        allow_writing_files=False,
        verbose=False,
    )
    return params


def cv_train_eval(params, X, y_log, T, fold, cat_idx):
    oof = np.zeros(len(X), dtype=float)
    test_folds, fold_scores, feat_imps = [], [], []

    for k in sorted(np.unique(fold)):
        tr_idx = np.where(fold != k)[0]
        va_idx = np.where(fold == k)[0]

        train_pool = Pool(X.iloc[tr_idx], label=y_log[tr_idx], cat_features=cat_idx)
        valid_pool = Pool(X.iloc[va_idx], label=y_log[va_idx], cat_features=cat_idx)
        test_pool  = Pool(T, cat_features=cat_idx)

        model = CatBoostRegressor(**params)
        model.fit(train_pool, eval_set=valid_pool, use_best_model=True)

        pred_va = model.predict(valid_pool)
        oof[va_idx] = pred_va
        fold_scores.append(float(np.sqrt(mean_squared_error(y_log[va_idx], pred_va))))
        test_folds.append(model.predict(test_pool))

        fi_vals = model.get_feature_importance(train_pool, type="PredictionValuesChange")
        feat_imps.append(pd.DataFrame({"feature": X.columns, "importance": fi_vals, "fold": int(k)}))

    return oof, test_folds, fold_scores, feat_imps


def make_importance_plots(agg_df, out_top_png, out_all_png, topk=30):
    top = agg_df.head(topk).sort_values("importance", ascending=True)
    plt.figure(figsize=(9, 10))
    plt.barh(top["feature"], top["importance"])
    plt.xlabel("CatBoost Importance (PredictionValuesChange, mean across folds)")
    plt.ylabel("Feature")
    plt.title(f"CatBoost Feature Importance — Top {topk}")
    plt.tight_layout(); plt.savefig(out_top_png, dpi=200, bbox_inches="tight"); plt.close()

    nfeat = len(agg_df)
    height_in = float(np.clip(nfeat * 0.25, 6.0, 60.0))
    all_sorted = agg_df.sort_values("importance", ascending=True)
    plt.figure(figsize=(10, height_in))
    plt.barh(all_sorted["feature"], all_sorted["importance"])
    plt.xlabel("CatBoost Importance (PredictionValuesChange, mean across folds)")
    plt.ylabel("Feature")
    plt.title(f"CatBoost Feature Importance — All {nfeat}")
    plt.tight_layout(); plt.savefig(out_all_png, dpi=200, bbox_inches="tight"); plt.close()


def main():
    args = parse_args()

    # MLflow
    os.environ["MLFLOW_TRACKING_URI"] = args.mlflow_tracking_uri
    mlflow.set_tracking_uri(args.mlflow_tracking_uri)
    mlflow.set_experiment(args.mlflow_experiment)

    # Artifacts
    ART = Path(args.artifacts_dir); ART.mkdir(parents=True, exist_ok=True)
    OOF_DIR = ART / "oof"; OOF_DIR.mkdir(exist_ok=True)
    SUB_DIR = ART / "submissions"; SUB_DIR.mkdir(exist_ok=True)
    CFG_DIR = ART / "configs"; CFG_DIR.mkdir(exist_ok=True)

    # Data
    df_tr, df_te, X, T, y_log, fold, feature_cols, cat_features, cat_idx = load_data(args)

    # Optuna
    study = optuna.create_study(
        study_name=args.study_name or f"{args.run_name}_study",
        direction="minimize",
        storage=args.optuna_storage,
        load_if_exists=True,
        sampler=TPESampler(seed=args.random_state),
        pruner=MedianPruner(n_warmup_steps=2),
    )

    def objective(trial):
        params = build_search_space(trial, args.iterations)
        scores = []
        for k in sorted(np.unique(fold)):
            tr_idx = np.where(fold != k)[0]
            va_idx = np.where(fold == k)[0]
            train_pool = Pool(X.iloc[tr_idx], label=y_log[tr_idx], cat_features=cat_idx)
            valid_pool = Pool(X.iloc[va_idx], label=y_log[va_idx], cat_features=cat_idx)
            model = CatBoostRegressor(**params)
            model.fit(train_pool, eval_set=valid_pool, use_best_model=True)
            pred = model.predict(valid_pool)
            rmse = float(np.sqrt(mean_squared_error(y_log[va_idx], pred)))
            scores.append(rmse)
            trial.report(rmse, step=int(k))
            if trial.should_prune():
                raise optuna.TrialPruned()
        return float(np.mean(scores))

    study.optimize(objective, n_trials=args.n_trials, gc_after_trial=True)
    best_params = dict(study.best_params)
    # Fill fixed defaults
    best_params.update(dict(loss_function="RMSE", iterations=args.iterations, eval_metric="RMSE",
                            od_type="Iter", od_wait=200, allow_writing_files=False, verbose=False))

    # Refit + artifacts
    oof, test_folds, fold_scores, feat_imps = cv_train_eval(best_params, X, y_log, T, fold, cat_idx)
    cv_mean, cv_std = float(np.mean(fold_scores)), float(np.std(fold_scores))
    test_mean = np.column_stack(test_folds).mean(axis=1)

    run_tag = args.run_name
    id_col = "Id"
    ART_PATHS = {
        "oof": OOF_DIR / f"{run_tag}_oof.csv",
        "sub": SUB_DIR / f"{run_tag}.csv",
        "imp_csv": ART / f"{run_tag}_feat_importance.csv",
        "imp_top_png": ART / f"{run_tag}_feat_importance_top{args.topk}.png",
        "imp_all_png": ART / f"{run_tag}_feat_importance_all.png",
        "cfg": CFG_DIR / f"{run_tag}_best_params.yaml",
        "trials": ART / f"{run_tag}_optuna_trials.json",
    }

    pd.DataFrame({id_col: df_tr[id_col].values, "pred_log": oof}).to_csv(ART_PATHS["oof"], index=False)
    pd.DataFrame({id_col: df_te[id_col].values, "SalePrice": np.expm1(test_mean)}).to_csv(ART_PATHS["sub"], index=False)

    fi_df = pd.concat(feat_imps, ignore_index=True)
    fi_agg = fi_df.groupby("feature", as_index=False)["importance"].mean().sort_values("importance", ascending=False)
    fi_agg.to_csv(ART_PATHS["imp_csv"], index=False)

    # Plots
    make_importance_plots(fi_agg, ART_PATHS["imp_top_png"], ART_PATHS["imp_all_png"], topk=args.topk)

    # Save best params & trials
    cfg_text = "\n".join(f"{k}: {v}" for k, v in sorted(best_params.items()))
    ART_PATHS["cfg"].write_text(cfg_text)
    trials_summary = [{"number": t.number, "state": str(t.state), "value": t.value, "params": t.params} for t in study.trials]
    ART_PATHS["trials"].write_text(json.dumps(trials_summary, indent=2))

    # MLflow log
    with mlflow.start_run(run_name=run_tag):
        mlflow.log_params(best_params)
        mlflow.log_param("feature_version", args.train_file)
        mlflow.log_param("folds_file", args.folds_file)
        mlflow.log_param("iterations", args.iterations)
        mlflow.log_param("random_state", args.random_state)
        mlflow.log_param("n_trials", args.n_trials)
        mlflow.log_param("study_name", study.study_name)

        mlflow.log_metric("cv_rmse_mean_log", cv_mean)
        mlflow.log_metric("cv_rmse_std_log", cv_std)
        for i, s in enumerate(fold_scores):
            mlflow.log_metric(f"fold{i}_rmse_log", float(s))

        for k, p in ART_PATHS.items():
            mlflow.log_artifact(str(p))

    print(json.dumps({
        "run_name": run_tag,
        "cv_rmse_mean_log": cv_mean,
        "cv_rmse_std_log": cv_std,
        "oof_path": str(ART_PATHS["oof"]),
        "submission_path": str(ART_PATHS["sub"]),
        "importance_csv": str(ART_PATHS["imp_csv"]),
        "best_params_file": str(ART_PATHS["cfg"])
    }, indent=2))


if __name__ == "__main__":
    main()