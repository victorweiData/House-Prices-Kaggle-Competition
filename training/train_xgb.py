#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
XGBoost trainer for Kaggle House Prices with:
- Neighborhood×price stratified folds
- Optuna tuning (Stage-A style)
- Refit best -> OOF (log), submission, feature importance (CSV + PNG)
- MLflow logging (params, metrics, artifacts)

Trains on log(SalePrice); submissions are expm1 back-transformed.
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
from xgboost import XGBRegressor

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
    p.add_argument("--run-name", type=str, default="xgb_optuna_run")
    p.add_argument("--study-name", type=str, default=None)
    p.add_argument("--optuna-storage", type=str, default="sqlite:///../optuna_studies.db")
    p.add_argument("--n-trials", type=int, default=30)
    p.add_argument("--early-stopping", type=int, default=200)
    p.add_argument("--random-state", type=int, default=42)

    # MLflow
    p.add_argument("--mlflow-tracking-uri", type=str, default="file:../mlruns")
    p.add_argument("--mlflow-experiment", type=str, default="houseprices_xgb")

    # Plotting
    p.add_argument("--topk", type=int, default=30)

    # Fixed base params
    p.add_argument("--n-estimators", type=int, default=10000)
    return p.parse_args()


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

    # Categorical list from meta (+ engineered if present)
    nominal_cols = set(meta.get("nominal_cols_final", [])) | set(meta.get("engineered_nominal", []))
    maybe_engineered = {"Nbhd_Qual_cat_v2", "Nbhd_Decade_cat_v2", "NbhdCluster4_v2"}
    nominal_cols |= {c for c in maybe_engineered if c in df_tr.columns}
    cat_features = [c for c in nominal_cols if c in df_tr.columns]

    # Native categorical: require pandas 'category'
    for c in cat_features:
        df_tr[c] = df_tr[c].astype("category")
        df_te[c] = df_te[c].astype("category")

    X = df_tr[feature_cols].copy()
    T = df_te[feature_cols].copy()
    y_log = np.log1p(df_tr[target_col]).astype(float)
    fold = df_tr["fold"].values

    return df_tr, df_te, X, T, y_log, fold, feature_cols, cat_features


def build_search_space(trial):
    """
    Depthwise trees + native categorical.
    - higher min_child_weight floor for stability on 1.5k rows
    - add colsample_bylevel to reduce feature co-adaptation
    """
    params = dict(
        objective="reg:squarederror",

        # learning
        learning_rate=trial.suggest_float("learning_rate", 1e-3, 2e-1, log=True),

        # capacity (depthwise)
        max_depth=trial.suggest_int("max_depth", 4, 10),
        min_child_weight=trial.suggest_float("min_child_weight", 0.5, 32.0, log=True),

        # sampling
        subsample=trial.suggest_float("subsample", 0.6, 1.0),
        colsample_bytree=trial.suggest_float("colsample_bytree", 0.6, 1.0),
        colsample_bylevel=trial.suggest_float("colsample_bylevel", 0.6, 1.0),

        # regularization
        reg_alpha=trial.suggest_float("reg_alpha", 1e-4, 50.0, log=True),
        reg_lambda=trial.suggest_float("reg_lambda", 1e-4, 50.0, log=True),
        gamma=trial.suggest_float("gamma", 1e-4, 10.0, log=True),

        # categorical knobs
        max_cat_to_onehot=trial.suggest_categorical("max_cat_to_onehot", [16, 32, 64, 128]),
        max_cat_threshold=trial.suggest_categorical("max_cat_threshold", [32, 64, 128]),

        # fixed
        tree_method="hist",
        enable_categorical=True,
    )
    return params


def cv_train_eval(params, X, y_log, T, fold, n_estimators, early_stopping, seed):
    oof = np.zeros(len(X), dtype=float)
    test_folds, fold_scores = [], []
    feat_imps_gain, feat_imps_weight = [], []

    for k in sorted(np.unique(fold)):
        tr_idx = np.where(fold != k)[0]
        va_idx = np.where(fold == k)[0]

        model = XGBRegressor(
            **params,
            n_estimators=n_estimators,
            early_stopping_rounds=early_stopping,
            random_state=seed,
            n_jobs=-1,
        )
        _ = model.fit(
            X.iloc[tr_idx], y_log.iloc[tr_idx],
            eval_set=[(X.iloc[va_idx], y_log.iloc[va_idx])],
            verbose=False,
        )
        pred_va = model.predict(X.iloc[va_idx], iteration_range=(0, model.best_iteration))
        oof[va_idx] = pred_va
        fold_scores.append(float(np.sqrt(mean_squared_error(y_log.iloc[va_idx], pred_va))))
        test_folds.append(model.predict(T, iteration_range=(0, model.best_iteration)))

        booster = model.get_booster()
        gain = booster.get_score(importance_type="gain")
        weight = booster.get_score(importance_type="weight")
        feat_imps_gain.append(pd.DataFrame({"feature": list(gain.keys()), "gain": list(gain.values()), "fold": int(k)}))
        feat_imps_weight.append(pd.DataFrame({"feature": list(weight.keys()), "weight": list(weight.values()), "fold": int(k)}))

    return oof, test_folds, fold_scores, feat_imps_gain, feat_imps_weight


def make_importance_plots(agg_df, out_top_png, out_all_png, topk=30, label="Gain"):
    top = agg_df.head(topk).sort_values(agg_df.columns[1], ascending=True)
    plt.figure(figsize=(9, 10))
    plt.barh(top["feature"], top.iloc[:, 1])
    plt.xlabel(f"{label} Importance (mean across folds)")
    plt.ylabel("Feature")
    plt.title(f"XGBoost Feature Importance — Top {topk}")
    plt.tight_layout(); plt.savefig(out_top_png, dpi=200, bbox_inches="tight"); plt.close()

    nfeat = len(agg_df)
    height_in = float(np.clip(nfeat * 0.25, 6.0, 60.0))
    all_sorted = agg_df.sort_values(agg_df.columns[1], ascending=True)
    plt.figure(figsize=(10, height_in))
    plt.barh(all_sorted["feature"], all_sorted.iloc[:, 1])
    plt.xlabel(f"{label} Importance (mean across folds)")
    plt.ylabel("Feature")
    plt.title(f"XGBoost Feature Importance — All {nfeat}")
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
    df_tr, df_te, X, T, y_log, fold, feature_cols, cat_features = load_data(args)

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
        params = build_search_space(trial)
        scores = []
        for k in sorted(np.unique(fold)):
            tr_idx = np.where(fold != k)[0]
            va_idx = np.where(fold == k)[0]
            model = XGBRegressor(
                **params,
                n_estimators=args.n_estimators,
                early_stopping_rounds=args.early_stopping,
                random_state=args.random_state,
                n_jobs=-1,
            )
            _ = model.fit(
                X.iloc[tr_idx], y_log.iloc[tr_idx],
                eval_set=[(X.iloc[va_idx], y_log.iloc[va_idx])],
                verbose=False,
            )
            pred = model.predict(X.iloc[va_idx], iteration_range=(0, model.best_iteration))
            rmse = float(np.sqrt(mean_squared_error(y_log.iloc[va_idx], pred)))
            scores.append(rmse)
            trial.report(rmse, step=int(k))
            if trial.should_prune():
                raise optuna.TrialPruned()
        return float(np.mean(scores))

    study.optimize(objective, n_trials=args.n_trials, gc_after_trial=True)
    best_params = dict(study.best_params)
    best_params.update(dict(objective="reg:squarederror", tree_method="hist", enable_categorical=True))

    # Refit + artifacts
    oof, test_folds, fold_scores, imp_gain, imp_weight = cv_train_eval(
        best_params, X, y_log, T, fold,
        n_estimators=args.n_estimators,
        early_stopping=args.early_stopping,
        seed=args.random_state
    )
    cv_mean, cv_std = float(np.mean(fold_scores)), float(np.std(fold_scores))
    test_mean = np.column_stack(test_folds).mean(axis=1)

    run_tag = args.run_name
    id_col = "Id"
    ART_PATHS = {
        "oof": OOF_DIR / f"{run_tag}_oof.csv",
        "sub": SUB_DIR / f"{run_tag}.csv",
        "imp_gain_csv": ART / f"{run_tag}_feat_importance_gain.csv",
        "imp_weight_csv": ART / f"{run_tag}_feat_importance_weight.csv",
        "imp_gain_top_png": ART / f"{run_tag}_feat_importance_gain_top{args.topk}.png",
        "imp_gain_all_png": ART / f"{run_tag}_feat_importance_gain_all.png",
        "cfg": CFG_DIR / f"{run_tag}_best_params.yaml",
        "trials": ART / f"{run_tag}_optuna_trials.json",
    }

    pd.DataFrame({id_col: df_tr[id_col].values, "pred_log": oof}).to_csv(ART_PATHS["oof"], index=False)
    pd.DataFrame({id_col: df_te[id_col].values, "SalePrice": np.expm1(test_mean)}).to_csv(ART_PATHS["sub"], index=False)

    imp_gain_df = pd.concat(imp_gain, ignore_index=True)
    imp_weight_df = pd.concat(imp_weight, ignore_index=True)
    gain_agg = imp_gain_df.groupby("feature", as_index=False)["gain"].mean().sort_values("gain", ascending=False)
    weight_agg = imp_weight_df.groupby("feature", as_index=False)["weight"].mean().sort_values("weight", ascending=False)
    gain_agg.to_csv(ART_PATHS["imp_gain_csv"], index=False)
    weight_agg.to_csv(ART_PATHS["imp_weight_csv"], index=False)

    # Plots
    make_importance_plots(gain_agg, ART_PATHS["imp_gain_top_png"], ART_PATHS["imp_gain_all_png"], topk=args.topk, label="Gain")

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
        mlflow.log_param("n_estimators", args.n_estimators)
        mlflow.log_param("early_stopping", args.early_stopping)
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
        "importance_csv": str(ART_PATHS["imp_gain_csv"]),
        "best_params_file": str(ART_PATHS["cfg"])
    }, indent=2))


if __name__ == "__main__":
    main()