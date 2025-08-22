#!/bin/bash
set -e  # stop if any command fails

# LightGBM
echo "=== Running LightGBM training ==="
python train_lgbm.py \
  --run-name lgbm_v02_optuna_nbhdstrat \
  --n-trials 100

sleep 10

# XGBoost
echo "=== Running XGBoost training ==="
python train_xgb.py \
  --run-name xgb_v02_optuna_nbhdstrat \
  --n-trials 100

sleep 10

# CatBoost
echo "=== Running CatBoost training ==="
python train_cat.py \
  --run-name cat_v02_optuna_nbhdstrat \
  --n-trials 100

echo "=== All trainings finished ==="