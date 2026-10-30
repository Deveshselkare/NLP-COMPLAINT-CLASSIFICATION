"""
train.py
========
Training pipeline for baseline and machine learning forecasting models
using chronological train-validation-test splitting.

MODELS IMPLEMENTED:
1. Naive Baseline: Predicts y_{t-1} (yesterday's consumption).
2. Seasonal Naive: Predicts y_{t-7} (same day of previous week).
3. Linear Regression: Fits linear relationship between calendar/lag features and energy.
4. Random Forest Regressor: Non-linear ensemble model capturing complex interactions.

CHRONOLOGICAL SPLIT STRATEGY (NO SHUFFLE):
------------------------------------------
- Training set: 2007-01-14 to 2009-12-31 (~1,080 days)
- Validation set: 2010-01-01 to 2010-06-30 (~181 days)
- Test set (Holdout): 2010-07-01 to 2010-11-26 (~149 days)
"""

import os
import json
import logging
from typing import Tuple, Dict, Any
import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from src.features import engineer_all_features, FEATURE_COLUMNS, TARGET_COLUMN
from src.evaluate import evaluate_forecast, compare_models

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

TRAIN_END_DATE = "2009-12-31"
VAL_END_DATE = "2010-06-30"


def split_chronologically(
    df: pd.DataFrame,
    train_end: str = TRAIN_END_DATE,
    val_end: str = VAL_END_DATE
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Splits the time series into strictly chronological train, validation, and test subsets.
    Never shuffles temporal data!
    """
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])

    train_df = df[df['date'] <= train_end].copy()
    val_df = df[(df['date'] > train_end) & (df['date'] <= val_end)].copy()
    test_df = df[df['date'] > val_end].copy()

    logger.info(
        f"Chronological split completed:\n"
        f"  - Train: {len(train_df)} days ({train_df['date'].min().date()} to {train_df['date'].max().date()})\n"
        f"  - Val:   {len(val_df)} days ({val_df['date'].min().date()} to {val_df['date'].max().date()})\n"
        f"  - Test:  {len(test_df)} days ({test_df['date'].min().date()} to {test_df['date'].max().date()})"
    )
    return train_df, val_df, test_df


def train_and_evaluate_all(
    processed_csv_path: str,
    models_dir: str = "models",
    outputs_dir: str = "outputs"
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """
    Executes complete training and out-of-sample evaluation workflow.
    """
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(outputs_dir, exist_ok=True)

    df_raw = pd.read_csv(processed_csv_path)
    df_features = engineer_all_features(df_raw, drop_incomplete=True)

    train_df, val_df, test_df = split_chronologically(df_features)

    # Feature matrices and targets
    X_train = train_df[FEATURE_COLUMNS]
    y_train = train_df[TARGET_COLUMN]

    X_val = val_df[FEATURE_COLUMNS]
    y_val = val_df[TARGET_COLUMN]

    X_test = test_df[FEATURE_COLUMNS]
    y_test = test_df[TARGET_COLUMN]

    eval_results = []
    trained_artifacts = {}

    # -------------------------------------------------------------
    # 1. Naive Baseline (Persistence: y_pred = lag_1)
    # -------------------------------------------------------------
    y_pred_naive = test_df['lag_1']
    res_naive = evaluate_forecast(y_test, y_pred_naive, model_name="Naive Baseline (t-1)")
    eval_results.append(res_naive)
    logger.info(f"Naive Baseline Test MAE: {res_naive['mae_kwh']} kWh | RMSE: {res_naive['rmse_kwh']} kWh")

    # -------------------------------------------------------------
    # 2. Seasonal Naive Baseline (Weekly persistence: y_pred = lag_7)
    # -------------------------------------------------------------
    y_pred_snaive = test_df['lag_7']
    res_snaive = evaluate_forecast(y_test, y_pred_snaive, model_name="Seasonal Naive (t-7)")
    eval_results.append(res_snaive)
    logger.info(f"Seasonal Naive Test MAE: {res_snaive['mae_kwh']} kWh | RMSE: {res_snaive['rmse_kwh']} kWh")

    # -------------------------------------------------------------
    # 3. Linear Regression with Feature Scaling
    # -------------------------------------------------------------
    logger.info("Training Linear Regression model...")
    lr_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('regressor', LinearRegression())
    ])
    lr_pipeline.fit(X_train, y_train)
    y_pred_lr = lr_pipeline.predict(X_test)
    res_lr = evaluate_forecast(y_test, y_pred_lr, model_name="Linear Regression")
    eval_results.append(res_lr)
    logger.info(f"Linear Regression Test MAE: {res_lr['mae_kwh']} kWh | RMSE: {res_lr['rmse_kwh']} kWh")
    
    # Save LR pipeline
    lr_path = os.path.join(models_dir, "linear_regression.joblib")
    joblib.dump(lr_pipeline, lr_path)
    trained_artifacts['linear_regression'] = lr_path

    # -------------------------------------------------------------
    # 4. Random Forest Regressor
    # -------------------------------------------------------------
    logger.info("Training Random Forest Regressor (n_estimators=100, max_depth=10, seed=42)...")
    rf_model = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    rf_model.fit(X_train, y_train)
    y_pred_rf = rf_model.predict(X_test)
    res_rf = evaluate_forecast(y_test, y_pred_rf, model_name="Random Forest Regressor")
    eval_results.append(res_rf)
    logger.info(f"Random Forest Test MAE: {res_rf['mae_kwh']} kWh | RMSE: {res_rf['rmse_kwh']} kWh")

    # Save RF model
    rf_path = os.path.join(models_dir, "random_forest.joblib")
    joblib.dump(rf_model, rf_path)
    trained_artifacts['random_forest'] = rf_path

    # Save metadata
    metadata = {
        "target": TARGET_COLUMN,
        "features": FEATURE_COLUMNS,
        "train_end": TRAIN_END_DATE,
        "val_end": VAL_END_DATE,
        "test_period": f"{test_df['date'].min().date()} to {test_df['date'].max().date()}",
        "random_state": 42
    }
    meta_path = os.path.join(models_dir, "model_metadata.json")
    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    # Comparison summary table
    comparison_df = compare_models(eval_results)
    metrics_path = os.path.join(outputs_dir, "metrics.csv")
    comparison_df.to_csv(metrics_path, index=False)
    logger.info(f"Saved model comparison table to: {metrics_path}")

    # Also save test predictions for plotting
    pred_export = test_df[['date', TARGET_COLUMN]].copy()
    pred_export['naive_pred'] = y_pred_naive.values
    pred_export['snaive_pred'] = y_pred_snaive.values
    pred_export['lr_pred'] = y_pred_lr
    pred_export['rf_pred'] = y_pred_rf
    pred_export.to_csv(os.path.join(outputs_dir, "test_predictions.csv"), index=False)

    return trained_artifacts, comparison_df


if __name__ == '__main__':
    default_csv = os.path.join("data", "processed", "daily_energy.csv")
    if os.path.exists(default_csv):
        artifacts, comp_table = train_and_evaluate_all(default_csv)
        print("\n--- Model Benchmark Evaluation Table ---")
        print(comp_table.to_string(index=False))
    else:
        print(f"File not found: {default_csv}. Run src/preprocess.py first.")
