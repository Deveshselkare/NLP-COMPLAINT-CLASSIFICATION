"""
evaluate.py
===========
Evaluation metrics and validation routines for time series energy forecasting.

Metrics:
- Mean Absolute Error (MAE): Measures average magnitude of errors in natural units (kWh).
- Root Mean Squared Error (RMSE): Penalizes large outlier forecast errors heavily (kWh).
- Mean Absolute Percentage Error (MAPE): Relative percentage error, with documented zero/near-zero guard.
"""

from typing import Dict, Any, Union
import numpy as np
import pandas as pd


def mean_absolute_error(y_true: Union[np.ndarray, pd.Series], y_pred: Union[np.ndarray, pd.Series]) -> float:
    """Calculates MAE in kWh."""
    y_t = np.asarray(y_true)
    y_p = np.asarray(y_pred)
    return float(np.mean(np.abs(y_t - y_p)))


def root_mean_squared_error(y_true: Union[np.ndarray, pd.Series], y_pred: Union[np.ndarray, pd.Series]) -> float:
    """Calculates RMSE in kWh."""
    y_t = np.asarray(y_true)
    y_p = np.asarray(y_pred)
    return float(np.sqrt(np.mean((y_t - y_p) ** 2)))


def mean_absolute_percentage_error(
    y_true: Union[np.ndarray, pd.Series],
    y_pred: Union[np.ndarray, pd.Series],
    epsilon: float = 1e-4
) -> float:
    """
    Calculates MAPE (%).
    
    Zero/Near-Zero Policy:
    ----------------------
    In domestic energy consumption, true consumption can approach near zero during
    vacation or vacant days. Dividing by tiny values causes numerical explosion.
    We apply a safety denominator floor (epsilon = 1e-4 kWh) and document this policy.
    """
    y_t = np.asarray(y_true, dtype=float)
    y_p = np.asarray(y_pred, dtype=float)
    denom = np.maximum(np.abs(y_t), epsilon)
    return float(np.mean(np.abs((y_t - y_p) / denom)) * 100.0)


def evaluate_forecast(
    y_true: Union[np.ndarray, pd.Series],
    y_pred: Union[np.ndarray, pd.Series],
    model_name: str = "Model"
) -> Dict[str, Any]:
    """
    Computes comprehensive metric report for a forecast series.
    """
    mae = mean_absolute_error(y_true, y_pred)
    rmse = root_mean_squared_error(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred)

    return {
        "model": model_name,
        "mae_kwh": round(mae, 3),
        "rmse_kwh": round(rmse, 3),
        "mape_pct": round(mape, 2),
        "n_eval_samples": len(y_true)
    }


def compare_models(results: list) -> pd.DataFrame:
    """Formats a list of evaluation metric dictionaries into a clean summary table."""
    df = pd.DataFrame(results)
    if 'mae_kwh' in df.columns:
        df = df.sort_values('mae_kwh').reset_index(drop=True)
    return df
