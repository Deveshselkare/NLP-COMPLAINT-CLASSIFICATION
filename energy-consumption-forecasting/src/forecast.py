"""
forecast.py
===========
Multi-step recursive time series forecasting for daily energy consumption
across 7, 14, and 30-day forecast horizons.

How Recursive Forecasting Works:
--------------------------------
1. Forecast origin: latest available historical day T.
2. Step k=1:
   - Calendar features for date T+1 are known exactly.
   - Lag features (lag_1, lag_7, lag_14, lag_28) are drawn from actual history.
   - Rolling 7-day mean uses the last 7 observed days.
   - Model outputs prediction y_hat_{T+1}.
3. Step k=2:
   - Date T+2 features: lag_1 is now the PREVIOUS PREDICTION y_hat_{T+1}.
   - Lag_7, lag_14, lag_28 are drawn from actual history.
   - Rolling 7-day mean incorporates y_hat_{T+1}.
   - Model outputs y_hat_{T+2}.
4. Repeat recursively up to horizon H in {7, 14, 30}.

Error Accumulation:
-------------------
Because subsequent forecast steps consume prior predicted values rather than true ground-truth,
errors compound recursively over time. This module computes widening uncertainty confidence bands
based on empirical test residuals:
    sigma_k = sigma_test * sqrt(1 + (k - 1) * 0.08)
"""

import os
import json
import logging
from typing import List, Dict, Any, Optional
import numpy as np
import pandas as pd
import joblib

from src.features import create_calendar_features, FEATURE_COLUMNS, TARGET_COLUMN

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def generate_recursive_forecast(
    model: Any,
    history_df: pd.DataFrame,
    horizon_days: int = 30,
    model_name: str = "Random Forest Regressor",
    empirical_residual_std: float = 4.2
) -> pd.DataFrame:
    """
    Generates multi-step recursive forecasts for specified horizon.

    Parameters:
    -----------
    model : trained scikit-learn model or pipeline
    history_df : pd.DataFrame with past 'date' and 'daily_energy_kwh' (at least 35 days)
    horizon_days : int in [7, 14, 30]
    model_name : str identifier
    empirical_residual_std : float out-of-sample residual standard deviation in kWh
    """
    hist = history_df.sort_values('date').copy()
    hist['date'] = pd.to_datetime(hist['date'])

    # Verify minimum required history
    if len(hist) < 35:
        raise ValueError(f"Need at least 35 days of history for lag_28 and rolling_7d, got {len(hist)}.")

    # Working list of daily energy history [..., y_{T-1}, y_T]
    simulated_series = hist[TARGET_COLUMN].tolist()
    last_date = hist['date'].max()

    forecast_records = []

    for step in range(1, horizon_days + 1):
        target_date = last_date + pd.Timedelta(days=step)

        # 1. Calendar features for future date
        row_dict = {
            'date': target_date,
            'dayofweek': target_date.dayofweek,
            'is_weekend': int(target_date.dayofweek >= 5),
            'month': target_date.month,
            'dayofyear': target_date.dayofyear,
            'year': target_date.year,
            'sin_dow': np.sin(2 * np.pi * target_date.dayofweek / 7.0),
            'cos_dow': np.cos(2 * np.pi * target_date.dayofweek / 7.0),
            'sin_month': np.sin(2 * np.pi * target_date.month / 12.0),
            'cos_month': np.cos(2 * np.pi * target_date.month / 12.0),
        }

        # 2. Lag features dynamically indexed from simulated_series
        # -1 corresponds to the most recent day (yesterday relative to target_date)
        row_dict['lag_1'] = simulated_series[-1]
        row_dict['lag_7'] = simulated_series[-7]
        row_dict['lag_14'] = simulated_series[-14]
        row_dict['lag_28'] = simulated_series[-28]

        # 3. Shifted rolling statistics (last 7 and 14 days of simulated series)
        recent_7 = simulated_series[-7:]
        recent_14 = simulated_series[-14:]
        row_dict['rolling_7d_mean'] = float(np.mean(recent_7))
        row_dict['rolling_7d_std'] = float(np.std(recent_7, ddof=1)) if len(recent_7) > 1 else 0.0
        row_dict['rolling_14d_mean'] = float(np.mean(recent_14))

        # Build feature vector
        X_step = pd.DataFrame([row_dict])[FEATURE_COLUMNS]

        # 4. Predict next day consumption
        pred_kwh = float(model.predict(X_step)[0])
        # Physical energy lower bound: consumption cannot be negative
        pred_kwh = max(0.5, pred_kwh)

        # 5. Append prediction to simulated history for future recursive steps
        simulated_series.append(pred_kwh)

        # 6. Uncertainty expansion over recursive horizon (95% prediction interval)
        step_uncertainty = empirical_residual_std * np.sqrt(1.0 + (step - 1) * 0.08)
        lower_bound = max(0.0, pred_kwh - 1.96 * step_uncertainty)
        upper_bound = pred_kwh + 1.96 * step_uncertainty

        forecast_records.append({
            'date': target_date.strftime('%Y-%m-%d'),
            'step': step,
            'forecast_energy_kwh': round(pred_kwh, 2),
            'lower_bound_kwh': round(lower_bound, 2),
            'upper_bound_kwh': round(upper_bound, 2),
            'model': model_name
        })

    forecast_df = pd.DataFrame(forecast_records)
    logger.info(f"Generated {horizon_days}-day recursive forecast using {model_name}.")
    return forecast_df


def save_forecast(forecast_df: pd.DataFrame, output_path: str = "outputs/forecast.csv") -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    forecast_df.to_csv(output_path, index=False)
    logger.info(f"Saved forecast results to: {output_path}")


if __name__ == '__main__':
    data_path = os.path.join("data", "processed", "daily_energy.csv")
    model_path = os.path.join("models", "random_forest.joblib")
    
    if os.path.exists(data_path) and os.path.exists(model_path):
        history = pd.read_csv(data_path)
        rf_model = joblib.load(model_path)
        for h in [7, 14, 30]:
            fc_df = generate_recursive_forecast(rf_model, history, horizon_days=h)
            out_file = f"outputs/forecast_{h}d.csv"
            save_forecast(fc_df, out_file)
            print(f"\n--- {h}-Day Forecast Preview ---")
            print(fc_df.head(5).to_string(index=False))
    else:
        print(f"Run src/preprocess.py and src/train.py first to create data and models.")
