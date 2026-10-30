"""
features.py
===========
Module for engineering calendar, autoregressive lag, and rolling window features
for daily energy forecasting with strict look-ahead bias prevention.

Preventing Data Leakage (Look-Ahead Bias):
-----------------------------------------
1. At forecast origin t, future target values y_{t+1}, y_{t+2}, ... are unknown.
2. Any rolling calculation at day t must ONLY summarize information up to day t-1.
   We achieve this deterministically via:
       df['daily_energy_kwh'].shift(1).rolling(window=7).mean()
   This guarantees day t's feature vector contains zero information from day t's consumption.
3. Calendar attributes (day of week, month, day of year) are known deterministically
   into the future, so they can be generated safely for any forecast horizon.
"""

import logging
from typing import List, Tuple
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

FEATURE_COLUMNS = [
    'dayofweek',
    'is_weekend',
    'month',
    'dayofyear',
    'sin_dow',
    'cos_dow',
    'sin_month',
    'cos_month',
    'lag_1',
    'lag_7',
    'lag_14',
    'lag_28',
    'rolling_7d_mean',
    'rolling_7d_std',
    'rolling_14d_mean'
]

TARGET_COLUMN = 'daily_energy_kwh'


def create_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extracts calendar and cyclical trigonometric features from the 'date' column.
    These features represent known structural calendar patterns (weekly rhythm,
    seasonal heating/cooling cycles).
    """
    df = df.copy()
    if not np.issubdtype(df['date'].dtype, np.datetime64):
        df['date'] = pd.to_datetime(df['date'])

    df['dayofweek'] = df['date'].dt.dayofweek  # 0 = Monday, 6 = Sunday
    df['is_weekend'] = (df['dayofweek'] >= 5).astype(int)
    df['month'] = df['date'].dt.month
    df['dayofyear'] = df['date'].dt.dayofyear
    df['year'] = df['date'].dt.year

    # Cyclical trigonometric transforms for periodic continuity
    # (e.g. Sunday day 6 wraps smoothly to Monday day 0)
    df['sin_dow'] = np.sin(2 * np.pi * df['dayofweek'] / 7.0)
    df['cos_dow'] = np.cos(2 * np.pi * df['dayofweek'] / 7.0)

    # Cyclical month transform (December month 12 wraps to January month 1)
    df['sin_month'] = np.sin(2 * np.pi * df['month'] / 12.0)
    df['cos_month'] = np.cos(2 * np.pi * df['month'] / 12.0)

    return df


def create_lag_features(df: pd.DataFrame, lags: List[int] = [1, 7, 14, 28]) -> pd.DataFrame:
    """
    Constructs autoregressive lag features for the daily energy consumption target.

    Lags:
    - lag_1: Immediate prior day (captures short-term persistence and thermal inertia).
    - lag_7: Same day of the previous week (captures weekly household routine).
    - lag_14: Same day 2 weeks ago (confirms weekly stability).
    - lag_28: Same day 4 weeks ago (~1 month prior, captures seasonal drift).
    """
    df = df.copy()
    for lag in lags:
        df[f'lag_{lag}'] = df[TARGET_COLUMN].shift(lag)
    return df


def create_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates rolling window statistics over past energy consumption.

    CRITICAL LEAKAGE PREVENTION:
    We apply .shift(1) BEFORE computing the rolling window:
        df[TARGET_COLUMN].shift(1).rolling(window=W).mean()
    This ensures that day t's rolling mean includes only days [t-W, ..., t-1],
    strictly excluding day t itself.
    """
    df = df.copy()
    
    # Strictly shifted historical series
    shifted_target = df[TARGET_COLUMN].shift(1)

    # 7-day rolling statistics
    df['rolling_7d_mean'] = shifted_target.rolling(window=7, min_periods=7).mean()
    df['rolling_7d_std'] = shifted_target.rolling(window=7, min_periods=7).std()

    # 14-day rolling mean for medium-term baseline trend
    df['rolling_14d_mean'] = shifted_target.rolling(window=14, min_periods=14).mean()

    return df


def engineer_all_features(df: pd.DataFrame, drop_incomplete: bool = True) -> pd.DataFrame:
    """
    Applies full feature engineering pipeline to daily dataset.

    Parameters:
    -----------
    df : pd.DataFrame
        Daily energy consumption dataframe (must contain 'date' and 'daily_energy_kwh').
    drop_incomplete : bool
        If True, filters out days marked is_eligible=False before computing continuous lags.
    
    Returns:
    --------
    feat_df : pd.DataFrame
        Dataframe containing calendar, lag, and rolling features, with unobservable
        initial rows (first 28 days) dropped.
    """
    df_sorted = df.sort_values('date').copy()

    if drop_incomplete and 'is_eligible' in df_sorted.columns:
        logger.info(f"Filtering eligible days (drop {len(df_sorted) - df_sorted['is_eligible'].sum()} incomplete days)...")
        # In practice, for daily time-series with sparse missing days, we can reindex with regular calendar
        # or filter to continuous eligible periods.
        df_sorted = df_sorted[df_sorted['is_eligible']].copy()

    # Calendar
    df_cal = create_calendar_features(df_sorted)

    # Autoregressive Lags
    df_lags = create_lag_features(df_cal, lags=[1, 7, 14, 28])

    # Rolling Windows (strictly shifted to prevent look-ahead bias)
    df_full = create_rolling_features(df_lags)

    # Drop the initial 28 rows that do not have 28 days of history
    initial_rows = len(df_full)
    feat_df = df_full.dropna(subset=['lag_28', 'rolling_14d_mean']).reset_index(drop=True)
    logger.info(f"Feature engineering complete: {len(feat_df)} rows ready (dropped {initial_rows - len(feat_df)} initial warmup days).")

    return feat_df


if __name__ == '__main__':
    import os
    sample_path = os.path.join("data", "processed", "daily_energy.csv")
    if os.path.exists(sample_path):
        df_in = pd.read_csv(sample_path)
        feat_df = engineer_all_features(df_in)
        print("Feature columns generated:")
        print(FEATURE_COLUMNS)
        print("\nFirst 3 feature rows:")
        print(feat_df[['date', TARGET_COLUMN] + FEATURE_COLUMNS[:6]].head(3))
    else:
        print(f"File not found: {sample_path}. Run src/preprocess.py first.")
