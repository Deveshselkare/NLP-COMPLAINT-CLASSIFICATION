"""
test_pipeline.py
================
Automated test suite using pytest for validating data integrity, energy conversion math,
zero look-ahead bias in feature engineering, chronological splits, and multi-step forecasting.

Run tests:
    pytest tests/ -v
"""

import os
import tempfile
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression

from src.preprocess import parse_timestamps, aggregate_daily_energy, EXPECTED_MINUTES_PER_DAY
from src.features import create_calendar_features, create_lag_features, create_rolling_features, TARGET_COLUMN
from src.train import split_chronologically
from src.evaluate import mean_absolute_error, root_mean_squared_error, mean_absolute_percentage_error
from src.forecast import generate_recursive_forecast


# -----------------------------------------------------------------------------
# Test 1: Timestamp parsing and deduplication
# -----------------------------------------------------------------------------
def test_timestamp_parsing_and_deduplication():
    raw_data = {
        'Date': ['16/12/2006', '16/12/2006', '16/12/2006'],
        'Time': ['17:24:00', '17:25:00', '17:25:00'],  # Contains duplicate 17:25:00
        'Global_active_power': [4.216, 5.360, 5.360],
        'Global_reactive_power': [0.418, 0.436, 0.436],
        'Voltage': [234.84, 233.63, 233.63],
        'Global_intensity': [18.4, 23.0, 23.0]
    }
    df = pd.DataFrame(raw_data)
    parsed = parse_timestamps(df)

    assert len(parsed) == 2, "Duplicate minute timestamp should be dropped."
    assert pd.api.types.is_datetime64_any_dtype(parsed['datetime']), "datetime column must be parsed to datetime64."
    assert parsed.iloc[0]['datetime'] == pd.Timestamp('2006-12-16 17:24:00')


# -----------------------------------------------------------------------------
# Test 2: Dimensional Energy Conversion (kW -> kWh)
# -----------------------------------------------------------------------------
def test_energy_unit_conversion_math():
    """
    Verifies that active power in kW sampled every minute converts correctly to kWh:
    Energy = sum(kW) / 60
    If a device draws exactly 6 kW for 60 minutes:
        Energy = (60 * 6) / 60 = 6.0 kWh.
    """
    # Create 60 minutes of 6.0 kW constant power
    times = pd.date_range('2020-01-01 00:00:00', periods=60, freq='1min')
    df = pd.DataFrame({
        'datetime': times,
        'Global_active_power': [6.0] * 60,
        'Voltage': [230.0] * 60,
        'Global_intensity': [26.0] * 60
    })
    daily_stats, _ = aggregate_daily_energy(df, coverage_threshold=0.01)

    assert len(daily_stats) == 1
    # 60 * 6.0 kW = 360 kW-minutes. Divided by 60 = 6.0 kWh.
    assert np.isclose(daily_stats.iloc[0]['daily_energy_kwh'], 6.0), (
        f"Expected 6.0 kWh, got {daily_stats.iloc[0]['daily_energy_kwh']}"
    )


# -----------------------------------------------------------------------------
# Test 3: Incomplete Day Eligibility Policy (95% coverage cutoff)
# -----------------------------------------------------------------------------
def test_incomplete_day_policy():
    """
    1,440 minutes in a standard day.
    - Day 1: 1,368 valid minutes (exactly 95.0%) -> MUST BE ELIGIBLE
    - Day 2: 1,367 valid minutes (94.93%) -> MUST BE EXCLUDED (is_eligible = False)
    """
    day1_times = pd.date_range('2020-01-01 00:00:00', periods=1368, freq='1min')
    day2_times = pd.date_range('2020-01-02 00:00:00', periods=1367, freq='1min')

    df = pd.DataFrame({
        'datetime': list(day1_times) + list(day2_times),
        'Global_active_power': [1.5] * (1368 + 1367),
        'Voltage': [230.0] * (1368 + 1367),
        'Global_intensity': [6.5] * (1368 + 1367)
    })
    daily_stats, summary = aggregate_daily_energy(df, coverage_threshold=0.95)

    assert len(daily_stats) == 2
    row_day1 = daily_stats[daily_stats['date'] == pd.Timestamp('2020-01-01')].iloc[0]
    row_day2 = daily_stats[daily_stats['date'] == pd.Timestamp('2020-01-02')].iloc[0]

    assert row_day1['is_eligible'] == True, "Day with 1368 minutes (95%) must qualify as eligible."
    assert row_day2['is_eligible'] == False, "Day with 1367 minutes (<95%) must be marked ineligible."
    assert summary['qualified_days'] == 1
    assert summary['excluded_incomplete_days'] == 1


# -----------------------------------------------------------------------------
# Test 4: Lag Features Alignment
# -----------------------------------------------------------------------------
def test_lag_features_alignment():
    dates = pd.date_range('2020-01-01', periods=30, freq='D')
    # Synthetic values 1, 2, ..., 30
    df = pd.DataFrame({
        'date': dates,
        'daily_energy_kwh': np.arange(1.0, 31.0)
    })
    lagged = create_lag_features(df, lags=[1, 7, 14, 28])

    # For day index 10 (value = 11.0 on day 11):
    # lag_1 should be 10.0 (day 10)
    # lag_7 should be 4.0 (day 4)
    assert lagged.iloc[10]['lag_1'] == 10.0
    assert lagged.iloc[10]['lag_7'] == 4.0
    assert lagged.iloc[28]['lag_28'] == 1.0


# -----------------------------------------------------------------------------
# Test 5: Leakage Prevention in Rolling Window Statistics
# -----------------------------------------------------------------------------
def test_rolling_feature_zero_leakage():
    """
    CRITICAL: Verifies that rolling_7d_mean at row t uses ONLY rows [t-7, ..., t-1],
    and does NOT incorporate the target value of row t.
    """
    dates = pd.date_range('2020-01-01', periods=20, freq='D')
    # Day 1-10 have consumption 10.0. Day 11 suddenly spikes to 100.0.
    values = [10.0] * 10 + [100.0] * 10
    df = pd.DataFrame({
        'date': dates,
        'daily_energy_kwh': values
    })
    rolled = create_rolling_features(df)

    # At row index 10 (Day 11, where value jumped to 100.0):
    # If there is NO lookahead leakage, the rolling mean of the past 7 days must still be 10.0!
    # If the current day leaked into the rolling window, the mean would be > 10.0.
    assert np.isclose(rolled.iloc[10]['rolling_7d_mean'], 10.0), (
        f"Data leakage detected! At row index 10, expected rolling 7d mean 10.0, but got {rolled.iloc[10]['rolling_7d_mean']}"
    )


# -----------------------------------------------------------------------------
# Test 6: Strict Chronological Splitting (No Shuffling)
# -----------------------------------------------------------------------------
def test_chronological_splitting_ordering():
    dates = pd.date_range('2009-01-01', '2010-10-01', freq='D')
    df = pd.DataFrame({
        'date': dates,
        'daily_energy_kwh': np.random.uniform(15.0, 35.0, size=len(dates))
    })
    train_df, val_df, test_df = split_chronologically(df, train_end='2009-12-31', val_end='2010-05-31')

    # Invariants
    assert train_df['date'].max() < val_df['date'].min(), "Train must strictly precede validation."
    assert val_df['date'].max() < test_df['date'].min(), "Validation must strictly precede test."
    assert len(train_df) + len(val_df) + len(test_df) == len(df), "No rows should be dropped during partition."


# -----------------------------------------------------------------------------
# Test 7: Evaluation Metrics Correctness
# -----------------------------------------------------------------------------
def test_evaluation_metrics():
    y_true = np.array([20.0, 30.0, 40.0])
    y_pred = np.array([22.0, 27.0, 44.0])  # Errors: +2, -3, +4

    # MAE = (2 + 3 + 4) / 3 = 3.0
    mae = mean_absolute_error(y_true, y_pred)
    assert np.isclose(mae, 3.0)

    # RMSE = sqrt((4 + 9 + 16)/3) = sqrt(29/3) ~= 3.1091
    rmse = root_mean_squared_error(y_true, y_pred)
    assert np.isclose(rmse, np.sqrt(29.0 / 3.0))

    # MAPE = ( (2/20) + (3/30) + (4/40) ) / 3 * 100 = (0.1 + 0.1 + 0.1)/3 * 100 = 10.0%
    mape = mean_absolute_percentage_error(y_true, y_pred)
    assert np.isclose(mape, 10.0)


# -----------------------------------------------------------------------------
# Test 8: Multi-Step Recursive Forecast Horizon and Schema
# -----------------------------------------------------------------------------
def test_recursive_forecast_generation():
    # Synthetic model
    class DummyModel:
        def predict(self, X):
            # Returns mean of lag_1 and lag_7
            return (X['lag_1'].values + X['lag_7'].values) / 2.0

    dates = pd.date_range('2020-01-01', periods=40, freq='D')
    history = pd.DataFrame({
        'date': dates,
        'daily_energy_kwh': np.linspace(20.0, 25.0, 40)
    })

    # Test 7, 14, and 30-day horizons
    for horizon in [7, 14, 30]:
        fc = generate_recursive_forecast(DummyModel(), history, horizon_days=horizon)
        assert len(fc) == horizon, f"Expected {horizon} forecast rows, got {len(fc)}."
        assert list(fc.columns) == ['date', 'step', 'forecast_energy_kwh', 'lower_bound_kwh', 'upper_bound_kwh', 'model']
        # Check consecutive dates
        fc_dates = pd.to_datetime(fc['date'])
        date_diffs = fc_dates.diff()[1:]
        assert all(diff == pd.Timedelta(days=1) for diff in date_diffs), "Forecast dates must be consecutive days."
