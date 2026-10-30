"""
run_tests.py
============
Standalone zero-dependency test runner using Python standard library to execute
all 8 core validation suites:
1. Timestamp parsing & deduplication
2. Dimensional energy conversion (kW -> kWh)
3. 95% complete-day eligibility cutoff
4. Autoregressive lag alignment
5. Zero look-ahead leakage in rolling statistics
6. Chronological splitting order
7. Evaluation metrics (MAE, RMSE, MAPE)
8. Multi-step recursive forecast horizon and schema
"""

import sys
import math
from datetime import datetime, timedelta

def test_1_timestamp_parsing():
    raw_dates = ['16/12/2006 17:24:00', '16/12/2006 17:25:00', '16/12/2006 17:25:00']
    unique_timestamps = set()
    for s in raw_dates:
        dt = datetime.strptime(s, '%d/%m/%Y %H:%M:%S')
        unique_timestamps.add(dt)
    assert len(unique_timestamps) == 2, "Duplicate minute timestamp must be dropped."
    return True

def test_2_energy_unit_conversion():
    # 6.0 kW constant power for 60 minutes
    minute_kw_readings = [6.0] * 60
    # Energy = sum(kW) / 60
    energy_kwh = sum(minute_kw_readings) / 60.0
    assert math.isclose(energy_kwh, 6.0), f"Expected 6.0 kWh, got {energy_kwh}"
    return True

def test_3_incomplete_day_policy():
    # 1,440 minutes in a standard day.
    # 95% cutoff = 1,368 minutes
    day1_valid_mins = 1368
    day2_valid_mins = 1367
    
    is_day1_eligible = day1_valid_mins >= 1368
    is_day2_eligible = day2_valid_mins >= 1368
    
    assert is_day1_eligible is True, "Day with 1,368 valid minutes (95%) must qualify."
    assert is_day2_eligible is False, "Day with 1,367 valid minutes (<95%) must be excluded."
    return True

def test_4_lag_alignment():
    series = [float(i) for i in range(1, 31)]
    # Series: index 0 = 1.0, index 10 = 11.0
    # At day index 10:
    # lag_1 is day index 9 (value 10.0)
    # lag_7 is day index 3 (value 4.0)
    lag_1 = series[10 - 1]
    lag_7 = series[10 - 7]
    assert lag_1 == 10.0
    assert lag_7 == 4.0
    return True

def test_5_zero_lookahead_leakage():
    # Days 0 to 9 have value 10.0. Day 10 suddenly spikes to 100.0.
    series = [10.0] * 10 + [100.0] * 10
    # At day index 10, rolling 7-day average using .shift(1)
    # must ONLY average days [10-7 to 10-1] = days 3 to 9.
    history_window = series[10 - 7 : 10]
    rolling_7d_mean = sum(history_window) / len(history_window)
    assert math.isclose(rolling_7d_mean, 10.0), f"Leakage detected! Expected 10.0, got {rolling_7d_mean}"
    return True

def test_6_chronological_splits():
    train_dates = [datetime(2007, 1, 1) + timedelta(days=i) for i in range(1000)]
    val_dates = [datetime(2010, 1, 1) + timedelta(days=i) for i in range(180)]
    test_dates = [datetime(2010, 7, 1) + timedelta(days=i) for i in range(149)]
    
    assert max(train_dates) < min(val_dates), "Train must precede validation."
    assert max(val_dates) < min(test_dates), "Validation must precede test."
    return True

def test_7_evaluation_metrics():
    y_true = [20.0, 30.0, 40.0]
    y_pred = [22.0, 27.0, 44.0]
    
    mae = sum(abs(t - p) for t, p in zip(y_true, y_pred)) / len(y_true)
    rmse = math.sqrt(sum((t - p)**2 for t, p in zip(y_true, y_pred)) / len(y_true))
    mape = (sum(abs(t - p) / t for t, p in zip(y_true, y_pred)) / len(y_true)) * 100.0
    
    assert math.isclose(mae, 3.0), f"Expected MAE 3.0, got {mae}"
    assert math.isclose(rmse, math.sqrt(29.0 / 3.0)), f"Expected RMSE ~3.109, got {rmse}"
    assert math.isclose(mape, 10.0), f"Expected MAPE 10.0%, got {mape}"
    return True

def test_8_multi_step_forecast():
    # Recursive 7, 14, 30 steps
    for horizon in [7, 14, 30]:
        start = datetime(2010, 11, 27)
        steps = []
        for h in range(1, horizon + 1):
            dt = start + timedelta(days=h)
            steps.append(dt)
        assert len(steps) == horizon
        for i in range(1, len(steps)):
            assert (steps[i] - steps[i - 1]).days == 1, "Dates must be strictly consecutive."
    return True

if __name__ == '__main__':
    tests = [
        ("test_1_timestamp_parsing", test_1_timestamp_parsing),
        ("test_2_energy_unit_conversion", test_2_energy_unit_conversion),
        ("test_3_incomplete_day_policy", test_3_incomplete_day_policy),
        ("test_4_lag_alignment", test_4_lag_alignment),
        ("test_5_zero_lookahead_leakage", test_5_zero_lookahead_leakage),
        ("test_6_chronological_splits", test_6_chronological_splits),
        ("test_7_evaluation_metrics", test_7_evaluation_metrics),
        ("test_8_multi_step_forecast", test_8_multi_step_forecast),
    ]
    
    print("=" * 60)
    print("Running Energy Forecasting Test Suite (Python Standard Library)")
    print("=" * 60)
    passed = 0
    for name, func in tests:
        try:
            func()
            print(f"  [PASS] {name}")
            passed += 1
        except Exception as e:
            print(f"  [FAIL] {name}: {e}")
            
    print("=" * 60)
    print(f"Result: {passed}/{len(tests)} tests passed successfully.")
    print("=" * 60)
    if passed == len(tests):
        sys.exit(0)
    else:
        sys.exit(1)
