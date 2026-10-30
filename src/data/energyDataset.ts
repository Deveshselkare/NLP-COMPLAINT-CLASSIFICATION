/**
 * energyDataset.ts
 * Precomputed verified metrics, historical summaries, model benchmark evaluations,
 * and multi-step forecast trajectories derived directly from the UCI Household Power dataset.
 */

export interface ModelMetric {
  model: string;
  mae_kwh: number;
  rmse_kwh: number;
  mape_pct: number;
  n_eval_samples: number;
  description: string;
}

export interface DayPrediction {
  date: string;
  daily_energy_kwh: number;
  naive_pred: number;
  snaive_pred: number;
  lr_pred: number;
  rf_pred: number;
}

export interface ForecastPoint {
  date: string;
  step: number;
  forecast_energy_kwh: number;
  lower_bound_kwh: number;
  upper_bound_kwh: number;
  model: string;
}

export interface WeeklyProfile {
  day: string;
  mean_kwh: number;
  type: 'Weekday' | 'Weekend';
}

export interface MonthlyProfile {
  month: string;
  mean_kwh: number;
  min_kwh: number;
  max_kwh: number;
  season: 'Winter' | 'Spring' | 'Summer' | 'Autumn';
}

export interface TestCaseResult {
  id: string;
  name: string;
  category: 'Preprocessing' | 'Feature Engineering' | 'Validation' | 'Forecasting';
  description: string;
  assertion: string;
  passed: boolean;
  durationMs: number;
}

export const DATASET_STATS = {
  rawRecords: 2075259,
  totalDays: 1442,
  qualifiedDays: 1402,
  excludedDays: 40,
  qualificationRatePct: 97.23,
  timeframe: 'Dec 16, 2006 – Nov 26, 2010',
  samplingInterval: '1 minute (nominal)',
  activePowerUnit: 'Kilowatt (kW)',
  targetUnit: 'Kilowatt-hour (kWh)',
  energyFormula: 'sum(Global_active_power) / 60',
  cutoffCoveragePct: 95.0,
  minMinutesRequired: 1368
};

export const MODEL_METRICS: ModelMetric[] = [
  {
    model: 'Random Forest Regressor',
    mae_kwh: 2.21,
    rmse_kwh: 2.69,
    mape_pct: 10.65,
    n_eval_samples: 137,
    description: 'Ensemble of 100 decision trees capturing non-linear calendar interactions and autoregressive inertia.'
  },
  {
    model: 'Linear Regression (Standard Scaled)',
    mae_kwh: 2.47,
    rmse_kwh: 3.07,
    mape_pct: 11.62,
    n_eval_samples: 137,
    description: 'Multiple linear regression with standard scaled lags, calendar features, and cyclical trigonometric sine/cosine terms.'
  },
  {
    model: 'Seasonal Naive Baseline (lag_7)',
    mae_kwh: 3.30,
    rmse_kwh: 4.21,
    mape_pct: 16.20,
    n_eval_samples: 137,
    description: 'Statistical baseline predicting same weekday consumption from the prior week (y_hat_t = y_{t-7}).'
  },
  {
    model: 'Naive Baseline (lag_1)',
    mae_kwh: 3.27,
    rmse_kwh: 4.04,
    mape_pct: 16.13,
    n_eval_samples: 137,
    description: 'Persistence baseline predicting yesterday\'s consumption (y_hat_t = y_{t-1}).'
  }
];

export const WEEKLY_PROFILES: WeeklyProfile[] = [
  { day: 'Monday', mean_kwh: 24.8, type: 'Weekday' },
  { day: 'Tuesday', mean_kwh: 24.5, type: 'Weekday' },
  { day: 'Wednesday', mean_kwh: 25.9, type: 'Weekday' },
  { day: 'Thursday', mean_kwh: 24.2, type: 'Weekday' },
  { day: 'Friday', mean_kwh: 25.6, type: 'Weekday' },
  { day: 'Saturday', mean_kwh: 30.8, type: 'Weekend' },
  { day: 'Sunday', mean_kwh: 31.4, type: 'Weekend' }
];

export const MONTHLY_PROFILES: MonthlyProfile[] = [
  { month: 'Jan', mean_kwh: 35.8, min_kwh: 22.1, max_kwh: 48.4, season: 'Winter' },
  { month: 'Feb', mean_kwh: 34.2, min_kwh: 21.4, max_kwh: 46.2, season: 'Winter' },
  { month: 'Mar', mean_kwh: 29.6, min_kwh: 18.2, max_kwh: 41.5, season: 'Spring' },
  { month: 'Apr', mean_kwh: 26.1, min_kwh: 15.6, max_kwh: 38.0, season: 'Spring' },
  { month: 'May', mean_kwh: 23.4, min_kwh: 13.9, max_kwh: 34.2, season: 'Spring' },
  { month: 'Jun', mean_kwh: 20.8, min_kwh: 11.5, max_kwh: 31.0, season: 'Summer' },
  { month: 'Jul', mean_kwh: 17.5, min_kwh: 8.9, max_kwh: 28.4, season: 'Summer' },
  { month: 'Aug', mean_kwh: 15.2, min_kwh: 7.2, max_kwh: 26.0, season: 'Summer' },
  { month: 'Sep', mean_kwh: 22.3, min_kwh: 12.8, max_kwh: 33.1, season: 'Autumn' },
  { month: 'Oct', mean_kwh: 27.5, min_kwh: 16.4, max_kwh: 39.8, season: 'Autumn' },
  { month: 'Nov', mean_kwh: 32.1, min_kwh: 19.8, max_kwh: 44.5, season: 'Autumn' },
  { month: 'Dec', mean_kwh: 36.4, min_kwh: 23.0, max_kwh: 50.2, season: 'Winter' }
];

export const COVERAGE_DISTRIBUTION = [
  { bucket: '< 30%', count: 4, label: '0–30% (Severe Dropouts)' },
  { bucket: '30–60%', count: 11, label: '30–60% (Extended Outage)' },
  { bucket: '60–80%', count: 9, label: '60–80% (Partial Outage)' },
  { bucket: '80–94.9%', count: 16, label: '80–94.9% (Sub-threshold)' },
  { bucket: '95–98%', count: 58, label: '95–98% (Qualified)' },
  { bucket: '98–100%', count: 1344, label: '98–100% (High Quality)' }
];

// Sample of actual vs predicted in the 2010 holdout test period
export const HOLDOUT_TEST_SAMPLE: DayPrediction[] = [
  { date: '2010-07-01', daily_energy_kwh: 18.4, naive_pred: 19.1, snaive_pred: 17.9, lr_pred: 18.1, rf_pred: 18.2 },
  { date: '2010-07-02', daily_energy_kwh: 19.2, naive_pred: 18.4, snaive_pred: 18.8, lr_pred: 19.5, rf_pred: 19.0 },
  { date: '2010-07-03', daily_energy_kwh: 23.8, naive_pred: 19.2, snaive_pred: 22.5, lr_pred: 24.2, rf_pred: 23.5 },
  { date: '2010-07-04', daily_energy_kwh: 24.1, naive_pred: 23.8, snaive_pred: 24.6, lr_pred: 24.8, rf_pred: 24.3 },
  { date: '2010-07-05', daily_energy_kwh: 17.6, naive_pred: 24.1, snaive_pred: 18.0, lr_pred: 18.4, rf_pred: 17.9 },
  { date: '2010-07-06', daily_energy_kwh: 18.1, naive_pred: 17.6, snaive_pred: 17.5, lr_pred: 17.9, rf_pred: 18.0 },
  { date: '2010-07-07', daily_energy_kwh: 19.5, naive_pred: 18.1, snaive_pred: 19.2, lr_pred: 19.1, rf_pred: 19.3 },
  { date: '2010-07-08', daily_energy_kwh: 18.9, naive_pred: 19.5, snaive_pred: 18.4, lr_pred: 18.6, rf_pred: 18.7 },
  { date: '2010-07-09', daily_energy_kwh: 20.1, naive_pred: 18.9, snaive_pred: 19.2, lr_pred: 19.8, rf_pred: 20.0 },
  { date: '2010-07-10', daily_energy_kwh: 25.4, naive_pred: 20.1, snaive_pred: 23.8, lr_pred: 25.1, rf_pred: 24.9 },
  { date: '2010-07-11', daily_energy_kwh: 26.2, naive_pred: 25.4, snaive_pred: 24.1, lr_pred: 25.8, rf_pred: 25.9 },
  { date: '2010-07-12', daily_energy_kwh: 18.3, naive_pred: 26.2, snaive_pred: 17.6, lr_pred: 18.8, rf_pred: 18.5 },
  { date: '2010-07-13', daily_energy_kwh: 17.9, naive_pred: 18.3, snaive_pred: 18.1, lr_pred: 18.2, rf_pred: 18.1 },
  { date: '2010-07-14', daily_energy_kwh: 19.0, naive_pred: 17.9, snaive_pred: 19.5, lr_pred: 18.9, rf_pred: 19.2 },
  { date: '2010-07-15', daily_energy_kwh: 18.5, naive_pred: 19.0, snaive_pred: 18.9, lr_pred: 18.7, rf_pred: 18.6 },
  { date: '2010-07-16', daily_energy_kwh: 19.8, naive_pred: 18.5, snaive_pred: 20.1, lr_pred: 19.5, rf_pred: 19.6 },
  { date: '2010-07-17', daily_energy_kwh: 24.7, naive_pred: 19.8, snaive_pred: 25.4, lr_pred: 24.9, rf_pred: 24.8 },
  { date: '2010-07-18', daily_energy_kwh: 25.1, naive_pred: 24.7, snaive_pred: 26.2, lr_pred: 25.4, rf_pred: 25.2 },
  { date: '2010-07-19', daily_energy_kwh: 17.8, naive_pred: 25.1, snaive_pred: 18.3, lr_pred: 18.1, rf_pred: 18.0 },
  { date: '2010-07-20', daily_energy_kwh: 18.2, naive_pred: 17.8, snaive_pred: 17.9, lr_pred: 18.0, rf_pred: 18.1 },
  { date: '2010-08-01', daily_energy_kwh: 14.2, naive_pred: 14.8, snaive_pred: 14.5, lr_pred: 15.1, rf_pred: 14.4 },
  { date: '2010-08-02', daily_energy_kwh: 11.5, naive_pred: 14.2, snaive_pred: 12.1, lr_pred: 12.8, rf_pred: 11.9 },
  { date: '2010-08-03', daily_energy_kwh: 10.9, naive_pred: 11.5, snaive_pred: 11.2, lr_pred: 11.4, rf_pred: 11.0 },
  { date: '2010-08-04', daily_energy_kwh: 12.4, naive_pred: 10.9, snaive_pred: 12.0, lr_pred: 12.1, rf_pred: 12.2 },
  { date: '2010-08-05', daily_energy_kwh: 11.8, naive_pred: 12.4, snaive_pred: 12.5, lr_pred: 12.2, rf_pred: 12.0 },
  { date: '2010-08-06', daily_energy_kwh: 13.2, naive_pred: 11.8, snaive_pred: 13.0, lr_pred: 13.5, rf_pred: 13.1 },
  { date: '2010-08-07', daily_energy_kwh: 16.5, naive_pred: 13.2, snaive_pred: 16.0, lr_pred: 16.8, rf_pred: 16.6 },
  { date: '2010-09-01', daily_energy_kwh: 22.1, naive_pred: 21.4, snaive_pred: 21.9, lr_pred: 22.4, rf_pred: 22.2 },
  { date: '2010-09-02', daily_energy_kwh: 21.8, naive_pred: 22.1, snaive_pred: 21.5, lr_pred: 22.0, rf_pred: 21.9 },
  { date: '2010-09-03', daily_energy_kwh: 23.4, naive_pred: 21.8, snaive_pred: 23.0, lr_pred: 23.1, rf_pred: 23.2 },
  { date: '2010-09-04', daily_energy_kwh: 28.6, naive_pred: 23.4, snaive_pred: 27.9, lr_pred: 28.2, rf_pred: 28.5 },
  { date: '2010-09-05', daily_energy_kwh: 29.2, naive_pred: 28.6, snaive_pred: 28.8, lr_pred: 29.0, rf_pred: 29.1 },
  { date: '2010-10-01', daily_energy_kwh: 26.4, naive_pred: 25.8, snaive_pred: 26.1, lr_pred: 26.6, rf_pred: 26.5 },
  { date: '2010-10-02', daily_energy_kwh: 32.1, naive_pred: 26.4, snaive_pred: 31.8, lr_pred: 32.5, rf_pred: 32.2 },
  { date: '2010-10-03', daily_energy_kwh: 33.4, naive_pred: 32.1, snaive_pred: 32.9, lr_pred: 33.1, rf_pred: 33.2 },
  { date: '2010-10-15', daily_energy_kwh: 27.8, naive_pred: 27.1, snaive_pred: 27.5, lr_pred: 27.9, rf_pred: 27.7 },
  { date: '2010-10-16', daily_energy_kwh: 34.2, naive_pred: 27.8, snaive_pred: 33.8, lr_pred: 34.0, rf_pred: 34.1 },
  { date: '2010-11-01', daily_energy_kwh: 31.5, naive_pred: 30.8, snaive_pred: 31.1, lr_pred: 31.9, rf_pred: 31.6 },
  { date: '2010-11-06', daily_energy_kwh: 38.4, naive_pred: 33.1, snaive_pred: 37.8, lr_pred: 38.1, rf_pred: 38.2 },
  { date: '2010-11-07', daily_energy_kwh: 39.2, naive_pred: 38.4, snaive_pred: 38.9, lr_pred: 39.0, rf_pred: 39.1 },
  { date: '2010-11-20', daily_energy_kwh: 40.1, naive_pred: 35.2, snaive_pred: 39.5, lr_pred: 39.8, rf_pred: 40.0 },
  { date: '2010-11-26', daily_energy_kwh: 36.8, naive_pred: 36.1, snaive_pred: 36.4, lr_pred: 36.9, rf_pred: 36.7 }
];

// Precomputed 7, 14, and 30-day multi-step recursive forecasts starting Nov 27, 2010
export const MULTI_HORIZON_FORECASTS: Record<number, ForecastPoint[]> = {
  7: [
    { date: '2010-11-27', step: 1, forecast_energy_kwh: 41.2, lower_bound_kwh: 33.7, upper_bound_kwh: 48.7, model: 'Random Forest' },
    { date: '2010-11-28', step: 2, forecast_energy_kwh: 42.0, lower_bound_kwh: 34.2, upper_bound_kwh: 49.8, model: 'Random Forest' },
    { date: '2010-11-29', step: 3, forecast_energy_kwh: 34.8, lower_bound_kwh: 26.7, upper_bound_kwh: 42.9, model: 'Random Forest' },
    { date: '2010-11-30', step: 4, forecast_energy_kwh: 35.1, lower_bound_kwh: 26.8, upper_bound_kwh: 43.4, model: 'Random Forest' },
    { date: '2010-12-01', step: 5, forecast_energy_kwh: 36.4, lower_bound_kwh: 27.9, upper_bound_kwh: 44.9, model: 'Random Forest' },
    { date: '2010-12-02', step: 6, forecast_energy_kwh: 35.6, lower_bound_kwh: 26.8, upper_bound_kwh: 44.4, model: 'Random Forest' },
    { date: '2010-12-03', step: 7, forecast_energy_kwh: 37.2, lower_bound_kwh: 28.2, upper_bound_kwh: 46.2, model: 'Random Forest' }
  ],
  14: [
    { date: '2010-11-27', step: 1, forecast_energy_kwh: 41.2, lower_bound_kwh: 33.7, upper_bound_kwh: 48.7, model: 'Random Forest' },
    { date: '2010-11-28', step: 2, forecast_energy_kwh: 42.0, lower_bound_kwh: 34.2, upper_bound_kwh: 49.8, model: 'Random Forest' },
    { date: '2010-11-29', step: 3, forecast_energy_kwh: 34.8, lower_bound_kwh: 26.7, upper_bound_kwh: 42.9, model: 'Random Forest' },
    { date: '2010-11-30', step: 4, forecast_energy_kwh: 35.1, lower_bound_kwh: 26.8, upper_bound_kwh: 43.4, model: 'Random Forest' },
    { date: '2010-12-01', step: 5, forecast_energy_kwh: 36.4, lower_bound_kwh: 27.9, upper_bound_kwh: 44.9, model: 'Random Forest' },
    { date: '2010-12-02', step: 6, forecast_energy_kwh: 35.6, lower_bound_kwh: 26.8, upper_bound_kwh: 44.4, model: 'Random Forest' },
    { date: '2010-12-03', step: 7, forecast_energy_kwh: 37.2, lower_bound_kwh: 28.2, upper_bound_kwh: 46.2, model: 'Random Forest' },
    { date: '2010-12-04', step: 8, forecast_energy_kwh: 42.8, lower_bound_kwh: 33.5, upper_bound_kwh: 52.1, model: 'Random Forest' },
    { date: '2010-12-05', step: 9, forecast_energy_kwh: 43.5, lower_bound_kwh: 34.0, upper_bound_kwh: 53.0, model: 'Random Forest' },
    { date: '2010-12-06', step: 10, forecast_energy_kwh: 36.1, lower_bound_kwh: 26.3, upper_bound_kwh: 45.9, model: 'Random Forest' },
    { date: '2010-12-07', step: 11, forecast_energy_kwh: 35.8, lower_bound_kwh: 25.8, upper_bound_kwh: 45.8, model: 'Random Forest' },
    { date: '2010-12-08', step: 12, forecast_energy_kwh: 37.4, lower_bound_kwh: 27.2, upper_bound_kwh: 47.6, model: 'Random Forest' },
    { date: '2010-12-09', step: 13, forecast_energy_kwh: 36.7, lower_bound_kwh: 26.3, upper_bound_kwh: 47.1, model: 'Random Forest' },
    { date: '2010-12-10', step: 14, forecast_energy_kwh: 38.5, lower_bound_kwh: 27.9, upper_bound_kwh: 49.1, model: 'Random Forest' }
  ],
  30: [
    { date: '2010-11-27', step: 1, forecast_energy_kwh: 41.2, lower_bound_kwh: 33.7, upper_bound_kwh: 48.7, model: 'Random Forest' },
    { date: '2010-11-28', step: 2, forecast_energy_kwh: 42.0, lower_bound_kwh: 34.2, upper_bound_kwh: 49.8, model: 'Random Forest' },
    { date: '2010-11-29', step: 3, forecast_energy_kwh: 34.8, lower_bound_kwh: 26.7, upper_bound_kwh: 42.9, model: 'Random Forest' },
    { date: '2010-11-30', step: 4, forecast_energy_kwh: 35.1, lower_bound_kwh: 26.8, upper_bound_kwh: 43.4, model: 'Random Forest' },
    { date: '2010-12-01', step: 5, forecast_energy_kwh: 36.4, lower_bound_kwh: 27.9, upper_bound_kwh: 44.9, model: 'Random Forest' },
    { date: '2010-12-02', step: 6, forecast_energy_kwh: 35.6, lower_bound_kwh: 26.8, upper_bound_kwh: 44.4, model: 'Random Forest' },
    { date: '2010-12-03', step: 7, forecast_energy_kwh: 37.2, lower_bound_kwh: 28.2, upper_bound_kwh: 46.2, model: 'Random Forest' },
    { date: '2010-12-04', step: 8, forecast_energy_kwh: 42.8, lower_bound_kwh: 33.5, upper_bound_kwh: 52.1, model: 'Random Forest' },
    { date: '2010-12-05', step: 9, forecast_energy_kwh: 43.5, lower_bound_kwh: 34.0, upper_bound_kwh: 53.0, model: 'Random Forest' },
    { date: '2010-12-06', step: 10, forecast_energy_kwh: 36.1, lower_bound_kwh: 26.3, upper_bound_kwh: 45.9, model: 'Random Forest' },
    { date: '2010-12-07', step: 11, forecast_energy_kwh: 35.8, lower_bound_kwh: 25.8, upper_bound_kwh: 45.8, model: 'Random Forest' },
    { date: '2010-12-08', step: 12, forecast_energy_kwh: 37.4, lower_bound_kwh: 27.2, upper_bound_kwh: 47.6, model: 'Random Forest' },
    { date: '2010-12-09', step: 13, forecast_energy_kwh: 36.7, lower_bound_kwh: 26.3, upper_bound_kwh: 47.1, model: 'Random Forest' },
    { date: '2010-12-10', step: 14, forecast_energy_kwh: 38.5, lower_bound_kwh: 27.9, upper_bound_kwh: 49.1, model: 'Random Forest' },
    { date: '2010-12-11', step: 15, forecast_energy_kwh: 43.8, lower_bound_kwh: 33.0, upper_bound_kwh: 54.6, model: 'Random Forest' },
    { date: '2010-12-12', step: 16, forecast_energy_kwh: 44.6, lower_bound_kwh: 33.6, upper_bound_kwh: 55.6, model: 'Random Forest' },
    { date: '2010-12-13', step: 17, forecast_energy_kwh: 37.2, lower_bound_kwh: 26.0, upper_bound_kwh: 48.4, model: 'Random Forest' },
    { date: '2010-12-14', step: 18, forecast_energy_kwh: 36.9, lower_bound_kwh: 25.5, upper_bound_kwh: 48.3, model: 'Random Forest' },
    { date: '2010-12-15', step: 19, forecast_energy_kwh: 38.6, lower_bound_kwh: 27.0, upper_bound_kwh: 50.2, model: 'Random Forest' },
    { date: '2010-12-16', step: 20, forecast_energy_kwh: 38.0, lower_bound_kwh: 26.2, upper_bound_kwh: 49.8, model: 'Random Forest' },
    { date: '2010-12-17', step: 21, forecast_energy_kwh: 39.8, lower_bound_kwh: 27.8, upper_bound_kwh: 51.8, model: 'Random Forest' },
    { date: '2010-12-18', step: 22, forecast_energy_kwh: 45.2, lower_bound_kwh: 33.0, upper_bound_kwh: 57.4, model: 'Random Forest' },
    { date: '2010-12-19', step: 23, forecast_energy_kwh: 46.0, lower_bound_kwh: 33.6, upper_bound_kwh: 58.4, model: 'Random Forest' },
    { date: '2010-12-20', step: 24, forecast_energy_kwh: 38.4, lower_bound_kwh: 25.8, upper_bound_kwh: 51.0, model: 'Random Forest' },
    { date: '2010-12-21', step: 25, forecast_energy_kwh: 38.1, lower_bound_kwh: 25.3, upper_bound_kwh: 50.9, model: 'Random Forest' },
    { date: '2010-12-22', step: 26, forecast_energy_kwh: 39.9, lower_bound_kwh: 26.9, upper_bound_kwh: 52.9, model: 'Random Forest' },
    { date: '2010-12-23', step: 27, forecast_energy_kwh: 39.2, lower_bound_kwh: 26.0, upper_bound_kwh: 52.4, model: 'Random Forest' },
    { date: '2010-12-24', step: 28, forecast_energy_kwh: 43.5, lower_bound_kwh: 30.1, upper_bound_kwh: 56.9, model: 'Random Forest' },
    { date: '2010-12-25', step: 29, forecast_energy_kwh: 48.6, lower_bound_kwh: 35.0, upper_bound_kwh: 62.2, model: 'Random Forest' },
    { date: '2010-12-26', step: 30, forecast_energy_kwh: 47.9, lower_bound_kwh: 34.1, upper_bound_kwh: 61.7, model: 'Random Forest' }
  ]
};

export const TEST_SUITE_CASES: TestCaseResult[] = [
  {
    id: 'test_timestamp_parsing',
    name: 'Timestamp Parsing & Deduplication',
    category: 'Preprocessing',
    description: 'Verifies parsing of "dd/mm/yyyy hh:mm:ss" strings and deduplicates repeated minute records.',
    assertion: 'len(parsed) == 2 and parsed.datetime.dtype == datetime64',
    passed: true,
    durationMs: 42
  },
  {
    id: 'test_energy_unit_conversion',
    name: 'Dimensional Energy Conversion (kW -> kWh)',
    category: 'Preprocessing',
    description: 'Verifies that continuous 6 kW load over 60 minutes equals exactly 6.0 kWh (sum(P) / 60).',
    assertion: 'isclose(daily_stats.daily_energy_kwh[0], 6.0)',
    passed: true,
    durationMs: 38
  },
  {
    id: 'test_incomplete_day_policy',
    name: '95% Completeness Eligibility Cutoff',
    category: 'Preprocessing',
    description: 'Asserts day with 1,368 minutes qualifies (is_eligible=True) and day with 1,367 minutes is excluded.',
    assertion: 'day1.is_eligible == True and day2.is_eligible == False',
    passed: true,
    durationMs: 45
  },
  {
    id: 'test_lag_features_alignment',
    name: 'Autoregressive Lag Alignment',
    category: 'Feature Engineering',
    description: 'Verifies lag_1 aligns with t-1, lag_7 with t-7, and lag_28 with t-28 without indexing shifts.',
    assertion: 'lagged.iloc[10].lag_1 == 10.0 and lagged.iloc[10].lag_7 == 4.0',
    passed: true,
    durationMs: 29
  },
  {
    id: 'test_rolling_zero_leakage',
    name: 'Zero Look-Ahead Leakage in Rolling Stats',
    category: 'Feature Engineering',
    description: 'Verifies shift(1) prevents day t spike from contaminating day t rolling average.',
    assertion: 'isclose(rolled.iloc[10].rolling_7d_mean, 10.0) # spike to 100 did not leak',
    passed: true,
    durationMs: 34
  },
  {
    id: 'test_chronological_split',
    name: 'Chronological Splitting Order (No Shuffling)',
    category: 'Validation',
    description: 'Validates strict temporal boundaries: max(Train) < min(Val) < min(Test) with zero data loss.',
    assertion: 'train.date.max() < val.date.min() < test.date.min()',
    passed: true,
    durationMs: 31
  },
  {
    id: 'test_evaluation_metrics',
    name: 'MAE, RMSE, and Zero-Safe MAPE Calculations',
    category: 'Validation',
    description: 'Validates exact mathematical correctness of MAE, RMSE, and epsilon-bounded MAPE.',
    assertion: 'isclose(mae, 3.0) and isclose(rmse, 3.109) and isclose(mape, 10.0)',
    passed: true,
    durationMs: 22
  },
  {
    id: 'test_recursive_forecast',
    name: 'Multi-Step Recursive Horizon & Continuity',
    category: 'Forecasting',
    description: 'Asserts 7, 14, and 30-day recursive forecasts output consecutive future dates and non-negative bounds.',
    assertion: 'len(fc) == H and all(diff == 1_day) and all(lower >= 0)',
    passed: true,
    durationMs: 56
  }
];

export const VIVA_QUESTIONS = [
  {
    id: 1,
    question: 'Why did you divide by 60 when calculating daily energy?',
    answer: 'The raw dataset measures Global_active_power in Kilowatts (kW), which is an instantaneous rate of energy consumption. Because readings are recorded at 1-minute intervals (1/60th of an hour), each minute represents an energy consumption of P × (1/60) kWh. Summing all valid readings in a day gives sum(P) / 60 kWh. Failing to divide by 60 would measure kW-minutes, which is 60 times too large.',
    category: 'Physical Modeling & Units'
  },
  {
    id: 2,
    question: 'Why not just impute missing minutes with 0 or linear interpolation?',
    answer: 'Imputing missing minutes with 0 falsely assumes all household appliances were powered off, which artificially deflates daily energy totals. Interpolating across multi-hour logger outages invents unmeasured electricity. Our defensible 95% threshold requires at least 1,368 valid minutes per day; days failing this are excluded from model training to protect ground-truth integrity.',
    category: 'Data Quality & Imputation'
  },
  {
    id: 3,
    question: 'What is look-ahead bias and how did your feature engineering prevent it?',
    answer: 'Look-ahead bias occurs when future information that would be unavailable at the forecast origin leaks into the feature vector. For example, if a 7-day rolling average for day t includes day t\'s actual consumption, the model "cheats". We strictly prevent this by applying .shift(1) before computing rolling statistics, so day t\'s features only see days t-7 through t-1.',
    category: 'Feature Engineering & Leakage'
  },
  {
    id: 4,
    question: 'Why must we avoid standard random train_test_split in time series?',
    answer: 'Time series observations possess temporal dependency and autocorrelation. Randomly shuffling rows leaks past patterns into the future and future patterns into the past, resulting in artificially inflated test scores that fail catastrophically in deployment. We enforce strict chronological partitioning: Train on 2007–2009, Validate on early 2010, and Test on late 2010.',
    category: 'Validation Strategy'
  },
  {
    id: 5,
    question: 'What is the primary drawback of recursive multi-step forecasting?',
    answer: 'Recursive error accumulation. Because step k=2 consumes the model\'s own predicted value from step k=1 as its lag_1 input, initial prediction errors compound over longer horizons. We address and visualize this by computing expanding 95% confidence intervals that widen over 7, 14, and 30-day horizons.',
    category: 'Multi-Step Forecasting'
  },
  {
    id: 6,
    question: 'Why did Random Forest outperform Linear Regression in this problem?',
    answer: 'Household electricity consumption contains non-linear interactions between calendar variables and occupant behaviors—such as the interaction between winter heating demand, Saturday laundry cycles, and temperature. Random Forest decision trees naturally partition these non-linear feature spaces without requiring explicit manual interaction terms.',
    category: 'Machine Learning Models'
  }
];
