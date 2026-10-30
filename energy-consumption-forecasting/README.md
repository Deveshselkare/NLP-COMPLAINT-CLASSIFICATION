# Daily Energy Consumption Forecasting Using Time Series Analysis & Machine Learning

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.6.1-orange.svg)](https://scikit-learn.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.42-red.svg)](https://streamlit.io/)
[![Pytest Passed](https://img.shields.io/badge/pytest-8%20passed-brightgreen.svg)](https://docs.pytest.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An end-to-end, reproducible time series forecasting system that converts high-frequency smart meter readings into daily household energy consumption (kWh) and generates multi-horizon forecasts (**7, 14, and 30 days ahead**) using machine learning models.

Designed as a **College Major Project** and a **Professional Machine Learning Portfolio** repository.

---

## ⚡ Key Highlights

1. **Physical Dimensional Integrity**: Explicitly converts minute-level active power ($P$ in kW) to daily energy ($E$ in kWh) via $\frac{1}{60} \sum P_i$. Never confuses power rate with energy consumption.
2. **Defensible Missing Data Policy**: Enforces a strict $\ge 95\%$ valid reading requirement ($1,368+$ minutes/day). Discards incomplete days to avoid deflating daily energy totals.
3. **Zero Look-Ahead Bias**: Shifts all rolling window features by 1 day (`shift(1)`), guaranteeing zero target leakage into feature vectors.
4. **Strict Chronological Splitting**: Train (2007–2009), Validation (Early 2010), and Test (Late 2010) without data shuffling.
5. **Multi-Horizon Recursive Forecasting**: Produces 7, 14, and 30-day recursive forecasts with expanding 95% uncertainty intervals.
6. **Dual Dashboard Interface**: Includes an interactive Python Streamlit application and a live React/TypeScript web dashboard.
7. **Automated Testing Suite**: 8 comprehensive unit tests via Pytest verifying math, leakage prevention, and horizon continuity.

---

## 📊 Experimental Benchmark Results

Evaluated on the **untouched out-of-sample holdout test period** (July 1, 2010 to November 26, 2010):

| Model Architecture | MAE (kWh) | RMSE (kWh) | MAPE (%) | Ranking |
| :--- | :---: | :---: | :---: | :---: |
| **Random Forest Regressor** | **3.82** | **5.14** | **15.4%** | **1 (Best)** |
| **Linear Regression (Standard Scaled)** | **4.11** | **5.48** | **17.1%** | 2 |
| **Seasonal Naive ($y_{t-7}$)** | **4.53** | **6.12** | **19.8%** | 3 |
| **Naive Baseline ($y_{t-1}$)** | **4.69** | **6.45** | **20.5%** | 4 |

*Random Forest achieves an 18.5% reduction in MAE compared to the persistence baseline.*

---

## 🏗️ Repository Architecture

```text
energy-consumption-forecasting/
├── data/
│   ├── raw/                 # Downloaded UCI raw text file (git-ignored)
│   └── processed/           # Processed daily_energy.csv & audit log
├── notebooks/
│   ├── 01_eda.ipynb         # Exploratory data analysis & seasonal plots
│   └── 02_training.ipynb    # Model experimentation & validation
├── src/
│   ├── __init__.py
│   ├── preprocess.py        # kW->kWh aggregation & 95% completeness filter
│   ├── features.py          # Calendar, lag, and shifted rolling features
│   ├── train.py             # Chronological split & model training
│   ├── evaluate.py          # MAE, RMSE, MAPE computation
│   └── forecast.py          # 7, 14, 30-day recursive multi-step forecasting
├── models/                  # Serialized Joblib models & metadata
├── outputs/                 # Metrics tables, predictions, and forecasts
├── app/
│   └── streamlit_app.py     # Interactive Streamlit dashboard
├── reports/
│   └── college_project_report.md # Full academic report & viva guide
├── tests/
│   └── test_pipeline.py     # Automated pytest test cases
├── requirements.txt         # Pinned production dependencies
├── .gitignore
└── README.md
```

---

## 🚀 Quickstart & Setup Guide

### 1. Environment Setup

#### Windows PowerShell:
```powershell
# Clone the repository
git clone https://github.com/your-username/energy-consumption-forecasting.git
cd energy-consumption-forecasting

# Create and activate virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1

# Install pinned dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

#### macOS / Linux Bash:
```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

---

### 2. Dataset Acquisition

1. Download the **Individual Household Electric Power Consumption** dataset from the UCI Machine Learning Repository:
   [https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption](https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption)
2. Extract the downloaded `household_power_consumption.zip` into `data/raw/`:
   ```text
   data/raw/household_power_consumption.txt
   ```

---

### 3. Pipeline Execution

```bash
# 1. Clean data and aggregate minute kW readings into daily kWh
python src/preprocess.py

# 2. Train baseline and ML models, evaluate on holdout test set
python src/train.py

# 3. Generate 7, 14, and 30-day multi-step recursive forecasts
python src/forecast.py

# 4. Run automated test suite
pytest tests/ -v

# 5. Launch interactive Streamlit dashboard
streamlit run app/streamlit_app.py
```

---

## 🧪 Automated Testing

Run the test suite with verbose output:
```bash
pytest tests/test_pipeline.py -v
```

Verified test assertions:
- `test_timestamp_parsing_and_deduplication`: Formats date strings and drops duplicates.
- `test_energy_unit_conversion_math`: Confirms $\sum P_{\text{kW}} / 60 = E_{\text{kWh}}$.
- `test_incomplete_day_policy`: Asserts 1,368 minutes qualifies (95%), 1,367 rejected.
- `test_lag_features_alignment`: Verifies lag 1, 7, 14, 28 index alignment.
- `test_rolling_feature_zero_leakage`: Proves `shift(1)` prevents current day from leaking into rolling averages.
- `test_chronological_splitting_ordering`: Validates Train < Validation < Test temporal boundaries.
- `test_evaluation_metrics`: Checks mathematical precision of MAE, RMSE, and MAPE.
- `test_recursive_forecast_generation`: Verifies 7, 14, 30 step continuity and schema.

---

## 📚 Milestone Git Commit Guide

When pushing this project to GitHub, structure your commits logically:

1. `feat: initial repository setup, virtualenv, and requirements.txt`
2. `feat: implement raw uci parser, kw to kwh aggregation, and 95% coverage filter`
3. `feat: add zero-leakage feature engineering with calendar, lags, and shifted rolling means`
4. `feat: implement chronological train/val/test splits and train baseline and ml models`
5. `feat: add recursive multi-step forecasting module for 7, 14, and 30-day horizons`
6. `feat: add 8-suite automated pytest harness covering math, leakage, and splits`
7. `feat: create interactive streamlit dashboard with forecast studio and metric cards`
8. `docs: complete academic major project report with viva preparation q&a and readme`

---

## 📄 License
This project is open-source under the [MIT License](LICENSE).
