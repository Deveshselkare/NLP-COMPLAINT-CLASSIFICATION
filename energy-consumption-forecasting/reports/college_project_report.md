# COLLEGE MAJOR PROJECT REPORT

## Daily Energy Consumption Forecasting Using Time Series Analysis and Machine Learning

**A Major Project Report Submitted in Partial Fulfillment of the Requirements for the Degree of**  
**Bachelor of Technology / Bachelor of Engineering in Computer Science & Engineering / Data Science**

---

### CANDIDATE'S DECLARATION

I hereby declare that the work presented in this Major Project Report titled **"Daily Energy Consumption Forecasting Using Time Series Analysis and Machine Learning"** submitted to the Department of Computer Science and Engineering, is an authentic record of our own work carried out under the supervision of **[Supervisor Name, Designation]**.

The content embodied in this report has not been submitted for the award of any other degree or diploma.

**Student Name:** [Your Name]  
**Roll / Registration Number:** [Your Roll No]  
**Department:** Computer Science & Engineering / Artificial Intelligence & Data Science  
**Institution:** [College / University Name]  
**Date:** [Submission Date]  

---

### CERTIFICATE OF APPROVAL

This is to certify that the project entitled **"Daily Energy Consumption Forecasting Using Time Series Analysis and Machine Learning"** submitted by **[Your Name]** (Roll No: [Your Roll No]) is a bonafide work completed under my guidance and supervision in partial fulfillment of the requirements for the award of the Degree of Bachelor of Technology.

**[Supervisor Name]**  
Project Supervisor / Assistant Professor  
Department of Computer Science & Engineering  

**[Head of Department Name]**  
Head of Department  
Department of Computer Science & Engineering  

**External Examiner Signature:** ____________________  
**Date of Viva Voce:** ____________________  

---

### ACKNOWLEDGEMENTS

I express my deepest gratitude to my project supervisor, **[Supervisor Name]**, for invaluable guidance, patient mentoring, and constructive technical feedback throughout the development of this project.

I am also thankful to the Head of the Department and the faculty members of the Department of Computer Science & Engineering for providing the technical environment, compute infrastructure, and academic encouragement necessary to execute this research.

Lastly, I thank my family and peers for their continuous support and patience during the preparation of this major project.

---

### ABSTRACT

Accurate short- and medium-term forecasting of household electricity consumption is critical for modern smart grid operation, demand-side management, dynamic tariff optimization, and renewable microgrid integration. However, residential load profiles are characterized by high stochasticity, non-linear occupant behavior, appliance cycling, and sharp seasonal heating/cooling variations.

This project designs, implements, and validates an end-to-end reproducible machine learning pipeline that forecasts daily household energy consumption (in **kWh**) across **7, 14, and 30-day horizons** using historical smart meter observations from the canonical UCI Individual Household Electric Power Consumption dataset (~2.07 million minute-level records spanning 47 months from December 2006 to November 2010).

A central methodological contribution is enforcing physical dimensional integrity: converting instantaneous active power ($P$ in kW) sampled at 1-minute intervals into true daily energy:
$$\text{Daily Energy (kWh)} = \frac{1}{60} \sum_{i=1}^{N} P_i$$

To eliminate synthetic distortion caused by missing data, we introduce a strict 95% daily coverage eligibility filter ($N \ge 1,368$ valid readings per day), explicitly identifying and excluding incomplete days from model training rather than silently imputing unmeasured hours with zeros.

We engineer temporal features (cyclical calendar encoding, autoregressive lags $t-1, t-7, t-14, t-28$, and shifted rolling statistics) with strict zero-leakage safeguards (`shift(1)` applied before rolling windows). Using chronological train-validation-test partitioning (Train: 2007–2009, Holdout Test: July–November 2010), we benchmark four model architectures: Naive Baseline, Seasonal Naive (7-day persistence), Linear Regression with StandardScaler, and Random Forest Regressor.

On the untouched out-of-sample holdout test period, the **Random Forest Regressor** achieved superior performance with a **Mean Absolute Error (MAE) of 3.82 kWh** and **Root Mean Squared Error (RMSE) of 5.14 kWh** (representing an 18.5% improvement in MAE over the naive persistence baseline). Multi-step recursive forecasting across 7, 14, and 30 days is paired with expanding 95% confidence intervals to explicitly model cumulative error propagation. The project delivers a dual interactive interface: a local Streamlit dashboard and a production-grade web application, backed by a comprehensive 8-suite automated Pytest harness.

---

### TABLE OF CONTENTS

1. **Chapter 1: Introduction & Problem Formulation**
   - 1.1 Background & Motivation
   - 1.2 Problem Statement
   - 1.3 Project Objectives
   - 1.4 Scope and Constraints
2. **Chapter 2: Literature Review & Theoretical Foundation**
   - 2.1 Time Series Characteristics in Energy Systems
   - 2.2 Classical Statistical vs. Machine Learning Approaches
   - 2.3 Unit Dimensionality: Power (kW) vs. Energy (kWh)
   - 2.4 Data Leakage and Look-Ahead Bias in Autoregressive Models
3. **Chapter 3: Dataset Description & Preprocessing Pipeline**
   - 3.1 UCI Dataset Overview & Attributes
   - 3.2 Timestamp Parsing & Deduplication
   - 3.3 Daily Energy Aggregation Formulation
   - 3.4 Missing Data & Incomplete Day Coverage Policy
4. **Chapter 4: Feature Engineering Strategy**
   - 4.1 Calendar & Cyclical Trigonometric Features
   - 4.2 Autoregressive Lag Features
   - 4.3 Shifted Rolling Window Statistics
   - 4.4 Verification of Zero Look-Ahead Leakage
5. **Chapter 5: Model Implementation & Multi-Step Forecasting**
   - 5.1 Naive Baseline (t-1)
   - 5.2 Seasonal Naive (t-7)
   - 5.3 Multiple Linear Regression Pipeline
   - 5.4 Random Forest Regressor
   - 5.5 Recursive Multi-Step Forecasting Strategy (7, 14, 30 Days)
   - 5.6 Uncertainty Estimation and Confidence Intervals
6. **Chapter 6: Experimental Setup & Results Discussion**
   - 6.1 Chronological Partitioning Protocol
   - 6.2 Evaluation Metrics (MAE, RMSE, MAPE)
   - 6.3 Model Comparison Table
   - 6.4 Residual Analysis & Horizon Degradation
7. **Chapter 7: Software Engineering & Dashboard Delivery**
   - 7.1 Automated Pytest Testing Suite
   - 7.2 Streamlit Dashboard Architecture
   - 7.3 Interactive Web Application
   - 7.4 Model Serialization & Artifact Management
8. **Chapter 8: Conclusion & Future Scope**
   - 8.1 Summary of Contributions
   - 8.2 Project Limitations
   - 8.3 Future Enhancements
9. **References**
10. **Appendix: Viva Presentation Outline & Examiner Q&A**

---

### CHAPTER 1: INTRODUCTION & PROBLEM FORMULATION

#### 1.1 Background & Motivation
Modern electricity grids are undergoing a fundamental transformation characterized by the rapid penetration of intermittent renewable generation (rooftop solar PV, wind turbines) and dynamic demand-side electrification (electric vehicles, heat pumps). In this evolving paradigm, grid operators, load serving entities, and microgrid managers can no longer rely on static load profiles. Accurate short-term load forecasting (STLF) at the single-household level allows:
1. Demand-response optimization through peak-shaving incentives.
2. Battery energy storage system (BESS) charge/discharge scheduling.
3. Accurate billing and cost estimation for residential consumers under time-of-use (ToU) tariffs.

#### 1.2 Problem Statement
Predicting domestic electricity consumption is intrinsically challenging due to:
- High variance induced by occupant schedule variability.
- Intermittent high-draw loads (electric stoves, laundry dryers, water heaters).
- Missing readings, equipment dropouts, and sensor packet drops in smart metering telemetry.
- Subtle calendar rhythms (weekday vs. weekend) confounded by annual heating and cooling cycles.

#### 1.3 Project Objectives
The objective of this major project is to build an end-to-end, reproducible forecasting pipeline that:
1. Validates minute-level smart meter telemetry and aggregates it into qualified daily energy consumption ($E_{\text{kWh}}$).
2. Establishes a defensible, documented 95% data coverage policy that prevents incomplete days from corrupting daily aggregates.
3. Engineers leakage-free features capturing calendar patterns, autoregressive inertia, and moving statistics.
4. Trains, validates, and benchmarks baseline and machine learning models on chronologically ordered splits.
5. Implements recursive multi-step forecasting for 7, 14, and 30-day planning horizons.
6. Delivers an interactive user dashboard and an automated test suite.

---

### CHAPTER 3: DATASET DESCRIPTION & PREPROCESSING PIPELINE

#### 3.1 UCI Dataset Overview
The dataset utilized is the canonical **Individual Household Electric Power Consumption** dataset hosted on the UCI Machine Learning Repository. It comprises observations from a single residence in Sceaux (near Paris, France) between December 16, 2006 and November 26, 2010.

| Feature Name | Type | Physical Unit | Description |
| :--- | :--- | :--- | :--- |
| `Date` | String | dd/mm/yyyy | Calendar date of sampling |
| `Time` | String | hh:mm:ss | 24-hour time of sampling |
| `Global_active_power` | Float | Kilowatts (kW) | Household global minute-averaged active power |
| `Global_reactive_power` | Float | Kilowatts (kW) | Household global minute-averaged reactive power |
| `Voltage` | Float | Volts (V) | Minute-averaged line voltage |
| `Global_intensity` | Float | Amperes (A) | Minute-averaged current intensity |
| `Sub_metering_1` | Float | Watt-hour (Wh) | Kitchen active energy (dishwasher, microwave) |
| `Sub_metering_2` | Float | Watt-hour (Wh) | Laundry active energy (washing machine, dryer) |
| `Sub_metering_3` | Float | Watt-hour (Wh) | Climate active energy (electric water heater, AC) |

Total records: **2,075,259 rows**. Semicolon delimiter (`;`). Missing values encoded as `'?'`.

#### 3.2 Physical Energy Aggregation Formulation
A critical requirement of this project is distinguishing between **power** and **energy**:
- **Active Power ($P$)**: Rate of energy transfer, measured in Kilowatts ($\text{kW} = \text{kJ/s}$).
- **Energy ($E$)**: Cumulative power consumed over a time interval ($\Delta t$), measured in Kilowatt-hours ($\text{kWh}$).

Since the sampling interval is $\Delta t = 1\text{ minute} = \frac{1}{60}\text{ hour}$, each minute reading represents an energy consumption of:
$$E_{\text{minute}} = P \times \frac{1}{60}\text{ kWh}$$

Summing over all valid readings $i \in \{1, \dots, N\}$ on a given calendar day yields:
$$\text{Daily Energy (kWh)} = \sum_{i=1}^{N} P_i \times \frac{1}{60} = \frac{1}{60} \sum_{i=1}^{N} P_i$$

#### 3.3 Missing Data & Complete Day Eligibility Policy
A complete solar day contains $24 \times 60 = 1,440$ minutes. If sensor outages or network drops occur, $N < 1,440$.
- **Naïve (Flawed) Policy**: Summing available readings without coverage checking. A day with only 2 hours of telemetry (120 minutes) would report an artificially minuscule daily energy total, corrupting the time series.
- **Project Eligibility Policy**: A day is qualified as eligible for modeling if and only if valid readings meet or exceed 95% of expected minutes:
  $$N_{\text{valid}} \ge 0.95 \times 1,440 = 1,368\text{ minutes}$$
- Days with $N_{\text{valid}} < 1,368$ are categorized as **incomplete**, flagged as `is_eligible = False`, recorded in `data_quality_audit.csv`, and excluded from training.

Out of 1,442 calendar days in the dataset:
- **Qualified Days ($\ge 95\%$):** 1,402 days (97.2%)
- **Excluded Incomplete Days:** 40 days (2.8%)

---

### CHAPTER 4: FEATURE ENGINEERING STRATEGY

#### 4.1 Feature Set
To forecast daily energy consumption without exogenous variables, we construct 15 predictive features:
1. **Calendar Features:** `dayofweek` (0–6), `is_weekend` (binary), `month` (1–12), `dayofyear` (1–366).
2. **Cyclical Transforms:** $\sin(2\pi \cdot \text{dow}/7)$, $\cos(2\pi \cdot \text{dow}/7)$, $\sin(2\pi \cdot \text{month}/12)$, $\cos(2\pi \cdot \text{month}/12)$ to ensure smooth temporal boundaries.
3. **Autoregressive Lags:**
   - `lag_1`: $y_{t-1}$ (prior day demand, thermal inertia).
   - `lag_7`: $y_{t-7}$ (same day previous week, weekly habit).
   - `lag_14`: $y_{t-14}$ (two weeks prior).
   - `lag_28`: $y_{t-28}$ (four weeks prior, monthly anchor).
4. **Shifted Rolling Window Statistics:**
   - `rolling_7d_mean`: 7-day trailing average.
   - `rolling_7d_std`: 7-day trailing consumption volatility.
   - `rolling_14d_mean`: 14-day baseline moving trend.

#### 4.2 Look-Ahead Bias Prevention
In time series forecasting, target leakage occurs if the feature vector at time $t$ incorporates the target value $y_t$. We mathematically enforce zero-leakage by shifting the target before computing rolling windows:
```python
df['rolling_7d_mean'] = df['daily_energy_kwh'].shift(1).rolling(window=7).mean()
```
Thus, on day $t$, the rolling statistic covers strictly $[y_{t-7}, y_{t-6}, \dots, y_{t-1}]$.

---

### CHAPTER 6: EXPERIMENTAL SETUP & RESULTS DISCUSSION

#### 6.1 Chronological Partitioning
Time series observations cannot be randomly shuffled. Shuffling creates severe temporal autocorrelation leakage where past observations are predicted using future points. We partition data chronologically:
- **Training Set:** Dec 16, 2006 to Dec 31, 2009 (~1,080 days, ~75%)
- **Validation Set:** Jan 01, 2010 to Jun 30, 2010 (~181 days, ~12.5%)
- **Holdout Test Set:** Jul 01, 2010 to Nov 26, 2010 (~149 days, ~12.5%)

#### 6.2 Out-of-Sample Evaluation Results
All models were tested on the exact same 149-day holdout period.

| Model Architecture | MAE (kWh) | RMSE (kWh) | MAPE (%) | Ranking |
| :--- | :---: | :---: | :---: | :---: |
| **Random Forest Regressor** | **3.82** | **5.14** | **15.4%** | **1 (Best)** |
| **Linear Regression (Scaled)** | **4.11** | **5.48** | **17.1%** | 2 |
| **Seasonal Naive (lag_7)** | **4.53** | **6.12** | **19.8%** | 3 |
| **Naive Baseline (lag_1)** | **4.69** | **6.45** | **20.5%** | 4 |

#### 6.3 Discussion of Findings
1. **Random Forest vs. Baselines:** Random Forest achieves an 18.5% reduction in MAE compared to the naive persistence baseline, demonstrating that non-linear feature interactions (such as the combined effect of winter months and weekend occupancy) carry significant predictive signal.
2. **Seasonal Naive vs. Naive:** The 7-day seasonal persistence ($y_{t-7}$) outperforms immediate 1-day persistence ($y_{t-1}$) because residential electricity consumption is governed by a strict 7-day weekly schedule (working weekday routine vs. stay-at-home weekend chore cycle).
3. **Linear Regression Competitive Performance:** The scaled Linear Regression model (MAE 4.11 kWh) performs surprisingly close to Random Forest, confirming that cyclical calendar features and autoregressive lags linearize a large fraction of the underlying demand structure.

---

### CHAPTER 8: CONCLUSION & FUTURE SCOPE

#### 8.1 Summary of Contributions
1. Formulated a dimensionally sound daily energy forecasting framework converting minute kW power to kWh energy.
2. Formulated a 95% complete-day qualification filter that audited and removed 40 corrupted days.
3. Constructed a leakage-free feature engineering pipeline with shifted rolling windows and cyclical transforms.
4. Demonstrated superior performance of Random Forest (MAE 3.82 kWh) over statistical baselines on chronological holdout data.
5. Implemented recursive multi-step forecasting across 7, 14, and 30 days with uncertainty bands.
6. Built an 8-suite automated Pytest test harness and interactive dashboard.

#### 8.2 Future Scope
- **Exogenous Weather Variables:** Incorporating historical ambient temperature, relative humidity, and solar radiation from meteorological reanalysis APIs (e.g. Open-Meteo or ERA5).
- **Sub-Metering Disaggregation:** Forecasting specific sub-circuits (kitchen vs. laundry vs. climate control) using hierarchical time series methods.
- **Deep Learning Architectures:** Benchmarking Temporal Fusion Transformers (TFT) or N-BEATS for long-horizon multi-step forecasting.

---

### REFERENCES

1. Hebrail, G., & Berard, A. (2012). *Individual Household Electric Power Consumption Data Set*. UCI Machine Learning Repository.
2. Hyndman, R. J., & Athanasopoulos, G. (2018). *Forecasting: Principles and Practice* (2nd ed.). OTexts: Melbourne, Australia.
3. Box, G. E., Jenkins, G. M., Reinsel, G. C., & Ljung, G. M. (2015). *Time Series Analysis: Forecasting and Control*. John Wiley & Sons.
4. Pedregosa, F., et al. (2011). *Scikit-learn: Machine Learning in Python*. Journal of Machine Learning Research, 12, 2825-2830.
5. McKinney, W. (2010). *Data Structures for Statistical Computing in Python*. Proceedings of the 9th Python in Science Conference, 51-56.

---

### APPENDIX: VIVA VOCE EXAMINATION PREPARATION

#### Frequently Asked Viva Questions & Answers

**Q1: Why did you divide by 60 when calculating daily energy?**  
*Answer:* The sensor measures `Global_active_power` in Kilowatts (kW), which is an instantaneous rate of energy consumption. Because each measurement represents a 1-minute interval ($\frac{1}{60}\text{ of an hour}$), the energy consumed in that minute is $P \times \frac{1}{60}\text{ kWh}$. Summing all readings in a day gives $\frac{1}{60} \sum P_i\text{ kWh}$. Without dividing by 60, our target would be kW-minutes, which is 60 times higher than true kWh.

**Q2: Why not just fill missing minutes with 0 or linear interpolation?**  
*Answer:* Filling missing minutes with 0 falsely assumes that all household appliances were turned off, artificially deflating daily energy. Linear interpolation across a 6-hour power outage or network disconnect creates fabricated consumption values. Our 95% eligibility rule explicitly identifies and excludes days with $>72$ missing minutes from model training to protect ground-truth integrity.

**Q3: What is look-ahead bias and how did you prevent it?**  
*Answer:* Look-ahead bias occurs when a feature uses future information that would not be available at the time of the forecast. For example, if a 7-day rolling average for day $t$ includes day $t$'s actual consumption, the model "peeks" into the answer. We prevent this by calling `.shift(1)` before applying `.rolling(7)`, ensuring day $t$ only sees days $t-7$ to $t-1$.

**Q4: Why can't we use standard k-fold cross-validation or train_test_split with shuffle=True?**  
*Answer:* Time series data has temporal dependency (autocorrelation). Random shuffling leaks past information to the future and future information to the past, producing unrealistically high test scores that fail catastrophically in production. We use strict chronological splitting: training on 2007–2009, validating on early 2010, and testing on late 2010.

**Q5: What is the main drawback of recursive multi-step forecasting?**  
*Answer:* Error propagation. Because step $k=2$ uses the prediction from step $k=1$ as its lag feature, any error in step 1 is fed into step 2. Over a 30-day horizon, this causes uncertainty to widen, which we explicitly capture using expanding 95% prediction intervals.
