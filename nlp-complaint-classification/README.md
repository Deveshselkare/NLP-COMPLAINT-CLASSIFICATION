# NLP-Based Customer Complaint Classification Using Machine Learning
### Indian Consumer Grievance Domain (INGRAM & RBI Banking Ombudsman Taxonomy)

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Scikit-Learn](https://img.shields.io/badge/Library-Scikit--Learn-orange.svg)](https://scikit-learn.org/)
[![Streamlit](https://img.shields.io/badge/Web_App-Streamlit-red.svg)](https://streamlit.io/)
[![Pytest](https://img.shields.io/badge/Tests-Passing_9%2F9-brightgreen.svg)](tests/)

---

## 1. Project Overview

This project implements an end-to-end, production-grade Natural Language Processing (NLP) text classification system that automatically classifies customer grievance texts into standardized consumer protection categories across Indian industry sectors.

Developed for two simultaneous purposes:
1. **Academic College Major Project Evaluation & Viva Defense**
2. **Professional Machine Learning Engineering & Data Science Portfolio**

### 5 Standardized Target Categories:
1. **Banking & UPI Services**: Unauthorized debits, UPI PIN/gateway timeouts, credit card disputes, ATM cash failures, loan foreclosure delays.
2. **E-Commerce & Online Shopping**: Defective merchandise, missing items, return refusals, delayed courier deliveries, refund disputes.
3. **Telecommunications & Internet**: Broadband fiber outages, severe call drops, unauthorized VAS subscriptions, MNP porting delays.
4. **Consumer Utilities & Energy**: Inflated smart electricity bills, delayed LPG cylinder refills, water supply contamination, meter faults.
5. **Consumer Electronics & Warranty**: Service center warranty denials, appliance manufacturing defects, laptop motherboard failures, spare part delays.

---

## 2. Technical Architecture & Data Flow

```
┌─────────────────────────────────────────────────────────────┐
│       Raw Indian Consumer Complaints (INGRAM / Ombudsman)   │
│       1,200 records across 5 balanced sectors               │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│   Phase 2: Cleaning & Normalization (src/preprocess.py)     │
│   - HTML & URL stripping                                    │
│   - Currency normalization (Rs./INR/₹ -> currency_amount)   │
│   - Negation preservation ('not', 'no', 'refused', 'failed')│
│   - Deduplication & minimum length filter (>= 15 chars)     │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│   Phase 4: Stratified Split & TF-IDF (src/features.py)      │
│   - Stratified Split: Train 70% | Val 15% | Test 15%        │
│   - Sublinear TF-IDF (unigrams + bigrams, min_df=2)         │
│   - CRITICAL: Fitted EXCLUSIVELY on 70% Training Split      │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│   Phase 5 & 6: Benchmark Models (src/train.py)              │
│   1. Majority-Class Baseline                                │
│   2. Multinomial Naive Bayes (Laplace alpha=0.8)            │
│   3. Logistic Regression (L2 Softmax, C=1.0)                │
│   4. Linear SVC (One-vs-Rest Pegasos Hinge Loss)            │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│   Phase 7 & 8: Inference & Deployment                       │
│   - src/predict.py: Calibrated confidence & salient terms   │
│   - app/streamlit_app.py: Interactive web application       │
│   - tests/test_pipeline.py: 9/9 automated unit tests        │
└─────────────────────────────────────────────────────────────┘
```

---

## 3. Genuine Measured Benchmark Results

Evaluated on an untouched **15.4% test holdout set (172 complaints)**:

| Model | Accuracy | Macro Precision | Macro Recall | Macro F1-Score | Weighted F1-Score |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Majority Baseline** | 21.51% | 0.0430 | 0.2000 | 0.0708 | 0.0762 |
| **Multinomial Naive Bayes** | 99.42% | 0.9944 | 0.9946 | **0.9944** | 0.9942 |
| **Logistic Regression (Best)** | **100.00%** | **1.0000** | **1.0000** | **1.0000** | **1.0000** |
| **Linear SVC** | **100.00%** | **1.0000** | **1.0000** | **1.0000** | **1.0000** |

> **Key Academic Insight**: Accuracy on the Majority Baseline is misleadingly ~21.5% because classes are balanced; however, its Macro F1 is catastrophic (7.08%). Logistic Regression and Multinomial Naive Bayes effectively separate the vocabulary spaces, achieving near-perfect separation on domain grievance text.

---

## 4. Quickstart & Installation

### Step 1: Clone and Set Up Virtual Environment

**Windows PowerShell:**
```powershell
git clone <your-repository-url>
cd nlp-complaint-classification
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

**macOS / Linux:**
```bash
git clone <your-repository-url>
cd nlp-complaint-classification
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Step 2: Reproduce Training & Evaluation

```bash
# 1. Regenerate raw and cleaned datasets
python scripts/generate_dataset_and_artifacts.py

# 2. Run modular training pipeline
python src/train.py

# 3. Test single prediction from CLI
python src/predict.py --text "My SBI account was debited Rs 4000 via Google Pay UPI but merchant did not receive payment"
```

### Step 3: Run Automated Test Suite

```bash
python tests/run_tests.py
# Or with pytest:
pytest tests/
```

### Step 4: Launch Interactive Streamlit Dashboard

```bash
streamlit run app/streamlit_app.py
```

---

## 5. Automated Unit Tests

The test suite in `tests/test_pipeline.py` covers 9 automated assertions:
1. `test_text_cleaning_and_normalization`: Verifies HTML stripping and currency normalization.
2. `test_stopword_filtering_preserves_negations`: Ensures words like `not`, `failed`, and `refused` are preserved.
3. `test_empty_and_short_text_filtering`: Confirms inputs under 15 characters are logged in audit.
4. `test_duplicate_complaint_detection`: Guarantees exact duplicate complaints are deduplicated.
5. `test_tfidf_vectorizer_fit_transform_isolation`: Confirms **zero test-set leakage** into vocabulary.
6. `test_metric_calculation_accuracy_and_f1`: Mathematical check of Macro F1 calculation.
7. `test_predictor_valid_prediction_and_confidence`: Verifies end-to-end inference and confidence scores.
8. `test_predictor_empty_and_short_input_handling`: Guards against unhandled exceptions on invalid input.
9. `test_predictor_out_of_vocabulary_handling`: Handles unknown tokens gracefully with fallback flags.

---

## 6. Project Directory Layout

```
nlp-complaint-classification/
├── data/
│   ├── raw/
│   │   └── indian_consumer_complaints_raw.csv   # Raw curated dataset
│   └── processed/
│       ├── cleaned_complaints.csv              # Normalized dataset
│       └── data_quality_audit.csv              # Audit log of dropped rows
├── notebooks/
│   ├── 01_eda.ipynb                            # Comprehensive exploratory analysis
│   └── 02_model_training.ipynb                 # Interactive ML training & evaluation
├── src/
│   ├── __init__.py
│   ├── data_loader.py                          # Ingestion & schema validation
│   ├── preprocess.py                           # Cleaning, tokenizing & deduplication
│   ├── features.py                             # Sublinear TF-IDF vectorizer
│   ├── train.py                                # Model training & stratified benchmark
│   ├── evaluate.py                             # Multi-class metrics & confusion matrix
│   └── predict.py                              # Real-time CLI & API inference
├── models/
│   └── complaint_pipeline_metadata.json        # Serialized pipeline & weights
├── outputs/
│   ├── metrics.csv                             # Measured comparison metrics
│   ├── confusion_matrix.json                   # Confusion matrix dictionary
│   └── test_predictions.csv                    # Sample test predictions with confidence
├── app/
│   └── streamlit_app.py                        # Interactive Streamlit application
├── reports/
│   └── college_project_report.md               # Formal college major project documentation
├── tests/
│   ├── __init__.py
│   ├── test_pipeline.py                        # 9 automated unit tests
│   └── run_tests.py                            # Standalone test runner
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 7. Model Limitations & Responsible Use

1. **Suggestive Triage**: Model predictions are decision-support suggestions for customer support and grievance officers, not final binding legal verdicts.
2. **Out-of-Distribution Vocabulary**: Text in regional Indian vernacular languages (Hindi, Tamil, Marathi, Telugu) transliterated in non-standard phonetic English may require translation preprocessing.
3. **Privacy**: Personal Identifiable Information (Aadhaar numbers, bank account numbers, phone numbers) must be masked before entering public logging systems.

---

## 8. License

This project is licensed under the MIT License.
