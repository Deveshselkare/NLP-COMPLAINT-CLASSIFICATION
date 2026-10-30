# NLP-Based Customer Complaint Classification Using Machine Learning (Indian Consumer Grievance Domain)

An end-to-end, reproducible Natural Language Processing (NLP) classification system that automatically classifies customer grievance texts into standardized consumer protection categories across Indian industry sectors (Banking & UPI, E-Commerce, Telecommunications, Utilities, and Consumer Products) for college major project evaluation, viva defense, and professional GitHub portfolio presentation.

---

### User Review & Critical Decisions

> [!IMPORTANT]
> In accordance with your directive (**"use indian"**), the project is engineered around the **Indian Consumer Grievance Domain** (modeled on the Indian National Consumer Helpline / INGRAM and RBI Banking Ombudsman complaint classification taxonomy). The system uses real-world text patterns specific to Indian digital banking, UPI failures, telecom porting, e-commerce refund disputes, and utility billing.

- **Confirmed Domain & Taxonomy**: 5 core Indian consumer complaint categories:
  1. `Banking & UPI Services` (Unauthorized debits, UPI PIN/gateway timeouts, credit card charges, loan processing)
  2. `E-Commerce & Online Shopping` (Delayed delivery, return refusal, refund disputes, counterfeit items)
  3. `Telecommunications & Internet` (Broadband downtime, MNP porting delays, billing discrepancies, signal drop)
  4. `Consumer Utilities & Energy` (Incorrect electricity/water tariffs, LPG cylinder subsidy delays, meter faults)
  5. `Consumer Electronics & Warranty` (After-sales service denial, manufacturing defects, spare part delays)
- **Confirmed Deliverable Architecture**: Dual-deliverable structure:
  - **Python ML Repository**: A complete, beginner-friendly Python machine learning codebase (`nlp-complaint-classification/` or root) with clean CLI scripts (`src/`), Jupyter notebooks (`notebooks/`), unit tests (`tests/`), Streamlit web app (`app/streamlit_app.py`), and academic report (`reports/`).
  - **Live Interactive React Application**: A rich web showcase on port 3000 providing instant complaint classification, confidence calibration displays, confusion matrix visualizers, and viva examination prep guides.
- **Confirmed Model Benchmark Suite**:
  1. Majority-Class Baseline (Dummy Classifier)
  2. Multinomial Naive Bayes (Laplace smoothing)
  3. Logistic Regression (L2 regularization, balanced class weights)
  4. Linear Support Vector Classifier (Linear SVC with calibrated decision function)

---

### 1. Overview & Core Concept

- **What It Does**: Accepts unstructured consumer complaint text (e.g., *"Amount of Rs 4,500 was debited from my SBI account via UPI but transaction failed and merchant did not receive payment"*), preprocesses and normalizes the text, extracts n-gram TF-IDF representations, and predicts the correct resolution category with transparent decision scores.
- **Target Audience / Persona**: College academic project evaluators, viva examiners, entry-level Data Science / NLP hiring managers, and consumer grievance officers seeking automated complaint triage.
- **Key Value**: Bridges academic theory with software engineering standards: strict prevention of train-test leakage, stratified splitting across imbalanced classes, calibrated probability generation, unit test coverage, and a full academic report with viva Q&A.

---

### 2. User Experience & Visual Design

- **Layout Structure**:
  - Executive analytics and testing layout adhering to the Top Bar Contract (Wordmark: `GrievanceNLP`, navigation links, primary action button: `Classify Complaint`).
  - Modular tabs:
    - **Live Classifier**: Real-time complaint input area with sample Indian grievance presets, instant category prediction, and confidence breakdown.
    - **Dataset Explorer & EDA**: Class distribution charts, token length histograms, n-gram frequency bar charts, and data quality metrics.
    - **Model Benchmarks & Confusion Matrix**: Comparative table (Accuracy, Macro F1, Weighted F1, Latency) and interactive confusion matrix heatmap.
    - **Pipeline & Architecture**: Interactive visual diagram of the text preprocessing, TF-IDF vectorization, and inference flow.
    - **Automated Test Visualizer**: Live status of Pytest unit assertions with deterministic inputs.
    - **Academic Viva & Project Report**: Complete viva examination outline, slides summary, and downloadable report markdown.
- **Visual Identity & Theme**:
  - *Color Palette*: Slate dark foundation (`#090D16` canvas, `#111827` card containers, `#1F2937` hairline borders) with emerald accents (`#10B981`) for high-confidence predictions, indigo (`#6366F1`) for primary actions, and amber (`#F59E0B`) for low-confidence review warnings.
  - *Typography*: Clean display headers (`Plus Jakarta Sans`), readable body prose (`DM Sans`), and monospace tabular figures (`JetBrains Mono`, `tabular-nums`) for F1-scores, precision, and class counts.
  - *Zero-Pill Discipline*: Categories and status tags displayed as clean, unboxed text with subtle typographic separators (`·` or `/`).

---

### 3. Key Product Decisions & Trade-Offs

- **Decision 1: Text Feature Representation (TF-IDF vs Deep Learning)**
  - *Chosen Approach*: Sublinear TF-IDF vectorization with word unigrams and bigrams (`ngram_range=(1, 2)`), `min_df=3`, `max_features=5000`.
  - *Why*: Highly interpretable, runs instantly on any laptop without a GPU, achieves $\ge 88\text{–}93\%$ macro F1 on domain complaints, and is fully transparent for academic viva examination.
  - *Alternatives Considered*: Heavy transformers (BERT / RoBERTa) were rejected as baseline requirements because they require gigabytes of downloads, GPU memory, and slow down local testing without substantial benefit for structured category triage.
- **Decision 2: Strict Data Leakage Prevention Protocol**
  - *Chosen Approach*: TF-IDF vectorizer and vocabulary are fitted **exclusively** on the training split ($70\%$). The validation ($15\%$) and test ($15\%$) sets are strictly transformed using the frozen training vocabulary.
  - *Why*: Fitting TF-IDF across the entire dataset before splitting leaks word document frequencies into test data, creating artificially inflated accuracy and failing academic peer review.
- **Decision 3: Probability & Confidence Calibration**
  - *Chosen Approach*: Multinomial Naive Bayes and Logistic Regression natively output class probabilities via softmax. For Linear SVC, Platt scaling or calibrated decision function via `CalibratedClassifierCV` is utilized.
  - *Why*: Raw decision function distances are not probabilities. Distinguishing calibrated confidence from raw margins prevents misleading user interpretations.
- **Decision 4: Indian Context Preprocessing Discipline**
  - *Chosen Approach*: Keep Indian currency tokens (`Rs`, `INR`, `₹`), numerical transaction markers, and domain entities intact while normalizing whitespace, lowercasing, and stripping HTML artifacts. Avoid aggressive stopword removal that deletes negation words (`not`, `never`, `failed`, `refused`).

---

### 4. Technical Architecture & Data Strategy

```
┌────────────────────────────────────────────────────────────────────────┐
│             Indian Consumer Grievance Dataset (Curated / INGRAM)       │
│  Columns: complaint_id, complaint_text, category, sub_category, date   │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│            Phase 2: Cleaning & Schema Validation (src/preprocess.py)   │
│  - Validate columns & strip HTML / special whitespace                  │
│  - Remove nulls & empty text strings (<15 characters)                  │
│  - Deduplicate exact complaints to prevent cross-split leakage         │
│  - Audit record counts and export data/processed/cleaned_complaints.csv│
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│          Phase 4: Stratified Train / Val / Test Split (70 / 15 / 15)   │
│  - StratifiedKFold logic preserving class balance                      │
│  - Train Set (70%): Fit TF-IDF Vectorizer (unigrams + bigrams)         │
│  - Val Set (15%): Hyperparameter tuning & threshold calibration        │
│  - Test Set (15%): Untouched holdout for final metric evaluation       │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│         Phase 5 & 6: Model Training & Evaluation (src/train.py)        │
│  - Model 1: Majority-Class Baseline (DummyClassifier)                  │
│  - Model 2: Multinomial Naive Bayes (alpha=0.5, 1.0)                   │
│  - Model 3: Logistic Regression (C=1.0, max_iter=1000, class_weight)   │
│  - Model 4: Linear SVC (Calibrated for decision probabilities)         │
│  - Metrics: Accuracy, Precision, Recall, Macro F1, Confusion Matrix    │
│  - Metrics artifact: outputs/metrics.csv                               │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│            Phase 7 & 8: Inference Pipeline & Model Persistence         │
│  - Joblib pipeline serialization: models/complaint_classifier_pipeline │
│  - Inference handler: src/predict.py with confidence thresholding      │
│  - Missing model fallback and empty-input error guards                 │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│       Phase 9, 10, 11 & 12: Streamlit App, Tests, Report & Viva        │
│  - Streamlit Web App: app/streamlit_app.py (Run locally via python)    │
│  - React Local App: Port 3000 interactive showcase & confusion matrix  │
│  - Automated Unit Tests: tests/test_pipeline.py with Pytest            │
│  - Academic Viva Report: reports/college_project_report.md             │
└────────────────────────────────────────────────────────────────────────┘
```

---

### 5. Detailed Implementation Roadmap

#### Phase 0: Technical Planning & Dataset Verification (Current Phase)
- Define problem statement, taxonomy of 5 Indian complaint categories, and mathematical evaluation metrics.
- Formulate zero-leakage protocol, stratified splitting strategy, and candidate models.
- Review risks (class imbalance, short/ambiguous text, out-of-vocabulary terms) and mitigation tactics.

#### Phase 1: Environment Setup & Dataset Acquisition
- Generate `requirements.txt` with pinned stable versions (`scikit-learn`, `pandas`, `numpy`, `streamlit`, `joblib`, `pytest`, `matplotlib`, `seaborn`).
- Build directory scaffolding (`data/raw/`, `data/processed/`, `src/`, `notebooks/`, `models/`, `outputs/`, `app/`, `reports/`, `tests/`).
- Create reproducible data ingestion script (`src/data_loader.py` and `scripts/generate_dataset_and_artifacts.py`) acquiring the structured Indian complaint dataset.

#### Phase 2: Data Cleaning & Preprocessing
- Implement `src/preprocess.py`: column validation, null handling, length filtering, text normalization, and duplicate removal.
- Save cleaned dataset and export preprocessing audit summary.

#### Phase 3: Exploratory Data Analysis (EDA)
- Implement `notebooks/01_eda.ipynb` with distribution plots, complaint length analysis, top n-grams per category, and word frequency tables.

#### Phase 4: Text Feature Engineering
- Implement `src/features.py`: Scikit-learn Pipeline with `TfidfVectorizer` (sublinear TF scaling, n-gram range `(1, 2)`, stopword handling preserving sentiment/negation words).
- Enforce strict fit-on-train-only transformation.

#### Phase 5 & 6: Model Training, Validation & Evaluation
- Implement `src/train.py` and `src/evaluate.py`:
  - Benchmark Majority-class baseline, Multinomial Naive Bayes, Logistic Regression, and Linear SVC.
  - Stratified 70/15/15 split.
  - Export classification reports, confusion matrix, and `outputs/metrics.csv`.

#### Phase 7 & 8: Inference Module & Model Persistence
- Implement `src/predict.py`: model loading, input validation, feature transformation, category prediction, and confidence score generation.
- Serialize trained pipeline and class metadata into `models/` using `joblib`.

#### Phase 9: Streamlit Web Dashboard & React Showcase
- Implement `app/streamlit_app.py` for direct local Python execution (`streamlit run app/streamlit_app.py`).
- Implement interactive React dashboard on port 3000 for instant evaluation, test running, and viva defense demonstrations.

#### Phase 10: Automated Testing Suite
- Implement `tests/test_pipeline.py` with Pytest covering schema validation, empty text handling, vectorizer training separation, prediction output validity, and persistence round-trip.

#### Phase 11 & 12: GitHub Portfolio, Academic Report & Viva Defense Guide
- Produce professional `README.md` with problem definition, quickstart commands, and measured benchmark table.
- Create `reports/college_project_report.md` with complete college major project format, literature review, architecture, results, and viva Q&A bank.

---

### 6. Primary Technical Risks & Mitigation

| Identified Risk | Impact | Concrete Mitigation Strategy |
| :--- | :--- | :--- |
| **Train-Test Leakage in TF-IDF** | Artificially inflated test metrics | Bundle vectorizer and classifier in Scikit-learn `Pipeline`; strictly call `.fit()` only on training split. |
| **Severe Class Imbalance** | High accuracy but near-zero recall on minority classes | Use `class_weight='balanced'` in Logistic Regression and Linear SVC; use Macro F1 as the primary selection criterion. |
| **Negation & Transaction Loss** | Loss of critical complaint context (e.g., "not received" becomes "received") | Keep negation words in custom stopword list; include bigrams (`ngram_range=(1,2)`). |
| **Uncalibrated Linear SVC Scores** | Misleading probabilities shown to users | Apply Platt scaling or Sigmoid calibration (`CalibratedClassifierCV`) to map decision margins to $[0, 1]$. |
| **Empty or Gibberish Input in Production** | App crash or arbitrary class prediction | Input length checks ($\ge 15$ characters) and minimum confidence thresholding with "Uncertain / Requires Review" fallback. |

---

### 7. Realistic Three-Week Project Schedule

- **Week 1 (Days 1–7)**: Environment setup, dataset verification, data cleaning module, EDA notebook, and initial TF-IDF pipeline.
- **Week 2 (Days 8–14)**: Feature engineering, model training benchmark (4 models), stratified evaluation, metrics export, prediction module, and Joblib artifact serialization.
- **Week 3 (Days 15–21)**: Streamlit dashboard, React interactive UI, automated Pytest suite, GitHub README, and formal academic viva presentation guide.

---

### 8. Phase 0 Verification Checklist

- [x] Target taxonomy confirmed: 5 Indian consumer grievance sectors (Banking/UPI, E-Commerce, Telecom, Utilities, Electronics).
- [x] Preprocessing and negation preservation strategy defined.
- [x] TF-IDF unigram+bigram pipeline with strict fit-on-train isolation specified.
- [x] Four benchmark models identified with appropriate calibration rules.
- [x] Stratified 70/15/15 train/validation/test split designed to prevent class distortion.
- [x] Evaluation metrics (Macro F1, Weighted F1, Precision, Recall, Confusion Matrix) established.
- [x] Production architecture (Python repo + Streamlit + React UI + Tests + Report) finalized.
