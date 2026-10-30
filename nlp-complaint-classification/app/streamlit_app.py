"""
app/streamlit_app.py
Interactive Streamlit Web Dashboard for Indian Consumer Complaint Classification.
Run locally via:
    streamlit run app/streamlit_app.py
"""

import os
import sys
import json
import csv
from typing import Dict, Any

# Ensure project imports resolve
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

try:
    import streamlit as st
    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False

from src.predict import ComplaintPredictor

CATEGORIES = [
    "Banking & UPI Services",
    "E-Commerce & Online Shopping",
    "Telecommunications & Internet",
    "Consumer Utilities & Energy",
    "Consumer Electronics & Warranty"
]

CATEGORY_DESCRIPTIONS = {
    "Banking & UPI Services": "UPI payment timeouts, unauthorized debit card transactions, loan processing delays, ATM dispensing failures, and credit card disputes.",
    "E-Commerce & Online Shopping": "Defective merchandise, missing items, return refusals, delayed courier deliveries, and wallet refund disputes.",
    "Telecommunications & Internet": "Broadband fiber outages, call drops, unauthorized VAS subscriptions, MNP mobile number porting delays, and billing discrepancies.",
    "Consumer Utilities & Energy": "Inflated smart electricity bills, delayed LPG cylinder refill deliveries, water supply contamination, and pipeline connection issues.",
    "Consumer Electronics & Warranty": "Service center warranty denials, appliance manufacturing defects, laptop motherboard failures, and spare part unavailability."
}

SAMPLE_PRESETS = {
    "Banking / UPI Failure": "Amount of Rs 8500 was deducted from my State Bank of India savings account via Google Pay UPI to merchant, but payment failed. SBI customer care says money is with beneficiary bank, but merchant did not receive money. Reference ID 4120984421.",
    "E-Commerce Refund Dispute": "Returned a defective air cooler on Amazon India within 7-day return window. Courier pickup completed 10 days ago, but refund of Rs 6,499 is still not initiated into original payment mode.",
    "Telecom Fiber Downtime": "Airtel broadband fiber connection has been down for 6 consecutive days in Sector 62 Noida. Daily automated SMS says issue will be resolved by 8 PM but no technician visits.",
    "Electricity Bill Inflation": "Received electricity bill of Rs 48,600 for domestic residential flat from BSES Rajdhani Delhi. Average monthly bill is Rs 2,500. Smart meter suspected of running 10x faster.",
    "Warranty Claim Denial": "Samsung 55 inch OLED TV developed vertical green lines across screen just 2 months after purchase. Authorized service center demanding Rs 18,000 for panel replacement claiming warranty exclusion."
}

def load_metrics():
    metrics_path = os.path.join(PROJECT_ROOT, "outputs", "metrics.csv")
    if not os.path.exists(metrics_path):
        return []
    records = []
    with open(metrics_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            records.append(r)
    return records

def load_confusion_matrix():
    cm_path = os.path.join(PROJECT_ROOT, "outputs", "confusion_matrix.json")
    if not os.path.exists(cm_path):
        return {}
    with open(cm_path, "r", encoding="utf-8") as f:
        return json.load(f)

def run_streamlit():
    if not STREAMLIT_AVAILABLE:
        print("Streamlit is not installed in the local environment. Install via `pip install streamlit`.")
        return

    st.set_page_config(
        page_title="Indian Consumer Grievance NLP Classifier",
        page_icon="⚖️",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    st.title("⚖️ Indian Consumer Complaint Classification")
    st.markdown("**Automated NLP Triage for National Consumer Helpline (INGRAM) and RBI Banking Ombudsman Categories**")
    st.markdown("---")

    # Sidebar Navigation
    st.sidebar.header("Navigation")
    app_mode = st.sidebar.radio(
        "Select View",
        ["🔍 Live Complaint Classifier", "📊 Dataset & EDA", "📈 Model Benchmarks & Confusion Matrix", "📋 Academic Project & Viva Outline"]
    )

    predictor = None
    try:
        predictor = ComplaintPredictor()
    except Exception as e:
        st.sidebar.error(f"Model artifact warning: {e}")

    if app_mode == "🔍 Live Complaint Classifier":
        st.subheader("Test Real-Time Grievance Classification")
        st.write("Enter an unstructured customer complaint or select an authentic Indian grievance preset below:")

        col_preset1, col_preset2, col_preset3, col_preset4, col_preset5 = st.columns(5)
        selected_text = ""
        
        with col_preset1:
            if st.button("Banking / UPI"):
                selected_text = SAMPLE_PRESETS["Banking / UPI Failure"]
        with col_preset2:
            if st.button("E-Commerce"):
                selected_text = SAMPLE_PRESETS["E-Commerce Refund Dispute"]
        with col_preset3:
            if st.button("Telecom"):
                selected_text = SAMPLE_PRESETS["Telecom Fiber Downtime"]
        with col_preset4:
            if st.button("Utilities"):
                selected_text = SAMPLE_PRESETS["Electricity Bill Inflation"]
        with col_preset5:
            if st.button("Electronics"):
                selected_text = SAMPLE_PRESETS["Warranty Claim Denial"]

        complaint_input = st.text_area(
            "Complaint Text:",
            value=selected_text,
            height=130,
            placeholder="Type customer grievance here (e.g. 'Amount debited via UPI but merchant payment failed...')"
        )

        if st.button("Classify Complaint", type="primary"):
            if not complaint_input or len(complaint_input.strip()) < 10:
                st.warning("Please enter a valid complaint text (at least 10 characters).")
            elif predictor is None:
                st.error("Model artifacts not loaded. Run `python3 src/train.py` first.")
            else:
                with st.spinner("Analyzing text representations & TF-IDF vectors..."):
                    result = predictor.predict(complaint_input)

                if result["status"] == "error":
                    st.error(result["message"])
                elif result["status"] == "uncertain":
                    st.warning(f"⚠️ {result['message']}")
                else:
                    pred_cat = result["predicted_category"]
                    conf = result["confidence"]

                    st.success(f"**Predicted Category:** {pred_cat}")
                    st.metric("Calibrated Model Confidence", f"{conf * 100:.2f}%")

                    st.markdown(f"**Category Description:** {CATEGORY_DESCRIPTIONS.get(pred_cat, '')}")
                    
                    if result.get("top_salient_terms"):
                        st.markdown(f"**Key Salient N-Grams:** `{', '.join(result['top_salient_terms'])}`")

                    st.subheader("Probability Distribution Across All Categories")
                    for cat, prob in result["probabilities"].items():
                        st.write(f"**{cat}** — {prob*100:.1f}%")
                        st.progress(float(prob))

    elif app_mode == "📊 Dataset & EDA":
        st.subheader("Dataset Structure & Exploratory Data Analysis")
        st.write("Source: Curated Indian Consumer Grievance Dataset (INGRAM & RBI Ombudsman taxonomy)")

        metrics_cols = st.columns(4)
        metrics_cols[0].metric("Total Raw Records", "1,200")
        metrics_cols[1].metric("Cleaned Valid Records", "1,115")
        metrics_cols[2].metric("Target Categories", "5 Classes")
        metrics_cols[3].metric("Features Extracted", "1,651 N-Grams")

        st.markdown("### Standard Target Taxonomy")
        for cat, desc in CATEGORY_DESCRIPTIONS.items():
            st.markdown(f"- **{cat}**: {desc}")

        st.markdown("### Preprocessing Pipeline Rules")
        st.markdown("""
        1. **HTML & URL Removal**: Strips markup and link noise.
        2. **Indian Currency Normalization**: Converts `Rs.`, `INR`, and `₹` amounts to token `currency_amount` so financial figures retain predictive semantic signal.
        3. **Negation Preservation**: Stopword filtering deliberately retains `not`, `no`, `never`, `failed`, `refused` to prevent inverted complaint sentiment.
        4. **Sublinear TF-IDF**: Evaluates unigrams and bigrams (`ngram_range=(1,2)`) with L2 normalization.
        5. **Strict Train-Test Isolation**: Vectorizer fitted ONLY on 70% train split.
        """)

    elif app_mode == "📈 Model Benchmarks & Confusion Matrix":
        st.subheader("Model Evaluation & Holdout Benchmark")
        metrics = load_metrics()

        if metrics:
            st.table(metrics)
        else:
            st.info("Metrics file outputs/metrics.csv not found. Run python3 src/train.py to generate.")

        st.subheader("Holdout Confusion Matrix (Logistic Regression)")
        cm = load_confusion_matrix()
        if cm:
            st.json(cm)
        else:
            st.info("Confusion matrix file not found.")

    elif app_mode == "📋 Academic Project & Viva Outline":
        st.subheader("College Major Project & Viva Defense Guide")
        st.markdown("""
        ### Key Viva Questions & Answers:
        **Q1: Why use TF-IDF instead of simple Bag of Words (BoW)?**
        *Answer:* BoW merely counts word frequencies, giving high weights to common non-informative words. TF-IDF applies inverse document frequency penalization, assigning higher weights to distinctive domain-specific terms (like 'upi', 'meter', 'warranty', 'broadband').

        **Q2: How is data leakage prevented in this pipeline?**
        *Answer:* The TF-IDF vectorizer's vocabulary and IDF weights are computed strictly on the 70% training split. Validation and test sets are transformed using the frozen vocabulary without recalculating IDF or expanding features.

        **Q3: Why is Macro F1 preferred over Accuracy?**
        *Answer:* Real-world complaint queues often have severe class imbalance. A naive baseline that always guesses the majority class can get 60% accuracy while failing completely on minority categories. Macro F1 calculates the unweighted mean of F1 scores across all classes, treating each category with equal importance.
        """)

if __name__ == "__main__":
    if STREAMLIT_AVAILABLE:
        run_streamlit()
    else:
        print("To run the Streamlit dashboard, install streamlit and run:")
        print("  streamlit run app/streamlit_app.py")
