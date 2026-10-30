export interface CategoryDetail {
  id: string;
  name: string;
  description: string;
  examples: string[];
  keyTokens: string[];
  color: string;
  count: number;
}

export const CATEGORIES: CategoryDetail[] = [
  {
    id: "banking",
    name: "Banking & UPI Services",
    description: "Unauthorized debits, UPI PIN/gateway timeouts, credit card charges, loan processing delays, ATM dispensing failures, and RBI Ombudsman disputes.",
    examples: [
      "Amount of Rs 8,500 debited via Google Pay UPI to merchant but payment failed.",
      "Unauthorized credit card transaction of INR 24,999 on HDFC card without OTP.",
      "ATM debited Rs 10,000 cash from account but no cash was dispensed from machine."
    ],
    keyTokens: ["upi", "bank", "savings_account", "debited", "not_receive", "atm", "transaction", "credit_card"],
    color: "#6366F1",
    count: 228
  },
  {
    id: "ecommerce",
    name: "E-Commerce & Online Shopping",
    description: "Defective merchandise, counterfeit items, return refusals, delayed courier deliveries, and wallet refund disputes on Flipkart, Amazon, Meesho, etc.",
    examples: [
      "Ordered smartphone on Flipkart received soap bar inside sealed box.",
      "Returned defective air cooler on Amazon India but refund not initiated after 10 days.",
      "Delivered shoes missing one piece seller refusing replacement."
    ],
    keyTokens: ["ordered", "refund", "return", "flipkart", "amazon", "damaged", "seller", "delivery"],
    color: "#EC4899",
    count: 224
  },
  {
    id: "telecom",
    name: "Telecommunications & Internet",
    description: "Broadband fiber outages, call drops, unauthorized VAS subscriptions, MNP mobile number porting delays, and billing discrepancies.",
    examples: [
      "Airtel broadband fiber connection has been down for 6 consecutive days in Noida.",
      "Operator rejected MNP porting request citing fake legal dues.",
      "Postpaid mobile bill charged Rs 1,450 for international roaming never activated."
    ],
    keyTokens: ["broadband", "fiber", "porting", "airtel", "jio", "call_drops", "mnp", "downtime"],
    color: "#06B6D4",
    count: 222
  },
  {
    id: "utilities",
    name: "Consumer Utilities & Energy",
    description: "Inflated smart electricity bills, delayed LPG cylinder refill deliveries, water supply contamination, and pipeline connection issues.",
    examples: [
      "Received electricity bill of Rs 48,600 for domestic flat from BSES Delhi.",
      "Indane LPG gas cylinder booked 14 days ago not delivered, delivery boy demanding extra tip.",
      "Drinking water supply contaminated with sewage in municipal pipeline."
    ],
    keyTokens: ["electricity", "bill", "meter", "lpg", "cylinder", "cylinder_refill", "discom", "water_supply"],
    color: "#F59E0B",
    count: 221
  },
  {
    id: "electronics",
    name: "Consumer Electronics & Warranty",
    description: "Service center warranty denials, appliance manufacturing defects, laptop motherboard failures, and spare part unavailability.",
    examples: [
      "Samsung 55 inch OLED TV developed vertical green lines 2 months after purchase, warranty denied.",
      "Voltas AC compressor failed within 9 months, service center says gas leak excluded.",
      "Laptop motherboard dead after BIOS update, company claiming obsolete parts."
    ],
    keyTokens: ["warranty", "service_center", "display", "compressor", "laptop", "defect", "refrigerator", "replacement"],
    color: "#10B981",
    count: 220
  }
];

export interface ModelMetric {
  model_name: string;
  accuracy: number;
  macro_precision: number;
  macro_recall: number;
  macro_f1: number;
  weighted_f1: number;
  latency_ms: number;
  status: "Baseline" | "Candidate" | "Champion";
}

export const MODEL_METRICS: ModelMetric[] = [
  {
    model_name: "Majority Baseline",
    accuracy: 0.2151,
    macro_precision: 0.0430,
    macro_recall: 0.2000,
    macro_f1: 0.0708,
    weighted_f1: 0.0762,
    latency_ms: 0.05,
    status: "Baseline"
  },
  {
    model_name: "Multinomial Naive Bayes",
    accuracy: 0.9942,
    macro_precision: 0.9944,
    macro_recall: 0.9946,
    macro_f1: 0.9944,
    weighted_f1: 0.9942,
    latency_ms: 0.8,
    status: "Candidate"
  },
  {
    model_name: "Logistic Regression (L2 Softmax)",
    accuracy: 1.0000,
    macro_precision: 1.0000,
    macro_recall: 1.0000,
    macro_f1: 1.0000,
    weighted_f1: 1.0000,
    latency_ms: 1.2,
    status: "Champion"
  },
  {
    model_name: "Linear Support Vector Classifier",
    accuracy: 1.0000,
    macro_precision: 1.0000,
    macro_recall: 1.0000,
    macro_f1: 1.0000,
    weighted_f1: 1.0000,
    latency_ms: 0.9,
    status: "Candidate"
  }
];

export const CONFUSION_MATRIX: Record<string, Record<string, number>> = {
  "Banking & UPI Services": {
    "Banking & UPI Services": 37,
    "E-Commerce & Online Shopping": 0,
    "Telecommunications & Internet": 0,
    "Consumer Utilities & Energy": 0,
    "Consumer Electronics & Warranty": 0
  },
  "E-Commerce & Online Shopping": {
    "Banking & UPI Services": 0,
    "E-Commerce & Online Shopping": 35,
    "Telecommunications & Internet": 0,
    "Consumer Utilities & Energy": 0,
    "Consumer Electronics & Warranty": 0
  },
  "Telecommunications & Internet": {
    "Banking & UPI Services": 0,
    "E-Commerce & Online Shopping": 0,
    "Telecommunications & Internet": 34,
    "Consumer Utilities & Energy": 0,
    "Consumer Electronics & Warranty": 0
  },
  "Consumer Utilities & Energy": {
    "Banking & UPI Services": 0,
    "E-Commerce & Online Shopping": 0,
    "Telecommunications & Internet": 0,
    "Consumer Utilities & Energy": 33,
    "Consumer Electronics & Warranty": 0
  },
  "Consumer Electronics & Warranty": {
    "Banking & UPI Services": 0,
    "E-Commerce & Online Shopping": 0,
    "Telecommunications & Internet": 0,
    "Consumer Utilities & Energy": 0,
    "Consumer Electronics & Warranty": 33
  }
};

export const SAMPLE_PRESETS = [
  {
    id: "upi_fail",
    title: "SBI UPI Deduction Failure",
    category: "Banking & UPI Services",
    text: "Amount of Rs 8,500 was deducted from my State Bank of India savings account via Google Pay UPI to merchant, but payment failed. SBI customer care says money is with beneficiary bank, but merchant did not receive payment. Reference ID 4120984421."
  },
  {
    id: "ecom_refund",
    title: "Amazon India Refund Refusal",
    category: "E-Commerce & Online Shopping",
    text: "Returned a defective air cooler on Amazon India within 7-day return window. Courier pickup completed 10 days ago, but refund of Rs 6,499 is still not initiated into original payment source."
  },
  {
    id: "telecom_fiber",
    title: "Airtel Fiber Downtime",
    category: "Telecommunications & Internet",
    text: "Airtel broadband fiber connection has been down for 6 consecutive days in Sector 62 Noida. Daily automated SMS says issue will be resolved by 8 PM but no technician visits site."
  },
  {
    id: "power_bill",
    title: "BSES Electricity Bill Inflation",
    category: "Consumer Utilities & Energy",
    text: "Received electricity bill of Rs 48,600 for domestic residential flat from BSES Rajdhani Delhi. Average monthly bill is Rs 2,500. Smart meter suspected of running 10x faster."
  },
  {
    id: "warranty_denial",
    title: "Samsung OLED Display Lines",
    category: "Consumer Electronics & Warranty",
    text: "Samsung 55 inch OLED TV developed vertical green lines across screen just 2 months after purchase. Authorized service center demanding Rs 18,000 for panel replacement claiming warranty exclusion."
  }
];

export const UNIT_TESTS = [
  {
    id: "schema_val",
    name: "Schema & Mandatory Columns Validation",
    target: "src/data_loader.py",
    assertion: "assert all(col in header for col in REQUIRED_COLUMNS)",
    description: "Verifies that complaint_id, complaint_text, category, date, and state exist with zero schema corruption."
  },
  {
    id: "text_clean",
    name: "HTML Stripping & Currency Normalization",
    target: "src/preprocess.py",
    assertion: "assert 'currency_amount' in cleaned and '<b>' not in cleaned",
    description: "Ensures Indian currency symbols (Rs./INR/₹) map to normalized tokens while markup is stripped."
  },
  {
    id: "negation_pres",
    name: "Negation Word Preservation in Tokenizer",
    target: "src/preprocess.py",
    assertion: "assert all(w in tokens for w in ['not', 'refused', 'failed'])",
    description: "Confirms critical negation words are retained to prevent inverted complaint sentiment."
  },
  {
    id: "empty_filter",
    name: "Short & Empty Text Filtering (<15 chars)",
    target: "src/preprocess.py",
    assertion: "assert len(audit_log) == 2 and len(cleaned) == 1",
    description: "Validates that empty strings and uninformative snippets are filtered into data quality audit log."
  },
  {
    id: "dedup_check",
    name: "Exact Duplicate Detection & Quarantine",
    target: "src/preprocess.py",
    assertion: "assert audit[0]['action'] == 'DEDUPLICATED'",
    description: "Prevents identical consumer grievances from leaking across train and test partitions."
  },
  {
    id: "zero_leakage",
    name: "TF-IDF Fit-on-Train Isolation (Zero Leakage)",
    target: "src/features.py",
    assertion: "assert 'airtel' not in vectorizer.vocabulary",
    description: "Guarantees test documents do not expand vocabulary or alter training IDF weights."
  },
  {
    id: "metric_calc",
    name: "Macro F1 Multi-Class Calculation",
    target: "src/evaluate.py",
    assertion: "assert 0.0 <= metrics['macro_f1'] <= 1.0",
    description: "Validates unweighted mean of per-class F1 scores across all 5 imbalanced sectors."
  },
  {
    id: "predict_valid",
    name: "End-to-End Inference & Calibrated Confidence",
    target: "src/predict.py",
    assertion: "assert res['confidence'] > 0.70 and sum(probs.values()) == 1.0",
    description: "Verifies probability normalization and correct sector assignment on unseen test text."
  },
  {
    id: "oov_fallback",
    name: "Out-of-Vocabulary Graceful Handling",
    target: "src/predict.py",
    assertion: "assert res['status'] in ['uncertain', 'success']",
    description: "Ensures nonsense input flags manual review rather than crashing or throwing 500 error."
  }
];
