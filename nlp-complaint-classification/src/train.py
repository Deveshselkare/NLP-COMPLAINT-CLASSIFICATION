"""
src/train.py
Model training pipeline for Indian Consumer Grievance Classification.
Benchmarks 4 models:
  1. Majority-Class Baseline
  2. Multinomial Naive Bayes
  3. Logistic Regression (L2 Softmax)
  4. Linear Support Vector Classifier (One-vs-Rest Pegasos Hinge Loss)
"""

import os
import sys
import json
import csv
import random
from collections import defaultdict, Counter
from typing import Dict, Any, List

# Ensure package imports resolve
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.features import ComplaintTFIDFVectorizer
from src.evaluate import evaluate_classifier

RANDOM_SEED = 42
random.seed(RANDOM_SEED)

CATEGORIES = [
    "Banking & UPI Services",
    "E-Commerce & Online Shopping",
    "Telecommunications & Internet",
    "Consumer Utilities & Energy",
    "Consumer Electronics & Warranty"
]

class MajorityBaselineModel:
    def __init__(self):
        self.majority_class = None
        self.classes = []
        self.distribution = {}

    def fit(self, y: List[str]):
        counts = Counter(y)
        self.majority_class = counts.most_common(1)[0][0]
        self.classes = sorted(list(counts.keys()))
        total = len(y)
        self.distribution = {c: counts[c] / total for c in self.classes}
        return self

    def predict_single(self, vec: Dict[int, float]):
        return self.majority_class, dict(self.distribution)

class MultinomialNBModel:
    def __init__(self, alpha: float = 0.8):
        self.alpha = alpha
        self.class_priors = {}
        self.feature_log_probs = {}
        self.classes = []

    def fit(self, X: List[Dict[int, float]], y: List[str], num_features: int):
        self.classes = sorted(list(set(y)))
        total_docs = len(y)
        class_docs = Counter(y)

        for c in self.classes:
            self.class_priors[c] = (class_docs[c] / total_docs)

        class_feature_totals = {c: defaultdict(float) for c in self.classes}
        class_total_weights = {c: 0.0 for c in self.classes}

        for vec, label in zip(X, y):
            for idx, val in vec.items():
                class_feature_totals[label][idx] += val
                class_total_weights[label] += val

        import math
        self.feature_log_probs = {c: {} for c in self.classes}
        for c in self.classes:
            denom = class_total_weights[c] + self.alpha * num_features
            denom_log = math.log(denom)
            for idx in range(num_features):
                num = class_feature_totals[c].get(idx, 0.0) + self.alpha
                self.feature_log_probs[c][idx] = math.log(num) - denom_log

        return self

    def predict_single(self, vec: Dict[int, float]):
        import math
        scores = {}
        for c in self.classes:
            s = math.log(self.class_priors[c])
            lp = self.feature_log_probs[c]
            for idx, val in vec.items():
                if idx in lp:
                    s += val * lp[idx]
            scores[c] = s

        max_s = max(scores.values())
        exp_s = {c: math.exp(scores[c] - max_s) for c in self.classes}
        sum_exp = sum(exp_s.values())
        probs = {c: exp_s[c] / sum_exp for c in self.classes}
        return max(probs, key=probs.get), probs

class LogisticRegressionModel:
    def __init__(self, lr: float = 0.5, epochs: int = 65, l2_reg: float = 0.001):
        self.lr = lr
        self.epochs = epochs
        self.l2_reg = l2_reg
        self.weights = {}
        self.bias = {}
        self.classes = []

    def fit(self, X: List[Dict[int, float]], y: List[str], num_features: int):
        import math
        self.classes = sorted(list(set(y)))
        for c in self.classes:
            self.weights[c] = [0.0] * num_features
            self.bias[c] = 0.0

        N = len(y)
        indices = list(range(N))

        for epoch in range(self.epochs):
            random.shuffle(indices)
            curr_lr = self.lr / (1.0 + 0.05 * epoch)

            for i in indices:
                vec = X[i]
                true_c = y[i]

                logits = {}
                for c in self.classes:
                    s = self.bias[c]
                    w = self.weights[c]
                    for idx, val in vec.items():
                        s += w[idx] * val
                    logits[c] = s

                max_l = max(logits.values())
                exps = {c: math.exp(logits[c] - max_l) for c in self.classes}
                sum_e = sum(exps.values())
                probs = {c: exps[c] / sum_e for c in self.classes}

                for c in self.classes:
                    target = 1.0 if c == true_c else 0.0
                    err = probs[c] - target
                    self.bias[c] -= curr_lr * err
                    w = self.weights[c]
                    for idx, val in vec.items():
                        grad = err * val + self.l2_reg * w[idx]
                        w[idx] -= curr_lr * grad

        return self

    def predict_single(self, vec: Dict[int, float]):
        import math
        logits = {}
        for c in self.classes:
            s = self.bias[c]
            w = self.weights[c]
            for idx, val in vec.items():
                s += w[idx] * val
            logits[c] = s

        max_l = max(logits.values())
        exps = {c: math.exp(logits[c] - max_l) for c in self.classes}
        sum_e = sum(exps.values())
        probs = {c: exps[c] / sum_e for c in self.classes}
        return max(probs, key=probs.get), probs

class LinearSVCModel:
    def __init__(self, lr: float = 0.1, epochs: int = 40, C: float = 1.0):
        self.lr = lr
        self.epochs = epochs
        self.C = C
        self.weights = {}
        self.bias = {}
        self.classes = []

    def fit(self, X: List[Dict[int, float]], y: List[str], num_features: int):
        self.classes = sorted(list(set(y)))
        N = len(y)

        for c in self.classes:
            w = [0.0] * num_features
            b = 0.0
            y_binary = [1.0 if label == c else -1.0 for label in y]

            for epoch in range(self.epochs):
                curr_lr = self.lr / (1.0 + 0.02 * epoch)
                for i in range(N):
                    vec = X[i]
                    yi = y_binary[i]
                    margin = b + sum(w[idx] * val for idx, val in vec.items())
                    
                    if yi * margin < 1.0:
                        b += curr_lr * yi
                        for idx, val in vec.items():
                            w[idx] += curr_lr * (self.C * yi * val - 0.001 * w[idx])

            self.weights[c] = w
            self.bias[c] = b

        return self

    def predict_single(self, vec: Dict[int, float]):
        import math
        scores = {}
        for c in self.classes:
            score = self.bias[c] + sum(self.weights[c][idx] * val for idx, val in vec.items())
            scores[c] = score

        calibrated = {}
        for c, s in scores.items():
            calibrated[c] = 1.0 / (1.0 + math.exp(-max(-15.0, min(15.0, s))))
        sum_cal = sum(calibrated.values())
        probs = {c: calibrated[c] / sum_cal for c in self.classes}

        return max(scores, key=scores.get), probs

def run_training_pipeline(base_dir: str = None) -> Dict[str, Any]:
    if base_dir is None:
        base_dir = PROJECT_ROOT

    cleaned_csv = os.path.join(base_dir, "data", "processed", "cleaned_complaints.csv")
    if not os.path.exists(cleaned_csv):
        raise FileNotFoundError(f"Cleaned dataset not found at {cleaned_csv}. Run preprocess.py first.")

    records = []
    with open(cleaned_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            records.append(r)

    by_cat = defaultdict(list)
    for r in records:
        by_cat[r["category"]].append(r)

    train_data = []
    val_data = []
    test_data = []

    for cat, items in by_cat.items():
        random.shuffle(items)
        n = len(items)
        n_train = int(n * 0.70)
        n_val = int(n * 0.15)
        train_data.extend(items[:n_train])
        val_data.extend(items[n_train:n_train + n_val])
        test_data.extend(items[n_train + n_val:])

    train_texts = [r["cleaned_text"] for r in train_data]
    train_labels = [r["category"] for r in train_data]

    # Fit TF-IDF exclusively on train data
    vectorizer = ComplaintTFIDFVectorizer(max_features=3000, min_df=2, ngram_range=(1, 2))
    vectorizer.fit(train_texts)

    X_train = vectorizer.transform(train_texts)
    y_train = train_labels

    X_test = vectorizer.transform([r["cleaned_text"] for r in test_data])
    y_test = [r["category"] for r in test_data]
    num_features = len(vectorizer.vocabulary)

    models = {
        "Majority Baseline": MajorityBaselineModel().fit(y_train),
        "Multinomial Naive Bayes": MultinomialNBModel(alpha=0.8).fit(X_train, y_train, num_features),
        "Logistic Regression": LogisticRegressionModel(lr=0.5, epochs=65).fit(X_train, y_train, num_features),
        "Linear SVC": LinearSVCModel(lr=0.1, epochs=40).fit(X_train, y_train, num_features)
    }

    metrics_rows = []
    evaluations = {}

    for name, model in models.items():
        preds = []
        for vec in X_test:
            pred_c, _ = model.predict_single(vec)
            preds.append(pred_c)
        res = evaluate_classifier(y_test, preds, CATEGORIES)
        evaluations[name] = res
        metrics_rows.append({
            "model_name": name,
            "accuracy": res["accuracy"],
            "macro_precision": res["macro_precision"],
            "macro_recall": res["macro_recall"],
            "macro_f1": res["macro_f1"],
            "weighted_f1": res["weighted_f1"]
        })

    # Save metrics.csv
    outputs_dir = os.path.join(base_dir, "outputs")
    os.makedirs(outputs_dir, exist_ok=True)
    metrics_path = os.path.join(outputs_dir, "metrics.csv")
    with open(metrics_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["model_name", "accuracy", "macro_precision", "macro_recall", "macro_f1", "weighted_f1"])
        writer.writeheader()
        writer.writerows(metrics_rows)

    # Save best model pipeline metadata
    best_model = models["Logistic Regression"]
    pipeline_artifact = {
        "model_name": "Logistic Regression",
        "classes": CATEGORIES,
        "feature_count": num_features,
        "vocabulary": vectorizer.vocabulary,
        "idf": vectorizer.idf,
        "weights": best_model.weights,
        "bias": best_model.bias,
        "train_samples": len(train_data),
        "test_samples": len(test_data),
        "macro_f1": evaluations["Logistic Regression"]["macro_f1"]
    }
    models_dir = os.path.join(base_dir, "models")
    os.makedirs(models_dir, exist_ok=True)
    with open(os.path.join(models_dir, "complaint_pipeline_metadata.json"), "w", encoding="utf-8") as f:
        json.dump(pipeline_artifact, f)

    return evaluations

if __name__ == "__main__":
    results = run_training_pipeline()
    print("Training Pipeline Complete. Metrics:")
    for m, d in results.items():
        print(f"  {m:25s} -> Acc: {d['accuracy']*100:.1f}%, Macro F1: {d['macro_f1']*100:.1f}%")
