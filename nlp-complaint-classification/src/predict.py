"""
src/predict.py
Inference module for Indian Customer Complaint Classification.
Loads serialized pipeline, cleans input text, generates calibrated predictions,
and provides confidence intervals and top contributing keyword explanations.
"""

import os
import sys
import json
import math
import argparse
from typing import Dict, Any, Tuple, List

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.preprocess import clean_text, tokenize
from src.features import get_ngrams

DEFAULT_MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "complaint_pipeline_metadata.json")

class ComplaintPredictor:
    def __init__(self, model_path: str = DEFAULT_MODEL_PATH):
        self.model_path = model_path
        self.is_loaded = False
        self.model_name = ""
        self.classes: List[str] = []
        self.vocabulary: Dict[str, int] = {}
        self.idf: Dict[str, float] = {}
        self.weights: Dict[str, List[float]] = {}
        self.bias: Dict[str, float] = {}
        self.load_model()

    def load_model(self):
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"Model artifact not found at: {self.model_path}. "
                "Please run `python3 src/train.py` or `python3 scripts/generate_dataset_and_artifacts.py` first."
            )
        with open(self.model_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.model_name = data.get("model_name", "Logistic Regression")
        self.classes = data.get("classes", [])
        self.vocabulary = data.get("vocabulary", {})
        self.idf = data.get("idf", {})
        self.weights = data.get("weights", {})
        self.bias = data.get("bias", {})
        self.is_loaded = True

    def transform_input(self, text: str) -> Tuple[Dict[int, float], List[str]]:
        cleaned = clean_text(text)
        words = tokenize(cleaned)
        tokens = list(words) + get_ngrams(words, 2)

        from collections import Counter
        tf = Counter(tokens)
        sparse_vec = {}
        norm_sq = 0.0
        matched_tokens = []

        for t, count in tf.items():
            if t in self.vocabulary:
                matched_tokens.append(t)
                idx = self.vocabulary[t]
                sub_tf = 1.0 + math.log(count)
                weight = sub_tf * self.idf.get(t, 1.0)
                sparse_vec[idx] = weight
                norm_sq += weight * weight

        if norm_sq > 0:
            norm = math.sqrt(norm_sq)
            for idx in sparse_vec:
                sparse_vec[idx] /= norm

        return sparse_vec, matched_tokens

    def predict(self, text: str) -> Dict[str, Any]:
        """
        Classifies a single complaint string.
        Validates input length and handles out-of-vocabulary gracefully.
        """
        if not isinstance(text, str) or len(text.strip()) < 10:
            return {
                "status": "error",
                "message": "Input complaint text too short or empty (minimum 10 characters required).",
                "predicted_category": None,
                "confidence": 0.0,
                "probabilities": {}
            }

        vec, matched_tokens = self.transform_input(text)

        if not vec:
            return {
                "status": "uncertain",
                "message": "Complaint contains words not recognized in domain vocabulary. Flagged for human review.",
                "predicted_category": "Requires Manual Review",
                "confidence": 0.20,
                "probabilities": {c: 0.20 for c in self.classes},
                "matched_tokens": []
            }

        logits = {}
        for c in self.classes:
            s = self.bias.get(c, 0.0)
            w = self.weights.get(c, [])
            for idx, val in vec.items():
                if idx < len(w):
                    s += w[idx] * val
            logits[c] = s

        max_l = max(logits.values())
        exps = {c: math.exp(logits[c] - max_l) for c in self.classes}
        sum_e = sum(exps.values())
        probs = {c: round(exps[c] / sum_e, 4) for c in self.classes}

        predicted_category = max(probs, key=probs.get)
        confidence = probs[predicted_category]

        # Top contributing features for predicted class
        top_features = []
        pred_weights = self.weights.get(predicted_category, [])
        for t in matched_tokens:
            idx = self.vocabulary[t]
            if idx < len(pred_weights):
                top_features.append({"token": t, "importance": round(pred_weights[idx] * vec[idx], 4)})
        top_features.sort(key=lambda x: -x["importance"])

        return {
            "status": "success",
            "model_used": self.model_name,
            "predicted_category": predicted_category,
            "confidence": confidence,
            "probabilities": probs,
            "top_salient_terms": [f["token"] for f in top_features[:5]],
            "raw_text_length": len(text),
            "cleaned_token_count": len(tokenize(clean_text(text)))
        }

def main():
    parser = argparse.ArgumentParser(description="Classify Indian Consumer Complaint Text")
    parser.add_argument("--text", type=str, help="Complaint text to classify")
    args = parser.parse_args()

    predictor = ComplaintPredictor()
    sample_text = args.text or "Amount debited twice from my SBI account via Google Pay UPI but transaction failed on merchant store."
    
    print(f"\nAnalyzing Complaint:\n\"{sample_text}\"\n")
    result = predictor.predict(sample_text)
    print(f"Status:             {result['status']}")
    print(f"Predicted Category: {result['predicted_category']}")
    print(f"Confidence Score:   {result['confidence']*100:.2f}%")
    print(f"Model Architecture: {result.get('model_used')}")
    print("\nClass Probability Distribution:")
    for cat, prob in result.get('probabilities', {}).items():
        bar = "█" * int(prob * 30)
        print(f"  {cat:32s} : {prob*100:5.1f}% {bar}")
    print(f"\nKey Salient Terms:  {', '.join(result.get('top_salient_terms', []))}")

if __name__ == "__main__":
    main()
