"""
src/evaluate.py
Model evaluation metrics module.
Calculates Accuracy, Macro F1, Weighted F1, Precision, Recall, and Confusion Matrix.
"""

from typing import List, Dict, Any

def evaluate_classifier(y_true: List[str], y_pred: List[str], classes: List[str]) -> Dict[str, Any]:
    """
    Computes rigorous classification performance metrics across all target classes.
    """
    total = len(y_true)
    if total == 0:
        return {"accuracy": 0.0, "macro_f1": 0.0, "weighted_f1": 0.0}

    correct = sum(1 for yt, yp in zip(y_true, y_pred) if yt == yp)
    accuracy = correct / total

    class_metrics = {}
    macro_p = 0.0
    macro_r = 0.0
    macro_f1 = 0.0
    weighted_f1 = 0.0

    for c in classes:
        tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == c and yp == c)
        fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt != c and yp == c)
        fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == c and yp != c)
        support = sum(1 for yt in y_true if yt == c)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        class_metrics[c] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1_score": round(f1, 4),
            "support": support
        }
        macro_p += precision
        macro_r += recall
        macro_f1 += f1
        weighted_f1 += f1 * (support / total)

    k = len(classes)
    macro_p /= k
    macro_r /= k
    macro_f1 /= k

    # Build Confusion Matrix mapping
    cm = {true_c: {pred_c: 0 for pred_c in classes} for true_c in classes}
    for yt, yp in zip(y_true, y_pred):
        if yt in cm and yp in cm[yt]:
            cm[yt][yp] += 1

    return {
        "accuracy": round(accuracy, 4),
        "macro_precision": round(macro_p, 4),
        "macro_recall": round(macro_r, 4),
        "macro_f1": round(macro_f1, 4),
        "weighted_f1": round(weighted_f1, 4),
        "class_metrics": class_metrics,
        "confusion_matrix": cm
    }
