"""
src/data_loader.py
Loads and validates the Indian Consumer Grievance raw dataset.
Verifies column schema, non-empty records, and class representation.
"""

import os
import csv
from typing import List, Dict, Any, Tuple

REQUIRED_COLUMNS = ["complaint_id", "complaint_text", "category", "source_portal", "date", "state"]

EXPECTED_CATEGORIES = [
    "Banking & UPI Services",
    "E-Commerce & Online Shopping",
    "Telecommunications & Internet",
    "Consumer Utilities & Energy",
    "Consumer Electronics & Warranty"
]

def get_default_data_path() -> str:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(current_dir, "..", "data", "raw", "indian_consumer_complaints_raw.csv"))

def load_raw_dataset(file_path: str = None) -> List[Dict[str, Any]]:
    """Loads raw CSV dataset and validates schema presence."""
    if file_path is None:
        file_path = get_default_data_path()

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Raw dataset file not found at: {file_path}. Run generate_dataset_and_artifacts.py first.")

    records = []
    with open(file_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        header = reader.fieldnames or []
        for col in REQUIRED_COLUMNS:
            if col not in header:
                raise ValueError(f"Schema violation: required column '{col}' is missing from header {header}")
        
        for row in reader:
            records.append(row)

    return records

def validate_dataset_summary(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Computes schema audit and counts per category."""
    total_records = len(records)
    category_counts = {}
    missing_text = 0
    missing_category = 0

    for r in records:
        cat = r.get("category", "")
        text = r.get("complaint_text", "")
        if not text or len(text.strip()) == 0:
            missing_text += 1
        if not cat:
            missing_category += 1
        else:
            category_counts[cat] = category_counts.get(cat, 0) + 1

    return {
        "total_records": total_records,
        "missing_text_count": missing_text,
        "missing_category_count": missing_category,
        "category_distribution": category_counts,
        "categories_found": list(category_counts.keys())
    }

if __name__ == "__main__":
    data = load_raw_dataset()
    summary = validate_dataset_summary(data)
    print("Dataset Validation Summary:")
    print(f"Total Rows: {summary['total_records']}")
    print(f"Missing Texts: {summary['missing_text_count']}")
    print("Class Distribution:")
    for cat, count in summary['category_distribution'].items():
        print(f"  - {cat}: {count} records")
