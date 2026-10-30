"""
src/preprocess.py
Text cleaning, normalization, and deduplication module.
Preserves negation words and currency tokens critical for complaint triage.
"""

import os
import re
import csv
from typing import List, Dict, Any, Tuple

# Domain stop words: Negation tokens ('not', 'no', 'never', 'refused', 'failed') are deliberately excluded
STOPWORDS = set([
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "as", "at",
    "be", "because", "been", "before", "being", "below", "between", "both", "but", "by", "could", "did",
    "do", "does", "doing", "down", "during", "each", "few", "for", "from", "further", "had", "has", "have",
    "having", "he", "her", "here", "hers", "herself", "him", "himself", "his", "how", "i", "if", "in",
    "into", "is", "it", "its", "itself", "me", "more", "most", "my", "myself", "of", "off", "on", "once",
    "only", "or", "other", "ought", "our", "ours", "ourselves", "out", "over", "own", "same", "she",
    "should", "so", "some", "such", "than", "that", "the", "their", "theirs", "them", "themselves", "then",
    "there", "these", "they", "this", "those", "through", "to", "too", "under", "until", "up", "very",
    "was", "we", "were", "what", "when", "where", "which", "while", "who", "whom", "why", "with", "would",
    "you", "your", "yours", "yourself", "yourselves"
])

def clean_text(text: str) -> str:
    """Cleans and normalizes complaint text without removing critical domain entities."""
    if not isinstance(text, str):
        return ""
    # Strip HTML tags
    text = re.sub(r"<[^>]+>", " ", text)
    # Strip URLs
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    # Lowercase
    text = text.lower()
    # Normalize Indian currency notation Rs. / INR / ₹ to token 'rupees'
    text = re.sub(r"(?:rs\.?|inr|₹)\s*\d+(?:,\d+)*(?:\.\d+)?", " currency_amount ", text)
    # Normalize numeric tokens
    text = re.sub(r"\b\d+\b", " number_token ", text)
    # Remove excessive punctuation
    text = re.sub(r"[^\w\s]", " ", text)
    # Collapse multiple whitespaces
    text = re.sub(r"\s+", " ", text).strip()
    return text

def tokenize(text: str, remove_stopwords: bool = True) -> List[str]:
    """Tokenizes normalized text into words, filtering standard non-negation stop words."""
    words = text.split()
    if remove_stopwords:
        words = [w for w in words if w not in STOPWORDS and len(w) > 1]
    return words

def clean_and_audit_dataset(raw_records: List[Dict[str, Any]], min_char_length: int = 15) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Applies text normalization and deduplication to raw dataset.
    Returns cleaned records and an audit log of dropped or deduplicated rows.
    """
    cleaned_records = []
    audit_log = []
    seen_texts = set()

    for r in raw_records:
        cid = r.get("complaint_id", "UNKNOWN")
        text = r.get("complaint_text", "")
        cat = r.get("category", "")

        # Validation checks
        if not text or len(text.strip()) < min_char_length:
            audit_log.append({"complaint_id": cid, "issue": f"Text too short (<{min_char_length} chars)", "action": "DROPPED"})
            continue
        
        if not cat:
            audit_log.append({"complaint_id": cid, "issue": "Missing category label", "action": "DROPPED"})
            continue

        normalized = clean_text(text)
        if normalized in seen_texts:
            audit_log.append({"complaint_id": cid, "issue": "Exact duplicate complaint text", "action": "DEDUPLICATED"})
            continue

        seen_texts.add(normalized)
        words = normalized.split()
        cleaned_records.append({
            "complaint_id": cid,
            "cleaned_text": normalized,
            "original_text": text,
            "category": cat,
            "token_count": len(words),
            "date": r.get("date", ""),
            "state": r.get("state", "")
        })

    return cleaned_records, audit_log

def save_cleaned_dataset(cleaned_records: List[Dict[str, Any]], audit_log: List[Dict[str, Any]], base_dir: str = None):
    if base_dir is None:
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    processed_dir = os.path.join(base_dir, "data", "processed")
    os.makedirs(processed_dir, exist_ok=True)

    cleaned_csv = os.path.join(processed_dir, "cleaned_complaints.csv")
    with open(cleaned_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["complaint_id", "cleaned_text", "original_text", "category", "token_count", "date", "state"])
        writer.writeheader()
        writer.writerows(cleaned_records)

    audit_csv = os.path.join(processed_dir, "data_quality_audit.csv")
    with open(audit_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["complaint_id", "issue", "action"])
        writer.writeheader()
        writer.writerows(audit_log)

if __name__ == "__main__":
    from data_loader import load_raw_dataset
    raw = load_raw_dataset()
    cleaned, audit = clean_and_audit_dataset(raw)
    save_cleaned_dataset(cleaned, audit)
    print(f"Cleaned {len(cleaned)} records. Logged {len(audit)} audit entries.")
