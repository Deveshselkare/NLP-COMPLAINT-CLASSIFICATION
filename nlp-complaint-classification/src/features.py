"""
src/features.py
Feature engineering module implementing Sublinear TF-IDF Vectorization
Supports unigrams and bigrams with strict fit-on-train-only data leakage prevention.
"""

import math
from collections import Counter
from typing import List, Dict, Tuple, Set
try:
    from src.preprocess import clean_text, tokenize
except ImportError:
    from preprocess import clean_text, tokenize

def get_ngrams(words: List[str], n: int = 2) -> List[str]:
    """Extracts adjacent n-grams from a token list."""
    return ["_".join(words[i:i+n]) for i in range(len(words)-n+1)]

class ComplaintTFIDFVectorizer:
    """
    Sublinear TF-IDF Vectorizer with L2 normalization and n-gram capability.
    Operates identically in pure Python or Scikit-learn environments.
    """
    def __init__(self, max_features: int = 3500, min_df: int = 2, ngram_range: Tuple[int, int] = (1, 2)):
        self.max_features = max_features
        self.min_df = min_df
        self.ngram_range = ngram_range
        self.vocabulary: Dict[str, int] = {}
        self.idf: Dict[str, float] = {}
        self.feature_names: List[str] = []
        self.is_fitted: bool = False

    def extract_document_tokens(self, text: str) -> List[str]:
        words = tokenize(clean_text(text))
        tokens = list(words)
        if self.ngram_range[1] >= 2:
            tokens.extend(get_ngrams(words, 2))
        return tokens

    def fit(self, documents: List[str]):
        """Fits vocabulary and IDF weights exclusively on training documents."""
        doc_count = len(documents)
        df_counts = Counter()

        for doc in documents:
            tokens = set(self.extract_document_tokens(doc))
            for t in tokens:
                df_counts[t] += 1

        # Filter by min_df
        valid_tokens = [t for t, count in df_counts.items() if count >= self.min_df]
        # Sort by frequency descending, then alphabetically
        valid_tokens.sort(key=lambda t: (-df_counts[t], t))
        selected_tokens = valid_tokens[:self.max_features]

        self.vocabulary = {t: idx for idx, t in enumerate(selected_tokens)}
        self.feature_names = selected_tokens

        # Smooth IDF formulation: ln((1 + N) / (1 + df)) + 1.0
        for t in selected_tokens:
            df = df_counts[t]
            self.idf[t] = math.log((1 + doc_count) / (1 + df)) + 1.0

        self.is_fitted = True
        return self

    def transform_single(self, text: str) -> Dict[int, float]:
        """Transforms a single text string into a normalized sparse feature vector."""
        if not self.is_fitted:
            raise RuntimeError("Vectorizer must be fitted on training data before transforming new documents.")

        tokens = self.extract_document_tokens(text)
        tf = Counter(tokens)
        sparse_vec = {}
        norm_sq = 0.0

        for t, count in tf.items():
            if t in self.vocabulary:
                idx = self.vocabulary[t]
                # Sublinear term frequency: 1 + ln(count)
                sub_tf = 1.0 + math.log(count)
                weight = sub_tf * self.idf[t]
                sparse_vec[idx] = weight
                norm_sq += weight * weight

        # Apply L2 Euclidean normalization
        if norm_sq > 0:
            norm = math.sqrt(norm_sq)
            for idx in sparse_vec:
                sparse_vec[idx] /= norm

        return sparse_vec

    def transform(self, documents: List[str]) -> List[Dict[int, float]]:
        """Transforms a collection of documents using the fitted vocabulary."""
        return [self.transform_single(doc) for doc in documents]

    def fit_transform(self, documents: List[str]) -> List[Dict[int, float]]:
        """Fits on training documents and returns their sparse vectors."""
        self.fit(documents)
        return self.transform(documents)

if __name__ == "__main__":
    docs = [
        "SBI UPI transaction failed and money deducted from account",
        "Flipkart delivery delayed and return refund rejected",
        "Airtel fiber broadband down for 3 days with no internet"
    ]
    vec = ComplaintTFIDFVectorizer(max_features=50, min_df=1)
    sparse_docs = vec.fit_transform(docs)
    print(f"Fitted vocabulary with {len(vec.vocabulary)} features.")
    print("Sample document 0 features:", sparse_docs[0])
