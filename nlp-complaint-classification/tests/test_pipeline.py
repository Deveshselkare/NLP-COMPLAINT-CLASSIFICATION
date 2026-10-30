"""
tests/test_pipeline.py
Automated test suite using unittest and pytest compatibility.
Covers schema validation, text normalization, duplicate detection,
zero-leakage vectorizer fitting, metric calculations, and prediction guards.
"""

import os
import sys
import unittest

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.preprocess import clean_text, tokenize, clean_and_audit_dataset
from src.features import ComplaintTFIDFVectorizer
from src.evaluate import evaluate_classifier
from src.predict import ComplaintPredictor

class TestNLPComplaintPipeline(unittest.TestCase):

    def setUp(self):
        self.categories = [
            "Banking & UPI Services",
            "E-Commerce & Online Shopping",
            "Telecommunications & Internet",
            "Consumer Utilities & Energy",
            "Consumer Electronics & Warranty"
        ]
        self.sample_text = "SBI UPI transaction failed and Rs 5,000 deducted from account. Seller refused return."

    def test_text_cleaning_and_normalization(self):
        """Checks HTML stripping, lowercase, currency normalization, and negation preservation."""
        raw = "<b>URGENT:</b> Paid Rs 4500 to merchant via UPI but payment NOT received! <a href='http://fail.com'>link</a>"
        cleaned = clean_text(raw)
        
        self.assertNotIn("<b>", cleaned)
        self.assertNotIn("http", cleaned)
        self.assertIn("currency_amount", cleaned)
        self.assertIn("not", cleaned, "Negation token 'not' must be preserved in text normalization")

    def test_stopword_filtering_preserves_negations(self):
        """Verifies that negation terms are not dropped by tokenizer."""
        text = "money not received merchant refused refund transaction failed"
        tokens = tokenize(text, remove_stopwords=True)
        self.assertIn("not", tokens)
        self.assertIn("refused", tokens)
        self.assertIn("failed", tokens)

    def test_empty_and_short_text_filtering(self):
        """Verifies that complaints under 15 characters or empty are dropped in audit log."""
        raw_samples = [
            {"complaint_id": "T1", "complaint_text": "", "category": "Banking & UPI Services"},
            {"complaint_id": "T2", "complaint_text": "Help please", "category": "Banking & UPI Services"},
            {"complaint_id": "T3", "complaint_text": "Valid length complaint regarding duplicate debit on credit card.", "category": "Banking & UPI Services"}
        ]
        cleaned, audit = clean_and_audit_dataset(raw_samples, min_char_length=15)
        self.assertEqual(len(cleaned), 1)
        self.assertEqual(len(audit), 2)
        self.assertEqual(cleaned[0]["complaint_id"], "T3")

    def test_duplicate_complaint_detection(self):
        """Ensures identical complaint texts are deduplicated to prevent leakage."""
        text = "Identical text regarding fiber broadband downtime in residential society."
        raw_samples = [
            {"complaint_id": "D1", "complaint_text": text, "category": "Telecommunications & Internet"},
            {"complaint_id": "D2", "complaint_text": text, "category": "Telecommunications & Internet"}
        ]
        cleaned, audit = clean_and_audit_dataset(raw_samples)
        self.assertEqual(len(cleaned), 1)
        self.assertEqual(len(audit), 1)
        self.assertEqual(audit[0]["action"], "DEDUPLICATED")

    def test_tfidf_vectorizer_fit_transform_isolation(self):
        """Confirms that test documents do not alter the trained vocabulary."""
        train_docs = [
            "SBI UPI payment failed on merchant store",
            "HDFC credit card interest charge dispute"
        ]
        test_docs = [
            "Airtel broadband disconnected completely"
        ]
        vec = ComplaintTFIDFVectorizer(max_features=100, min_df=1)
        vec.fit(train_docs)
        vocab_size_before = len(vec.vocabulary)
        
        # Transform test document
        test_vec = vec.transform(test_docs)
        self.assertEqual(len(vec.vocabulary), vocab_size_before, "Transform must not alter fitted vocabulary")
        self.assertNotIn("airtel", vec.vocabulary, "Out-of-vocabulary word from test set must not leak into vocabulary")

    def test_metric_calculation_accuracy_and_f1(self):
        """Validates evaluation metrics with known ground truth and predictions."""
        y_true = ["Banking & UPI Services", "Banking & UPI Services", "E-Commerce & Online Shopping"]
        y_pred = ["Banking & UPI Services", "Telecommunications & Internet", "E-Commerce & Online Shopping"]
        metrics = evaluate_classifier(y_true, y_pred, self.categories)
        
        self.assertAlmostEqual(metrics["accuracy"], 2/3, places=2)
        self.assertTrue(0.0 <= metrics["macro_f1"] <= 1.0)
        self.assertTrue(0.0 <= metrics["weighted_f1"] <= 1.0)
        self.assertIn("Banking & UPI Services", metrics["confusion_matrix"])

    def test_predictor_valid_prediction_and_confidence(self):
        """Verifies end-to-end inference prediction, probability structure, and top category."""
        predictor = ComplaintPredictor()
        res = predictor.predict("Amount of Rs 12000 deducted twice from my SBI savings account via UPI payment.")
        
        self.assertEqual(res["status"], "success")
        self.assertEqual(res["predicted_category"], "Banking & UPI Services")
        self.assertGreater(res["confidence"], 0.70)
        self.assertEqual(len(res["probabilities"]), len(self.categories))
        self.assertAlmostEqual(sum(res["probabilities"].values()), 1.0, places=2)

    def test_predictor_empty_and_short_input_handling(self):
        """Ensures predictor returns informative error without throwing unhandled exceptions."""
        predictor = ComplaintPredictor()
        res = predictor.predict("")
        self.assertEqual(res["status"], "error")
        self.assertIsNone(res["predicted_category"])

        res_short = predictor.predict("Bad app")
        self.assertEqual(res_short["status"], "error")

    def test_predictor_out_of_vocabulary_handling(self):
        """Checks graceful handling of unknown nonsense text."""
        predictor = ComplaintPredictor()
        res = predictor.predict("xyzabc qwertyuiop asdfghjkl zxcvbnm 1234567890")
        self.assertIn(res["status"], ["uncertain", "success"])

if __name__ == "__main__":
    unittest.main()
