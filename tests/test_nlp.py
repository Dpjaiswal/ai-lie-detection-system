"""
NLP Tests — AI-Powered Lie Detection System
Unit tests for preprocessing, feature extraction, and models.
"""
from __future__ import annotations

import numpy as np
import pytest


# ──────────────────────────────────────────────────────────────
# TextCleaner Tests
# ──────────────────────────────────────────────────────────────

class TestTextCleaner:
    def setup_method(self):
        from src.nlp.preprocessor import TextCleaner
        self.cleaner = TextCleaner()

    def test_lowercase(self):
        assert self.cleaner.clean("HELLO WORLD") == "hello world"

    def test_remove_html(self):
        result = self.cleaner.clean("<p>Hello <b>World</b></p>")
        assert "<p>" not in result and "<b>" not in result

    def test_remove_url(self):
        result = self.cleaner.clean("Visit https://example.com for more info")
        assert "https://" not in result

    def test_clean_empty_string(self):
        assert self.cleaner.clean("") == ""

    def test_clean_none(self):
        assert self.cleaner.clean(None) == ""

    def test_whitespace_normalization(self):
        result = self.cleaner.clean("Hello    World   ")
        assert "  " not in result
        assert result == "hello world"

    def test_clean_batch(self):
        texts = ["Hello World", "<b>Bold</b>", "http://url.com text"]
        results = self.cleaner.clean_batch(texts)
        assert len(results) == 3
        assert all(isinstance(r, str) for r in results)


# ──────────────────────────────────────────────────────────────
# LinguisticFeatureExtractor Tests
# ──────────────────────────────────────────────────────────────

class TestLinguisticFeatureExtractor:
    def setup_method(self):
        from src.nlp.preprocessor import LinguisticFeatureExtractor
        self.extractor = LinguisticFeatureExtractor()

    def test_feature_shape(self):
        features = self.extractor.extract("I never did anything wrong in my life.")
        assert features.shape == (15,)

    def test_feature_dtype(self):
        features = self.extractor.extract("This is a test sentence.")
        assert features.dtype == np.float32

    def test_empty_text_returns_zeros(self):
        features = self.extractor.extract("")
        assert np.allclose(features, 0)

    def test_negation_detection(self):
        high_negation = self.extractor.extract("I never not did nothing wrong at all.")
        low_negation = self.extractor.extract("I walked to the store and bought milk.")
        # Negation ratio is feature index 3
        assert high_negation[3] > low_negation[3]

    def test_certainty_detection(self):
        certain = self.extractor.extract("I absolutely definitely certainly always do this.")
        uncertain = self.extractor.extract("I walked to the store.")
        assert certain[4] > uncertain[4]

    def test_batch_shape(self):
        texts = ["First sentence.", "Second sentence here.", "Third one."]
        batch_features = self.extractor.extract_batch(texts)
        assert batch_features.shape == (3, 15)

    def test_feature_names_count(self):
        from src.nlp.preprocessor import LinguisticFeatureExtractor
        assert len(LinguisticFeatureExtractor.feature_names()) == 15


# ──────────────────────────────────────────────────────────────
# TFIDFExtractor Tests
# ──────────────────────────────────────────────────────────────

class TestTFIDFExtractor:
    def setup_method(self):
        from src.nlp.feature_extractor import TFIDFExtractor
        self.extractor = TFIDFExtractor(max_features=1000)
        self.train_texts = [
            "I was at home the entire evening",
            "I never met that person in my life",
            "I went to the store and bought groceries",
            "The defendant was not present at the scene",
            "I work at a software company downtown",
        ]

    def test_fit_transform_shape(self):
        X = self.extractor.fit_transform(self.train_texts)
        assert X.shape[0] == len(self.train_texts)
        assert X.shape[1] <= 1000

    def test_transform_without_fit_raises(self):
        from src.nlp.feature_extractor import TFIDFExtractor
        unfitted = TFIDFExtractor()
        with pytest.raises(RuntimeError):
            unfitted.transform(["test text"])

    def test_transform_new_text(self):
        self.extractor.fit(self.train_texts)
        X_new = self.extractor.transform(["This is a new sentence."])
        assert X_new.shape[0] == 1

    def test_feature_names_available(self):
        self.extractor.fit(self.train_texts)
        names = self.extractor.get_feature_names()
        assert len(names) > 0
        assert all(isinstance(n, str) for n in names)

    def test_save_load(self, tmp_path):
        self.extractor.fit(self.train_texts)
        save_path = str(tmp_path / "tfidf.pkl")
        self.extractor.save(save_path)

        from src.nlp.feature_extractor import TFIDFExtractor
        loaded = TFIDFExtractor().load(save_path)
        X_orig = self.extractor.transform(self.train_texts)
        X_loaded = loaded.transform(self.train_texts)
        assert X_orig.shape == X_loaded.shape


# ──────────────────────────────────────────────────────────────
# Logistic Regression Classifier Tests
# ──────────────────────────────────────────────────────────────

class TestTFIDFLogisticRegression:
    def setup_method(self):
        from src.nlp.feature_extractor import TFIDFExtractor
        from src.nlp.baseline_model import TFIDFLogisticRegression

        self.texts = [
            "I was at home all night",
            "I never did that ever",
            "The weather was nice today",
            "I absolutely did not steal anything",
            "She went to the market yesterday",
            "I definitely have never met him before",
            "We had dinner at the restaurant",
            "I completely forgot about the meeting",
        ]
        self.labels = np.array([0, 1, 0, 1, 0, 1, 0, 1])

        self.tfidf = TFIDFExtractor(max_features=500)
        self.X = self.tfidf.fit_transform(self.texts)
        self.model = TFIDFLogisticRegression()

    def test_fit_does_not_raise(self):
        self.model.fit(self.X, self.labels)

    def test_predict_shape(self):
        self.model.fit(self.X, self.labels)
        preds = self.model.predict(self.X)
        assert len(preds) == len(self.labels)

    def test_predict_proba_shape(self):
        self.model.fit(self.X, self.labels)
        probs = self.model.predict_proba(self.X)
        assert probs.shape == (len(self.labels), 2)

    def test_predict_proba_sums_to_one(self):
        self.model.fit(self.X, self.labels)
        probs = self.model.predict_proba(self.X)
        assert np.allclose(probs.sum(axis=1), 1.0)

    def test_evaluate_returns_auc(self):
        self.model.fit(self.X, self.labels)
        metrics = self.model.evaluate(self.X, self.labels)
        assert "roc_auc" in metrics
        assert 0.0 <= metrics["roc_auc"] <= 1.0

    def test_save_load(self, tmp_path):
        self.model.fit(self.X, self.labels)
        save_path = str(tmp_path / "lr_model.pkl")
        self.model.save(save_path)

        from src.nlp.baseline_model import TFIDFLogisticRegression
        loaded = TFIDFLogisticRegression().load(save_path)
        preds_orig = self.model.predict(self.X)
        preds_loaded = loaded.predict(self.X)
        np.testing.assert_array_equal(preds_orig, preds_loaded)
