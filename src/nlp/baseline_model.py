"""
Baseline NLP Models — AI-Powered Lie Detection System
TF-IDF + Logistic Regression and Word2Vec + XGBoost classifiers.
"""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from src.utils.logger import logger
from src.utils.config import config
from src.utils.helpers import set_seed


# ──────────────────────────────────────────────────────────────
# Logistic Regression Classifier (TF-IDF baseline)
# ──────────────────────────────────────────────────────────────

class TFIDFLogisticRegression:
    """
    TF-IDF + Logistic Regression baseline.

    Why: Fast, interpretable, and provides a reliable lower bound.
    TF-IDF captures vocabulary-level deception signals.
    LR produces well-calibrated probabilities.

    Computational requirements: CPU-only, trains in seconds.
    Expected accuracy on LIAR (binary): ~63–66%
    """

    def __init__(self, seed: int = 42):
        set_seed(seed)
        cfg = config.get_nlp_config("baseline")
        lr_cfg = cfg.get("logistic_regression", {})
        self.model = LogisticRegression(
            C=lr_cfg.get("C", 1.0),
            max_iter=lr_cfg.get("max_iter", 1000),
            solver=lr_cfg.get("solver", "lbfgs"),
            class_weight=lr_cfg.get("class_weight", "balanced"),
            random_state=seed,
            multi_class="ovr",
        )
        self.is_fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> "TFIDFLogisticRegression":
        """Train on TF-IDF features."""
        logger.info("Training TF-IDF + Logistic Regression...")
        self.model.fit(X, y)
        self.is_fitted = True
        train_acc = self.model.score(X, y)
        logger.info(f"Training accuracy: {train_acc:.4f}")
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict:
        preds = self.predict(X)
        probas = self.predict_proba(X)[:, 1]
        report = classification_report(y, preds, output_dict=True)
        auc = roc_auc_score(y, probas)
        logger.info(f"Evaluation — AUC: {auc:.4f}")
        return {"classification_report": report, "roc_auc": auc}

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self.model, f)
        logger.info(f"Model saved to {path}")

    def load(self, path: str) -> "TFIDFLogisticRegression":
        with open(path, "rb") as f:
            self.model = pickle.load(f)
        self.is_fitted = True
        return self


# ──────────────────────────────────────────────────────────────
# XGBoost Classifier
# ──────────────────────────────────────────────────────────────

class XGBoostTextClassifier:
    """
    XGBoost text classifier usable with TF-IDF, Word2Vec, or combined features.

    Why XGBoost: Gradient boosting captures non-linear feature interactions
    that LR misses. Works well with both sparse TF-IDF and dense embeddings.

    Computational requirements: CPU-only (or GPU with tree_method=gpu_hist).
    Expected improvement over LR: ~2–5% F1.
    """

    def __init__(self, seed: int = 42):
        set_seed(seed)
        cfg = config.get_nlp_config("baseline").get("xgboost", {})
        try:
            import xgboost as xgb
            self.model = xgb.XGBClassifier(
                n_estimators=cfg.get("n_estimators", 300),
                max_depth=cfg.get("max_depth", 6),
                learning_rate=cfg.get("learning_rate", 0.05),
                subsample=cfg.get("subsample", 0.8),
                colsample_bytree=cfg.get("colsample_bytree", 0.8),
                eval_metric=cfg.get("eval_metric", "logloss"),
                random_state=seed,
                use_label_encoder=False,
            )
        except ImportError:
            raise ImportError("xgboost not installed. Run: pip install xgboost")
        self.is_fitted = False

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> "XGBoostTextClassifier":
        """Train with optional early stopping on validation set."""
        eval_set = [(X_val, y_val)] if X_val is not None else None
        logger.info("Training XGBoost classifier...")
        self.model.fit(
            X_train, y_train,
            eval_set=eval_set,
            verbose=50,
        )
        self.is_fitted = True
        logger.info("XGBoost training complete.")
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)

    def get_feature_importance(self, feature_names: Optional[List[str]] = None) -> Dict:
        """Return feature importance scores."""
        importances = self.model.feature_importances_
        if feature_names:
            return dict(sorted(
                zip(feature_names, importances),
                key=lambda x: x[1], reverse=True
            ))
        return {"importances": importances.tolist()}

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict:
        preds = self.predict(X)
        probas = self.predict_proba(X)[:, 1]
        report = classification_report(y, preds, output_dict=True)
        auc = roc_auc_score(y, probas)
        return {"classification_report": report, "roc_auc": auc}

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.model.save_model(path)

    def load(self, path: str) -> "XGBoostTextClassifier":
        self.model.load_model(path)
        self.is_fitted = True
        return self


# ──────────────────────────────────────────────────────────────
# SVM Classifier
# ──────────────────────────────────────────────────────────────

class SVMTextClassifier:
    """
    SVM text classifier with RBF kernel.
    Works well with dense embeddings (Word2Vec, transformer).
    """

    def __init__(self, C: float = 10.0, kernel: str = "rbf", seed: int = 42):
        set_seed(seed)
        self.scaler = StandardScaler()
        self.model = SVC(
            C=C, kernel=kernel, probability=True,
            class_weight="balanced", random_state=seed
        )
        self.is_fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SVMTextClassifier":
        logger.info("Training SVM classifier...")
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(self.scaler.transform(X))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(self.scaler.transform(X))

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict:
        preds = self.predict(X)
        probas = self.predict_proba(X)[:, 1]
        return {
            "classification_report": classification_report(y, preds, output_dict=True),
            "roc_auc": roc_auc_score(y, probas),
        }

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"model": self.model, "scaler": self.scaler}, f)

    def load(self, path: str) -> "SVMTextClassifier":
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.model = data["model"]
        self.scaler = data["scaler"]
        self.is_fitted = True
        return self
