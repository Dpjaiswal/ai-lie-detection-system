"""
Classical Audio Models — AI-Powered Lie Detection System
Random Forest, XGBoost, and SVM classifiers on audio feature vectors.
"""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from src.utils.logger import logger
from src.utils.config import config
from src.utils.helpers import set_seed


class AudioRandomForest:
    """
    Random Forest on 163-dim audio feature vectors.

    Why Random Forest: Robust to outliers, handles non-linear relationships,
    naturally provides feature importance for XAI. Works well out-of-the-box
    with tabular audio features.

    Computational: CPU, ~1–2 min training on 5K samples.
    Expected accuracy: 65–72% on stress/emotion proxy tasks.
    """

    def __init__(self, seed: int = 42):
        set_seed(seed)
        cfg = config.get_audio_config("classical").get("random_forest", {})
        self.model = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(
                n_estimators=cfg.get("n_estimators", 200),
                max_depth=cfg.get("max_depth", 10),
                min_samples_split=cfg.get("min_samples_split", 5),
                class_weight=cfg.get("class_weight", "balanced"),
                random_state=seed,
                n_jobs=-1,
            )),
        ])
        self.is_fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> "AudioRandomForest":
        logger.info(f"Training Random Forest on features of shape {X.shape}")
        self.model.fit(X, y)
        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)

    def evaluate(self, X: np.ndarray, y: np.ndarray, feature_names: Optional[List[str]] = None) -> Dict:
        preds = self.predict(X)
        probas = self.predict_proba(X)[:, 1]
        report = classification_report(y, preds, output_dict=True, zero_division=0)
        auc = roc_auc_score(y, probas) if len(set(y)) > 1 else 0.5

        result = {"classification_report": report, "roc_auc": auc}

        rf = self.model.named_steps["clf"]
        importances = rf.feature_importances_
        if feature_names:
            sorted_idx = np.argsort(importances)[::-1]
            result["top_features"] = [
                {"feature": feature_names[i], "importance": float(importances[i])}
                for i in sorted_idx[:20]
            ]

        logger.info(f"Random Forest — AUC: {auc:.4f}, Accuracy: {report.get('accuracy', 0):.4f}")
        return result

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self.model, f)
        logger.info(f"Random Forest model saved to {path}")

    def load(self, path: str) -> "AudioRandomForest":
        with open(path, "rb") as f:
            self.model = pickle.load(f)
        self.is_fitted = True
        return self


class AudioXGBoost:
    """
    XGBoost on 163-dim audio feature vectors.

    Why XGBoost: Handles feature interactions better than RF, supports
    early stopping on validation set. Often best classical model for tabular data.

    Computational: CPU/GPU, ~2–5 min training.
    Expected improvement over RF: 1–3% F1.
    """

    def __init__(self, seed: int = 42):
        set_seed(seed)
        cfg = config.get_audio_config("classical").get("xgboost", {})
        try:
            import xgboost as xgb
            self.model = Pipeline([
                ("scaler", StandardScaler()),
                ("clf", xgb.XGBClassifier(
                    n_estimators=cfg.get("n_estimators", 200),
                    max_depth=cfg.get("max_depth", 6),
                    learning_rate=cfg.get("learning_rate", 0.05),
                    eval_metric="logloss",
                    random_state=seed,
                    use_label_encoder=False,
                    n_jobs=-1,
                )),
            ])
        except ImportError:
            raise ImportError("xgboost not installed.")
        self.is_fitted = False

    def fit(
        self, X_train: np.ndarray, y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None,
    ) -> "AudioXGBoost":
        logger.info(f"Training XGBoost on audio features, shape={X_train.shape}")
        X_scaled = self.model.named_steps["scaler"].fit_transform(X_train)
        eval_set = [(self.model.named_steps["scaler"].transform(X_val), y_val)] if X_val is not None else None
        self.model.named_steps["clf"].fit(X_scaled, y_train, eval_set=eval_set, verbose=False)
        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict:
        preds = self.predict(X)
        probas = self.predict_proba(X)[:, 1]
        return {
            "classification_report": classification_report(y, preds, output_dict=True, zero_division=0),
            "roc_auc": roc_auc_score(y, probas) if len(set(y)) > 1 else 0.5,
        }

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self.model, f)

    def load(self, path: str) -> "AudioXGBoost":
        with open(path, "rb") as f:
            self.model = pickle.load(f)
        self.is_fitted = True
        return self


class AudioSVM:
    """
    SVM with RBF kernel on audio features.

    Why SVM: Good for high-dimensional, smaller datasets.
    RBF kernel captures non-linear boundaries in feature space.
    Works well when features are properly scaled.

    Computational: CPU only, slow on >10K samples. Use XGBoost for large data.
    Expected accuracy: 66–70%.
    """

    def __init__(self, seed: int = 42):
        set_seed(seed)
        cfg = config.get_audio_config("classical").get("svm", {})
        self.model = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(
                kernel=cfg.get("kernel", "rbf"),
                C=cfg.get("C", 10.0),
                gamma=cfg.get("gamma", "scale"),
                probability=cfg.get("probability", True),
                class_weight="balanced",
                random_state=seed,
            )),
        ])
        self.is_fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> "AudioSVM":
        logger.info(f"Training SVM on audio features, shape={X.shape}")
        self.model.fit(X, y)
        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict:
        preds = self.predict(X)
        probas = self.predict_proba(X)[:, 1]
        return {
            "classification_report": classification_report(y, preds, output_dict=True, zero_division=0),
            "roc_auc": roc_auc_score(y, probas) if len(set(y)) > 1 else 0.5,
        }

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self.model, f)

    def load(self, path: str) -> "AudioSVM":
        with open(path, "rb") as f:
            self.model = pickle.load(f)
        self.is_fitted = True
        return self
