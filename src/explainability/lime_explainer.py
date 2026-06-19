"""
LIME Explainer — AI-Powered Lie Detection System
LIME-based local interpretable explanations for text and tabular models.
"""
from __future__ import annotations

from typing import Callable, Dict, List, Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

from src.utils.logger import logger


class TextLIMEExplainer:
    """
    LIME text explainer for deception detection models.

    LIME perturbs the input text (by removing words) and trains a local
    linear model to approximate the black-box model's behavior.

    Explains: Which words contribute most to the 'lie' or 'truth' prediction.
    """

    def __init__(
        self,
        predict_fn: Callable,
        class_names: List[str] = ("truth", "lie"),
        num_features: int = 15,
        num_samples: int = 500,
    ):
        """
        Args:
            predict_fn: Function that takes List[str] and returns (N, num_classes) probabilities
            class_names: Names for the output classes
            num_features: Number of features to show in explanation
            num_samples: Number of perturbations for local approximation
        """
        self.predict_fn = predict_fn
        self.class_names = class_names
        self.num_features = num_features
        self.num_samples = num_samples
        self.explainer = None

    def _build_explainer(self) -> None:
        try:
            from lime.lime_text import LimeTextExplainer
            self.explainer = LimeTextExplainer(class_names=self.class_names)
        except ImportError:
            raise ImportError("lime not installed. Run: pip install lime")

    def explain(self, text: str, label_index: int = 1) -> Dict:
        """
        Explain a single text prediction.

        Args:
            text: Input text to explain
            label_index: Class index to explain (1 = lie)

        Returns:
            Dict with word importances and explanation metadata
        """
        if self.explainer is None:
            self._build_explainer()

        exp = self.explainer.explain_instance(
            text,
            self.predict_fn,
            num_features=self.num_features,
            num_samples=self.num_samples,
            labels=[label_index],
        )

        word_weights = exp.as_list(label=label_index)

        return {
            "text": text,
            "predicted_label": self.class_names[label_index],
            "word_importances": [
                {
                    "word": word,
                    "weight": float(weight),
                    "direction": "lie" if weight > 0 else "truth",
                }
                for word, weight in word_weights
            ],
            "top_lie_words": [w for w, s in word_weights if s > 0][:5],
            "top_truth_words": [w for w, s in word_weights if s < 0][:5],
        }

    def plot_explanation(
        self,
        text: str,
        label_index: int = 1,
        save_path: str = "reports/lime_explanation.html",
    ) -> str:
        """Generate and save LIME explanation as HTML."""
        if self.explainer is None:
            self._build_explainer()

        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        exp = self.explainer.explain_instance(
            text, self.predict_fn,
            num_features=self.num_features,
            num_samples=self.num_samples,
            labels=[label_index],
        )
        exp.save_to_file(save_path)
        logger.info(f"LIME explanation saved to {save_path}")
        return save_path

    def plot_word_importances(
        self,
        word_importances: List[Dict],
        save_path: str = "reports/lime_words.png",
        title: str = "LIME Word Importances",
    ) -> None:
        """Plot word importance bar chart."""
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        words = [w["word"] for w in word_importances[:10]]
        weights = [w["weight"] for w in word_importances[:10]]
        colors = ["#FF6B6B" if w > 0 else "#51CF66" for w in weights]

        plt.figure(figsize=(10, 5))
        bars = plt.barh(words, weights, color=colors)
        plt.axvline(x=0, color="black", linewidth=0.8)
        plt.xlabel("LIME Weight (positive → lie, negative → truth)")
        plt.title(title)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"LIME word importance plot saved to {save_path}")


class TabularLIMEExplainer:
    """
    LIME tabular explainer for audio feature vectors.
    Explains which audio features (pitch, MFCC, etc.) drive stress predictions.
    """

    def __init__(
        self,
        predict_fn: Callable,
        feature_names: Optional[List[str]] = None,
        class_names: List[str] = ("calm", "stress_detected"),
        num_features: int = 15,
        num_samples: int = 300,
    ):
        self.predict_fn = predict_fn
        self.feature_names = feature_names
        self.class_names = class_names
        self.num_features = num_features
        self.num_samples = num_samples
        self.explainer = None

    def _build_explainer(self, X_train: np.ndarray) -> None:
        try:
            from lime.lime_tabular import LimeTabularExplainer
            self.explainer = LimeTabularExplainer(
                training_data=X_train,
                feature_names=self.feature_names,
                class_names=self.class_names,
                mode="classification",
                discretize_continuous=True,
            )
        except ImportError:
            raise ImportError("lime not installed. Run: pip install lime")

    def explain(self, x: np.ndarray, X_train: np.ndarray, label_index: int = 1) -> Dict:
        """Explain a single audio feature vector prediction."""
        if self.explainer is None:
            self._build_explainer(X_train)

        exp = self.explainer.explain_instance(
            x, self.predict_fn,
            num_features=self.num_features,
            num_samples=self.num_samples,
            labels=[label_index],
        )

        feature_weights = exp.as_list(label=label_index)
        return {
            "audio_feature_importances": [
                {
                    "feature": feat,
                    "weight": float(weight),
                    "direction": "stress" if weight > 0 else "calm",
                }
                for feat, weight in feature_weights
            ],
            "predicted_class": self.class_names[label_index],
        }
