"""
SHAP Explainer — AI-Powered Lie Detection System
SHAP-based explanations for text and audio models.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

from src.utils.logger import logger


class TextSHAPExplainer:
    """
    SHAP explainer for text-based deception models.
    Uses KernelSHAP for model-agnostic explanation or
    DeepSHAP for PyTorch transformers.

    Explains:
    - Which words push prediction toward 'lie'
    - Which words push prediction toward 'truth'
    - Global feature importance across dataset
    """

    def __init__(
        self,
        model,  # sklearn classifier or PyTorch model
        feature_names: Optional[List[str]] = None,
        model_type: str = "sklearn",  # sklearn | pytorch
        background_size: int = 100,
    ):
        self.model = model
        self.feature_names = feature_names
        self.model_type = model_type
        self.background_size = background_size
        self.explainer = None

    def build_explainer(self, X_background: np.ndarray) -> "TextSHAPExplainer":
        """Build SHAP explainer with background data."""
        try:
            import shap
        except ImportError:
            raise ImportError("shap not installed. Run: pip install shap")

        # Sample background
        idx = np.random.choice(len(X_background), min(self.background_size, len(X_background)), replace=False)
        background = X_background[idx]

        if self.model_type == "sklearn":
            self.explainer = shap.KernelExplainer(
                self.model.predict_proba, background
            )
        else:
            self.explainer = shap.KernelExplainer(
                lambda x: self._torch_predict(x), background
            )
        logger.info("SHAP KernelExplainer built.")
        return self

    def _torch_predict(self, X: np.ndarray) -> np.ndarray:
        """Wrapper for PyTorch model predictions."""
        import torch
        self.model.eval()
        with torch.no_grad():
            tensor = torch.tensor(X, dtype=torch.float32)
            logits = self.model(tensor)
            return torch.softmax(logits, dim=1).numpy()

    def explain(self, X: np.ndarray, n_samples: int = 50) -> np.ndarray:
        """
        Compute SHAP values for samples.

        Returns:
            shap_values: List of arrays, one per class, each shape (N, features)
        """
        if self.explainer is None:
            raise RuntimeError("Call build_explainer() first.")
        import shap
        shap_values = self.explainer.shap_values(X, nsamples=n_samples)
        return shap_values

    def explain_instance(self, x: np.ndarray, n_samples: int = 100) -> Dict:
        """
        Explain a single instance.

        Returns:
            Dict with lie_shap_values and top contributing features
        """
        shap_values = self.explain(x.reshape(1, -1), n_samples=n_samples)
        lie_shap = shap_values[1][0] if isinstance(shap_values, list) else shap_values[0]

        result = {"shap_values": lie_shap.tolist()}

        if self.feature_names and len(lie_shap) == len(self.feature_names):
            # Top positive (lie) and negative (truth) features
            sorted_idx = np.argsort(np.abs(lie_shap))[::-1]
            result["top_features"] = [
                {
                    "feature": self.feature_names[i],
                    "shap_value": float(lie_shap[i]),
                    "direction": "lie" if lie_shap[i] > 0 else "truth",
                }
                for i in sorted_idx[:10]
            ]

        return result

    def plot_summary(
        self,
        shap_values: np.ndarray,
        X: np.ndarray,
        save_path: str = "reports/shap_summary.png",
        max_display: int = 20,
    ) -> None:
        """Generate and save SHAP summary plot."""
        try:
            import shap
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.figure(figsize=(12, 8))
            shap.summary_plot(
                shap_values[1] if isinstance(shap_values, list) else shap_values,
                X,
                feature_names=self.feature_names,
                max_display=max_display,
                show=False,
            )
            plt.tight_layout()
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            plt.close()
            logger.info(f"SHAP summary plot saved to {save_path}")
        except Exception as e:
            logger.warning(f"SHAP plot failed: {e}")

    def plot_waterfall(
        self,
        shap_values: np.ndarray,
        instance_idx: int = 0,
        save_path: str = "reports/shap_waterfall.png",
    ) -> None:
        """Generate SHAP waterfall plot for a single instance."""
        try:
            import shap
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.figure(figsize=(10, 6))
            sv = shap_values[1][instance_idx] if isinstance(shap_values, list) else shap_values[instance_idx]
            shap.waterfall_plot(
                shap.Explanation(values=sv, feature_names=self.feature_names),
                show=False,
            )
            plt.tight_layout()
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            plt.close()
        except Exception as e:
            logger.warning(f"Waterfall plot failed: {e}")


class AudioSHAPExplainer:
    """
    SHAP explainer for audio classical ML models.
    Explains which audio features (MFCC, pitch, jitter...) drive predictions.
    """

    def __init__(self, model, feature_names: Optional[List[str]] = None):
        self.model = model
        self.feature_names = feature_names
        self.explainer = None

    def build_tree_explainer(self, X_train: np.ndarray) -> "AudioSHAPExplainer":
        """Use TreeSHAP for tree-based models (RF, XGBoost)."""
        try:
            import shap
            clf = self.model.named_steps.get("clf", self.model)
            self.explainer = shap.TreeExplainer(clf)
            logger.info("TreeSHAP explainer built.")
        except Exception as e:
            logger.warning(f"TreeSHAP failed, falling back to KernelSHAP: {e}")
            self.explainer = None
        return self

    def explain_instance(self, x: np.ndarray) -> Dict:
        """Explain a single audio feature vector."""
        if self.explainer is None:
            return {"error": "Explainer not built"}

        import shap
        # Scale features if pipeline has scaler
        if hasattr(self.model, "named_steps"):
            scaler = self.model.named_steps.get("scaler")
            if scaler:
                x = scaler.transform(x.reshape(1, -1))[0]

        shap_values = self.explainer.shap_values(x.reshape(1, -1))
        lie_shap = shap_values[1][0] if isinstance(shap_values, list) else shap_values[0]

        result = {"shap_values": lie_shap.tolist()}
        if self.feature_names:
            sorted_idx = np.argsort(np.abs(lie_shap))[::-1]
            result["top_audio_features"] = [
                {
                    "feature": self.feature_names[i],
                    "shap_value": float(lie_shap[i]),
                    "direction": "stress_detected" if lie_shap[i] > 0 else "calm",
                }
                for i in sorted_idx[:10]
            ]
        return result

    def plot_feature_importance(
        self,
        X_test: np.ndarray,
        save_path: str = "reports/audio_shap_importance.png",
        top_k: int = 20,
    ) -> None:
        """Plot mean absolute SHAP values for audio features."""
        if self.explainer is None:
            return

        import shap
        shap_values = self.explainer.shap_values(X_test)
        mean_abs = np.abs(
            shap_values[1] if isinstance(shap_values, list) else shap_values
        ).mean(axis=0)

        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        indices = np.argsort(mean_abs)[::-1][:top_k]
        labels = [self.feature_names[i] for i in indices] if self.feature_names else [str(i) for i in indices]

        plt.figure(figsize=(12, 6))
        plt.barh(range(len(indices)), mean_abs[indices], color="#6C63FF")
        plt.yticks(range(len(indices)), labels)
        plt.xlabel("Mean |SHAP Value|")
        plt.title("Audio Feature Importance (SHAP)")
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"Audio SHAP importance plot saved to {save_path}")
