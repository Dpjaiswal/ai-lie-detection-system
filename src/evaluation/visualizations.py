"""
Evaluation Visualizations — AI-Powered Lie Detection System
Confusion Matrix, ROC Curve, PR Curve, and Model Comparison charts.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from src.utils.logger import logger

# ── Consistent styling ────────────────────────────────────────
PALETTE = {
    "primary": "#6C63FF",
    "secondary": "#FF6584",
    "success": "#43D8A0",
    "warning": "#FFD93D",
    "background": "#1A1A2E",
    "surface": "#16213E",
    "text": "#E0E0E0",
}

plt.rcParams.update({
    "figure.facecolor": PALETTE["background"],
    "axes.facecolor": PALETTE["surface"],
    "axes.edgecolor": PALETTE["text"],
    "axes.labelcolor": PALETTE["text"],
    "text.color": PALETTE["text"],
    "xtick.color": PALETTE["text"],
    "ytick.color": PALETTE["text"],
    "grid.color": "#2C2C54",
    "font.family": "DejaVu Sans",
})


class EvaluationVisualizer:
    """
    Generates publication-quality evaluation visualizations.
    All plots saved as PNG to the specified output directory.
    """

    def __init__(self, output_dir: str = "reports/plots"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def plot_confusion_matrix(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        class_names: List[str] = ("Truth", "Lie"),
        title: str = "Confusion Matrix",
        save_name: str = "confusion_matrix.png",
    ) -> str:
        """Plot normalized confusion matrix with counts and percentages."""
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(y_true, y_pred)
        cm_norm = cm.astype(float) / cm.sum(axis=1)[:, np.newaxis]

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.patch.set_facecolor(PALETTE["background"])

        for ax, data, fmt, title_suffix in zip(
            axes, [cm, cm_norm * 100], ["d", ".1f"], ["Counts", "Normalized (%)"]
        ):
            sns.heatmap(
                data, annot=True, fmt=fmt, cmap="RdPu",
                xticklabels=class_names, yticklabels=class_names,
                linewidths=0.5, linecolor="#2C2C54",
                ax=ax, cbar=True,
                annot_kws={"size": 14, "weight": "bold"},
            )
            ax.set_title(f"{title} — {title_suffix}", fontsize=13, pad=12)
            ax.set_ylabel("True Label", fontsize=11)
            ax.set_xlabel("Predicted Label", fontsize=11)

        plt.suptitle(title, fontsize=15, fontweight="bold", y=1.02)
        plt.tight_layout()
        save_path = str(self.output_dir / save_name)
        plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=PALETTE["background"])
        plt.close()
        logger.info(f"Confusion matrix saved to {save_path}")
        return save_path

    def plot_roc_curve(
        self,
        roc_data: Dict,
        model_name: str = "Model",
        save_name: str = "roc_curve.png",
    ) -> str:
        """Plot ROC curve with AUC score."""
        fpr = roc_data["fpr"]
        tpr = roc_data["tpr"]
        auc = roc_data["auc"]

        fig, ax = plt.subplots(figsize=(8, 6))
        fig.patch.set_facecolor(PALETTE["background"])

        ax.plot(fpr, tpr, color=PALETTE["primary"], lw=2.5,
                label=f"{model_name} (AUC = {auc:.3f})")
        ax.plot([0, 1], [0, 1], color="#555", lw=1.5, linestyle="--", label="Random Classifier")
        ax.fill_between(fpr, tpr, alpha=0.15, color=PALETTE["primary"])

        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel("False Positive Rate", fontsize=12)
        ax.set_ylabel("True Positive Rate", fontsize=12)
        ax.set_title("ROC Curve — Deception Detection", fontsize=14, fontweight="bold")
        ax.legend(loc="lower right", fontsize=11, facecolor=PALETTE["surface"])
        ax.grid(True, alpha=0.3)

        save_path = str(self.output_dir / save_name)
        plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=PALETTE["background"])
        plt.close()
        logger.info(f"ROC curve saved to {save_path}")
        return save_path

    def plot_multi_roc_curves(
        self,
        roc_data_dict: Dict[str, Dict],
        save_name: str = "multi_roc_curves.png",
    ) -> str:
        """Plot ROC curves for multiple models on the same axis."""
        colors = [PALETTE["primary"], PALETTE["secondary"], PALETTE["success"],
                  PALETTE["warning"], "#FF9F43", "#74B9FF"]

        fig, ax = plt.subplots(figsize=(9, 7))
        fig.patch.set_facecolor(PALETTE["background"])

        for (model_name, roc_data), color in zip(roc_data_dict.items(), colors):
            ax.plot(
                roc_data["fpr"], roc_data["tpr"], color=color, lw=2,
                label=f"{model_name} (AUC={roc_data['auc']:.3f})"
            )

        ax.plot([0, 1], [0, 1], color="#666", lw=1.5, linestyle="--", label="Chance")
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel("False Positive Rate", fontsize=12)
        ax.set_ylabel("True Positive Rate", fontsize=12)
        ax.set_title("ROC Curve Comparison — All Models", fontsize=14, fontweight="bold")
        ax.legend(loc="lower right", fontsize=10, facecolor=PALETTE["surface"])
        ax.grid(True, alpha=0.3)

        save_path = str(self.output_dir / save_name)
        plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=PALETTE["background"])
        plt.close()
        return save_path

    def plot_precision_recall_curve(
        self,
        pr_data: Dict,
        model_name: str = "Model",
        save_name: str = "pr_curve.png",
    ) -> str:
        """Plot Precision-Recall curve."""
        precision = pr_data["precision"]
        recall = pr_data["recall"]
        pr_auc = pr_data["pr_auc"]

        fig, ax = plt.subplots(figsize=(8, 6))
        fig.patch.set_facecolor(PALETTE["background"])

        ax.plot(recall, precision, color=PALETTE["secondary"], lw=2.5,
                label=f"{model_name} (PR-AUC = {pr_auc:.3f})")
        ax.fill_between(recall, precision, alpha=0.15, color=PALETTE["secondary"])
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel("Recall", fontsize=12)
        ax.set_ylabel("Precision", fontsize=12)
        ax.set_title("Precision-Recall Curve", fontsize=14, fontweight="bold")
        ax.legend(loc="upper right", fontsize=11, facecolor=PALETTE["surface"])
        ax.grid(True, alpha=0.3)

        save_path = str(self.output_dir / save_name)
        plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=PALETTE["background"])
        plt.close()
        logger.info(f"PR curve saved to {save_path}")
        return save_path

    def plot_model_comparison(
        self,
        comparison_data: List[Dict],
        metrics: List[str] = ("accuracy", "f1_macro", "roc_auc", "pr_auc"),
        save_name: str = "model_comparison.png",
    ) -> str:
        """
        Grouped bar chart comparing multiple models across metrics.
        """
        models = [d["model"] for d in comparison_data]
        n_models = len(models)
        n_metrics = len(metrics)
        x = np.arange(n_models)
        width = 0.8 / n_metrics

        metric_colors = [PALETTE["primary"], PALETTE["secondary"],
                         PALETTE["success"], PALETTE["warning"]]

        fig, ax = plt.subplots(figsize=(max(10, n_models * 2.5), 6))
        fig.patch.set_facecolor(PALETTE["background"])

        for i, (metric, color) in enumerate(zip(metrics, metric_colors)):
            values = [d.get(metric, 0) for d in comparison_data]
            offset = (i - n_metrics / 2 + 0.5) * width
            bars = ax.bar(x + offset, values, width, label=metric.replace("_", " ").title(),
                          color=color, alpha=0.85, edgecolor="white", linewidth=0.5)
            for bar in bars:
                h = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2, h + 0.01,
                        f"{h:.2f}", ha="center", va="bottom", fontsize=8)

        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=30, ha="right", fontsize=10)
        ax.set_ylim(0, 1.15)
        ax.set_ylabel("Score", fontsize=12)
        ax.set_title("Model Comparison — All Metrics", fontsize=14, fontweight="bold")
        ax.legend(fontsize=10, facecolor=PALETTE["surface"])
        ax.grid(axis="y", alpha=0.3)

        save_path = str(self.output_dir / save_name)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=PALETTE["background"])
        plt.close()
        logger.info(f"Model comparison chart saved to {save_path}")
        return save_path

    def plot_confidence_distribution(
        self,
        y_prob: np.ndarray,
        y_true: np.ndarray,
        save_name: str = "confidence_distribution.png",
    ) -> str:
        """Plot confidence score distribution by true label."""
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.patch.set_facecolor(PALETTE["background"])

        # KDE by label
        truth_probs = y_prob[y_true == 0]
        lie_probs = y_prob[y_true == 1]

        axes[0].hist(truth_probs, bins=30, alpha=0.7, color=PALETTE["success"],
                     label="Truth Samples", density=True, edgecolor="white", linewidth=0.3)
        axes[0].hist(lie_probs, bins=30, alpha=0.7, color=PALETTE["secondary"],
                     label="Lie Samples", density=True, edgecolor="white", linewidth=0.3)
        axes[0].set_xlabel("Predicted Lie Probability", fontsize=11)
        axes[0].set_ylabel("Density", fontsize=11)
        axes[0].set_title("Confidence Distribution by True Label", fontsize=12)
        axes[0].legend(facecolor=PALETTE["surface"])
        axes[0].grid(alpha=0.3)

        # Calibration plot
        from sklearn.calibration import calibration_curve
        fraction_pos, mean_pred = calibration_curve(y_true, y_prob, n_bins=10)
        axes[1].plot(mean_pred, fraction_pos, "s-", color=PALETTE["primary"],
                     lw=2, markersize=6, label="Model Calibration")
        axes[1].plot([0, 1], [0, 1], "--", color="#666", label="Perfect Calibration")
        axes[1].set_xlabel("Mean Predicted Probability", fontsize=11)
        axes[1].set_ylabel("Fraction of Positives", fontsize=11)
        axes[1].set_title("Calibration Curve", fontsize=12)
        axes[1].legend(facecolor=PALETTE["surface"])
        axes[1].grid(alpha=0.3)

        plt.suptitle("Prediction Confidence Analysis", fontsize=14, fontweight="bold")
        plt.tight_layout()

        save_path = str(self.output_dir / save_name)
        plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=PALETTE["background"])
        plt.close()
        return save_path
