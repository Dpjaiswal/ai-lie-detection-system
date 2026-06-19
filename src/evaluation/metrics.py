"""
Evaluation Metrics — AI-Powered Lie Detection System
Complete evaluation framework: classification, probability, and comparison metrics.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, log_loss, classification_report,
    confusion_matrix, average_precision_score,
)

from src.utils.logger import logger
from src.utils.helpers import save_json


class EvaluationMetrics:
    """
    Comprehensive model evaluation framework.

    Computes:
    - Classification: Accuracy, Precision, Recall, F1 (macro/weighted/per-class)
    - Probability: ROC-AUC, Log Loss, Average Precision (PR-AUC)
    - Multi-model comparison table
    """

    @staticmethod
    def compute_all(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: Optional[np.ndarray] = None,
        pos_label: int = 1,
    ) -> Dict:
        """
        Compute all classification metrics.

        Args:
            y_true: Ground truth labels (N,)
            y_pred: Predicted labels (N,)
            y_prob: Predicted probabilities for positive class (N,) — optional
            pos_label: Positive class label

        Returns:
            Dict of all metric values
        """
        metrics = {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
            "precision_weighted": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
            "precision_positive": float(precision_score(y_true, y_pred, pos_label=pos_label, average="binary", zero_division=0)),
            "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
            "recall_weighted": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
            "recall_positive": float(recall_score(y_true, y_pred, pos_label=pos_label, average="binary", zero_division=0)),
            "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
            "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
            "f1_positive": float(f1_score(y_true, y_pred, pos_label=pos_label, average="binary", zero_division=0)),
            "classification_report": classification_report(
                y_true, y_pred, output_dict=True, zero_division=0
            ),
        }

        if y_prob is not None and len(set(y_true)) > 1:
            metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob))
            metrics["log_loss"] = float(log_loss(y_true, y_prob))
            metrics["pr_auc"] = float(average_precision_score(y_true, y_prob, pos_label=pos_label))

        return metrics

    @staticmethod
    def confusion_matrix_data(y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
        """Compute confusion matrix as nested dict."""
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (0, 0, 0, 0)
        return {
            "matrix": cm.tolist(),
            "true_negative": int(tn),
            "false_positive": int(fp),
            "false_negative": int(fn),
            "true_positive": int(tp),
            "false_positive_rate": float(fp / max(fp + tn, 1)),
            "false_negative_rate": float(fn / max(fn + tp, 1)),
        }

    @staticmethod
    def roc_curve_data(
        y_true: np.ndarray, y_prob: np.ndarray
    ) -> Dict[str, List[float]]:
        """Compute ROC curve data points."""
        from sklearn.metrics import roc_curve
        fpr, tpr, thresholds = roc_curve(y_true, y_prob)
        return {
            "fpr": fpr.tolist(),
            "tpr": tpr.tolist(),
            "thresholds": thresholds.tolist(),
            "auc": float(roc_auc_score(y_true, y_prob)),
        }

    @staticmethod
    def pr_curve_data(
        y_true: np.ndarray, y_prob: np.ndarray
    ) -> Dict[str, List[float]]:
        """Compute Precision-Recall curve data points."""
        from sklearn.metrics import precision_recall_curve, average_precision_score
        precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
        return {
            "precision": precision.tolist(),
            "recall": recall.tolist(),
            "thresholds": thresholds.tolist(),
            "pr_auc": float(average_precision_score(y_true, y_prob)),
        }

    @staticmethod
    def compare_models(model_results: Dict[str, Dict]) -> Dict:
        """
        Create comparison table across multiple models.

        Args:
            model_results: Dict mapping model_name → metrics dict

        Returns:
            Sorted comparison table
        """
        comparison = []
        for model_name, metrics in model_results.items():
            row = {
                "model": model_name,
                "accuracy": metrics.get("accuracy", 0.0),
                "f1_macro": metrics.get("f1_macro", 0.0),
                "f1_positive": metrics.get("f1_positive", 0.0),
                "roc_auc": metrics.get("roc_auc", 0.0),
                "log_loss": metrics.get("log_loss", float("inf")),
                "pr_auc": metrics.get("pr_auc", 0.0),
            }
            comparison.append(row)

        # Sort by ROC-AUC descending
        comparison.sort(key=lambda x: x["roc_auc"], reverse=True)
        return {"ranking": comparison, "best_model": comparison[0]["model"] if comparison else None}

    @staticmethod
    def print_report(metrics: Dict, model_name: str = "Model") -> None:
        """Pretty-print evaluation metrics to console."""
        print(f"\n{'='*55}")
        print(f"  Evaluation Report: {model_name}")
        print(f"{'='*55}")
        print(f"  Accuracy        : {metrics.get('accuracy', 0):.4f}")
        print(f"  F1 (Macro)      : {metrics.get('f1_macro', 0):.4f}")
        print(f"  F1 (Lie class)  : {metrics.get('f1_positive', 0):.4f}")
        print(f"  Precision (Lie) : {metrics.get('precision_positive', 0):.4f}")
        print(f"  Recall (Lie)    : {metrics.get('recall_positive', 0):.4f}")
        if "roc_auc" in metrics:
            print(f"  ROC-AUC         : {metrics['roc_auc']:.4f}")
            print(f"  Log Loss        : {metrics['log_loss']:.4f}")
            print(f"  PR-AUC          : {metrics['pr_auc']:.4f}")
        print(f"{'='*55}\n")

    def save_results(self, results: Dict, path: str) -> None:
        """Save all evaluation results to JSON."""
        save_json(results, path)
        logger.info(f"Evaluation results saved to {path}")


def main():
    """Demo: generate synthetic results and evaluate."""
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-file", type=str, default="reports/evaluation_results.json")
    args = parser.parse_args()

    # Simulate predictions
    np.random.seed(42)
    y_true = np.random.randint(0, 2, 200)
    y_prob = np.random.beta(2, 2, 200)
    y_pred = (y_prob > 0.5).astype(int)

    metrics = EvaluationMetrics.compute_all(y_true, y_pred, y_prob)
    EvaluationMetrics.print_report(metrics, "Demo Model")
    EvaluationMetrics().save_results(metrics, args.results_file)


if __name__ == "__main__":
    main()
