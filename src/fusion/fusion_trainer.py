"""
Fusion Trainer — AI-Powered Lie Detection System
Trains and evaluates all fusion methods.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, roc_auc_score

from src.fusion.hybrid_fusion import (
    MultiModalDataset, EarlyFusionModel, LateFusionModel, HybridAttentionFusion
)
from src.utils.logger import logger
from src.utils.helpers import set_seed, save_json


class FusionTrainer:
    """
    Trains and compares Early, Late, and Hybrid Attention fusion models.
    """

    def __init__(
        self,
        text_dim: int = 768,
        audio_dim: int = 163,
        output_dir: str = "models/fusion",
        results_dir: str = "reports",
        seed: int = 42,
    ):
        set_seed(seed)
        self.text_dim = text_dim
        self.audio_dim = audio_dim
        self.output_dir = Path(output_dir)
        self.results_dir = Path(results_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.results: Dict = {}

        self.device = (
            torch.device("cuda") if torch.cuda.is_available()
            else torch.device("mps") if hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
            else torch.device("cpu")
        )

    def _make_loaders(
        self,
        text_emb_train: np.ndarray, audio_feat_train: np.ndarray, y_train: np.ndarray,
        text_emb_val: np.ndarray, audio_feat_val: np.ndarray, y_val: np.ndarray,
        text_emb_test: np.ndarray, audio_feat_test: np.ndarray, y_test: np.ndarray,
        batch_size: int = 32,
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        train_ds = MultiModalDataset(text_emb_train, audio_feat_train, list(y_train))
        val_ds = MultiModalDataset(text_emb_val, audio_feat_val, list(y_val))
        test_ds = MultiModalDataset(text_emb_test, audio_feat_test, list(y_test))
        return (
            DataLoader(train_ds, batch_size=batch_size, shuffle=True),
            DataLoader(val_ds, batch_size=batch_size),
            DataLoader(test_ds, batch_size=batch_size),
        )

    def _train_torch_model(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int = 30,
        lr: float = 1e-3,
        patience: int = 5,
        checkpoint_path: str = "best_fusion.pth",
    ) -> Dict:
        model = model.to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)
        criterion = nn.CrossEntropyLoss()
        history = {"train_loss": [], "val_loss": [], "val_acc": []}
        best_val_acc = 0.0
        counter = 0

        for epoch in range(1, num_epochs + 1):
            model.train()
            total_loss = 0.0
            for batch in train_loader:
                text = batch["text"].to(self.device)
                audio = batch["audio"].to(self.device)
                labels = batch["label"].to(self.device)
                optimizer.zero_grad()
                logits = model(text, audio)
                loss = criterion(logits, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                total_loss += loss.item()

            avg_loss = total_loss / len(train_loader)
            history["train_loss"].append(avg_loss)

            # Validation
            model.eval()
            val_loss, correct, total = 0.0, 0, 0
            with torch.no_grad():
                for batch in val_loader:
                    text = batch["text"].to(self.device)
                    audio = batch["audio"].to(self.device)
                    labels = batch["label"].to(self.device)
                    logits = model(text, audio)
                    val_loss += criterion(logits, labels).item()
                    preds = logits.argmax(dim=1)
                    correct += (preds == labels).sum().item()
                    total += labels.size(0)

            val_acc = correct / max(total, 1)
            val_loss /= len(val_loader)
            scheduler.step(val_loss)
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)
            logger.info(f"Epoch {epoch}/{num_epochs} — Loss: {avg_loss:.4f}, Val Acc: {val_acc:.4f}")

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                counter = 0
                torch.save(model.state_dict(), checkpoint_path)
            else:
                counter += 1
                if counter >= patience:
                    logger.info(f"Early stopping at epoch {epoch}")
                    break

        model.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
        return history

    @torch.no_grad()
    def _evaluate_torch_model(
        self, model: nn.Module, test_loader: DataLoader, y_test: np.ndarray
    ) -> Dict:
        model.eval()
        all_preds, all_probs = [], []
        for batch in test_loader:
            text = batch["text"].to(self.device)
            audio = batch["audio"].to(self.device)
            logits = model(text, audio)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            preds = np.argmax(probs, axis=1)
            all_preds.extend(preds)
            all_probs.extend(probs)

        all_preds = np.array(all_preds)
        all_probs = np.array(all_probs)
        report = classification_report(y_test, all_preds, output_dict=True, zero_division=0)
        auc = roc_auc_score(y_test, all_probs[:, 1]) if len(set(y_test)) > 1 else 0.5
        return {"classification_report": report, "roc_auc": auc}

    def train_early_fusion(
        self,
        text_emb_splits: Dict[str, np.ndarray],
        audio_feat_splits: Dict[str, np.ndarray],
        label_splits: Dict[str, np.ndarray],
        **kwargs,
    ) -> Dict:
        """Train Early Fusion model."""
        logger.info("Training Early Fusion model...")
        train_loader, val_loader, test_loader = self._make_loaders(
            text_emb_splits["train"], audio_feat_splits["train"], label_splits["train"],
            text_emb_splits["val"], audio_feat_splits["val"], label_splits["val"],
            text_emb_splits["test"], audio_feat_splits["test"], label_splits["test"],
        )
        model = EarlyFusionModel(text_dim=self.text_dim, audio_dim=self.audio_dim)
        history = self._train_torch_model(
            model, train_loader, val_loader,
            checkpoint_path=str(self.output_dir / "early_fusion_best.pth"),
            **kwargs
        )
        eval_result = self._evaluate_torch_model(model, test_loader, label_splits["test"])
        result = {"history": history, **eval_result}
        logger.info(f"Early Fusion AUC: {eval_result['roc_auc']:.4f}")
        torch.save(model.state_dict(), str(self.output_dir / "early_fusion.pth"))
        return result

    def train_late_fusion(
        self,
        text_probs_splits: Dict[str, np.ndarray],
        audio_probs_splits: Dict[str, np.ndarray],
        label_splits: Dict[str, np.ndarray],
        method: str = "weighted_average",
        text_weight: float = 0.5,
        audio_weight: float = 0.5,
    ) -> Dict:
        """Evaluate Late Fusion combination."""
        logger.info(f"Evaluating Late Fusion (method={method})...")
        model = LateFusionModel(text_weight=text_weight, audio_weight=audio_weight, method=method)
        if method == "learned":
            model.fit_meta(
                text_probs_splits["val"], audio_probs_splits["val"], label_splits["val"]
            )
        result = model.evaluate(
            text_probs_splits["test"], audio_probs_splits["test"], label_splits["test"]
        )
        logger.info(f"Late Fusion AUC: {result['roc_auc']:.4f}")
        return result

    def train_hybrid_fusion(
        self,
        text_emb_splits: Dict[str, np.ndarray],
        audio_feat_splits: Dict[str, np.ndarray],
        label_splits: Dict[str, np.ndarray],
        **kwargs,
    ) -> Dict:
        """Train Hybrid Attention Fusion model."""
        logger.info("Training Hybrid Attention Fusion model...")
        train_loader, val_loader, test_loader = self._make_loaders(
            text_emb_splits["train"], audio_feat_splits["train"], label_splits["train"],
            text_emb_splits["val"], audio_feat_splits["val"], label_splits["val"],
            text_emb_splits["test"], audio_feat_splits["test"], label_splits["test"],
        )
        model = HybridAttentionFusion(text_dim=self.text_dim, audio_dim=self.audio_dim)
        history = self._train_torch_model(
            model, train_loader, val_loader,
            checkpoint_path=str(self.output_dir / "hybrid_fusion_best.pth"),
            **kwargs
        )
        eval_result = self._evaluate_torch_model(model, test_loader, label_splits["test"])
        result = {"history": history, **eval_result}
        logger.info(f"Hybrid Attention Fusion AUC: {eval_result['roc_auc']:.4f}")
        torch.save(model.state_dict(), str(self.output_dir / "hybrid_fusion.pth"))
        return result

    def compare_all(self, results: Dict) -> Dict:
        """Print comparison table of all fusion methods."""
        print("\n" + "="*60)
        print("  FUSION METHOD COMPARISON")
        print("="*60)
        print(f"{'Method':<30} {'ROC-AUC':>10} {'F1 (lie)':>10}")
        print("-"*60)
        for method, result in results.items():
            auc = result.get("roc_auc", 0.0)
            f1 = result.get("classification_report", {}).get("1", {}).get("f1-score", 0.0)
            print(f"{method:<30} {auc:>10.4f} {f1:>10.4f}")
        print("="*60 + "\n")
        return results
