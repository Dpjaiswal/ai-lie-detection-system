"""
Transformer Model — AI-Powered Lie Detection System
Fine-tuning BERT, RoBERTa, and DeBERTa for deception detection.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)
from sklearn.metrics import classification_report, roc_auc_score
from tqdm import tqdm

from src.utils.logger import logger
from src.utils.config import config
from src.utils.helpers import set_seed


# ──────────────────────────────────────────────────────────────
# PyTorch Dataset
# ──────────────────────────────────────────────────────────────

class TextDeceptionDataset(Dataset):
    """
    PyTorch Dataset for text deception classification.
    Wraps tokenized inputs for transformer models.
    """

    def __init__(
        self,
        texts: List[str],
        labels: List[int],
        tokenizer,
        max_length: int = 512,
    ):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        encoding = self.tokenizer(
            self.texts[idx],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "token_type_ids": encoding.get("token_type_ids", torch.zeros(self.max_length, dtype=torch.long)).squeeze(0),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }


# ──────────────────────────────────────────────────────────────
# Transformer Classifier
# ──────────────────────────────────────────────────────────────

class TransformerDeceptionClassifier:
    """
    Fine-tuning wrapper for BERT, RoBERTa, and DeBERTa.

    Model comparison:
    ┌─────────────────────────────┬──────────┬──────────┬─────────────┐
    │ Model                       │ Params   │ GPU RAM  │ Exp. F1     │
    ├─────────────────────────────┼──────────┼──────────┼─────────────┤
    │ bert-base-uncased           │ 110M     │ ~3GB     │ 0.68–0.72   │
    │ roberta-base                │ 125M     │ ~3.5GB   │ 0.70–0.74   │
    │ microsoft/deberta-v3-base   │ 184M     │ ~5GB     │ 0.72–0.76   │
    └─────────────────────────────┴──────────┴──────────┴─────────────┘
    """

    def __init__(
        self,
        model_name: str = "roberta-base",
        num_labels: int = 2,
        max_length: int = 512,
        seed: int = 42,
    ):
        set_seed(seed)
        self.model_name = model_name
        self.num_labels = num_labels
        self.max_length = max_length
        self.seed = seed
        self.device = self._get_device()
        self.tokenizer = None
        self.model = None

    def _get_device(self) -> torch.device:
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    def load(self) -> "TransformerDeceptionClassifier":
        """Load pretrained tokenizer and model."""
        logger.info(f"Loading {self.model_name} on {self.device}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=self.num_labels,
        )
        self.model.to(self.device)
        return self

    def _make_dataloader(
        self, texts: List[str], labels: List[int], batch_size: int, shuffle: bool = False
    ) -> DataLoader:
        dataset = TextDeceptionDataset(texts, labels, self.tokenizer, self.max_length)
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)

    def train(
        self,
        train_texts: List[str],
        train_labels: List[int],
        val_texts: Optional[List[str]] = None,
        val_labels: Optional[List[int]] = None,
        num_epochs: int = 5,
        batch_size: int = 16,
        learning_rate: float = 2e-5,
        weight_decay: float = 0.01,
        warmup_ratio: float = 0.1,
        gradient_accumulation_steps: int = 2,
        fp16: bool = False,
        output_dir: str = "models/nlp",
        early_stopping_patience: int = 3,
    ) -> Dict:
        """
        Fine-tune the transformer model.

        Returns dict of training history with per-epoch metrics.
        """
        if self.model is None:
            self.load()

        train_loader = self._make_dataloader(train_texts, train_labels, batch_size, shuffle=True)
        val_loader = (
            self._make_dataloader(val_texts, val_labels, batch_size)
            if val_texts else None
        )

        optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
        total_steps = len(train_loader) * num_epochs // gradient_accumulation_steps
        warmup_steps = int(total_steps * warmup_ratio)
        scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

        scaler = torch.cuda.amp.GradScaler() if fp16 and self.device.type == "cuda" else None

        history = {"train_loss": [], "val_loss": [], "val_f1": [], "val_auc": []}
        best_val_f1 = 0.0
        patience_counter = 0

        for epoch in range(1, num_epochs + 1):
            # ── Training ──────────────────────────────────────
            self.model.train()
            total_loss = 0.0
            optimizer.zero_grad()

            for step, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs}")):
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                if scaler:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(input_ids, attention_mask=attention_mask, labels=labels)
                    scaler.scale(outputs.loss / gradient_accumulation_steps).backward()
                else:
                    outputs = self.model(input_ids, attention_mask=attention_mask, labels=labels)
                    (outputs.loss / gradient_accumulation_steps).backward()

                total_loss += outputs.loss.item()

                if (step + 1) % gradient_accumulation_steps == 0:
                    if scaler:
                        scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    if scaler:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()

            avg_loss = total_loss / len(train_loader)
            history["train_loss"].append(avg_loss)
            logger.info(f"Epoch {epoch} — Train Loss: {avg_loss:.4f}")

            # ── Validation ────────────────────────────────────
            if val_loader:
                val_metrics = self._evaluate_loader(val_loader)
                history["val_loss"].append(val_metrics["loss"])
                history["val_f1"].append(val_metrics["f1"])
                history["val_auc"].append(val_metrics["auc"])
                logger.info(
                    f"Epoch {epoch} — Val Loss: {val_metrics['loss']:.4f}, "
                    f"F1: {val_metrics['f1']:.4f}, AUC: {val_metrics['auc']:.4f}"
                )
                if val_metrics["f1"] > best_val_f1:
                    best_val_f1 = val_metrics["f1"]
                    patience_counter = 0
                    self.save(output_dir)
                    logger.info(f"New best model saved (F1={best_val_f1:.4f})")
                else:
                    patience_counter += 1
                    if patience_counter >= early_stopping_patience:
                        logger.info(f"Early stopping triggered at epoch {epoch}")
                        break

        return history

    @torch.no_grad()
    def _evaluate_loader(self, loader: DataLoader) -> Dict:
        """Run evaluation on a DataLoader and return metrics."""
        self.model.eval()
        all_preds, all_labels, all_probs = [], [], []
        total_loss = 0.0

        for batch in loader:
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["labels"].to(self.device)
            outputs = self.model(input_ids, attention_mask=attention_mask, labels=labels)
            total_loss += outputs.loss.item()
            probs = torch.softmax(outputs.logits, dim=-1)[:, 1].cpu().numpy()
            preds = outputs.logits.argmax(dim=-1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs)

        report = classification_report(all_labels, all_preds, output_dict=True, zero_division=0)
        auc = roc_auc_score(all_labels, all_probs) if len(set(all_labels)) > 1 else 0.5

        return {
            "loss": total_loss / len(loader),
            "f1": report.get("1", {}).get("f1-score", 0.0),
            "accuracy": report.get("accuracy", 0.0),
            "auc": auc,
            "report": report,
        }

    @torch.no_grad()
    def predict(self, texts: List[str], batch_size: int = 32) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict labels and probabilities for a list of texts.
        Returns: (labels array, probabilities array of shape (N,2))
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load() or train() first.")

        dummy_labels = [0] * len(texts)
        loader = self._make_dataloader(texts, dummy_labels, batch_size)
        self.model.eval()

        all_preds, all_probs = [], []
        for batch in loader:
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            outputs = self.model(input_ids, attention_mask=attention_mask)
            probs = torch.softmax(outputs.logits, dim=-1).cpu().numpy()
            preds = np.argmax(probs, axis=1)
            all_preds.extend(preds)
            all_probs.extend(probs)

        return np.array(all_preds), np.array(all_probs)

    def save(self, output_dir: str) -> None:
        """Save model and tokenizer."""
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        logger.info(f"Transformer model saved to {output_dir}")

    def load_from_checkpoint(self, checkpoint_dir: str) -> "TransformerDeceptionClassifier":
        """Load fine-tuned model from checkpoint."""
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(checkpoint_dir)
        self.model.to(self.device)
        self.model.eval()
        logger.info(f"Model loaded from {checkpoint_dir}")
        return self
