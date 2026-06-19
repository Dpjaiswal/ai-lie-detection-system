"""
Multi-Modal Fusion Models — AI-Powered Lie Detection System
Early Fusion, Late Fusion, and Hybrid Attention Fusion.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import classification_report, roc_auc_score

from src.utils.logger import logger
from src.utils.helpers import set_seed


# ──────────────────────────────────────────────────────────────
# Multi-Modal Dataset
# ──────────────────────────────────────────────────────────────

class MultiModalDataset(Dataset):
    """Dataset combining text embeddings, audio features, and labels."""

    def __init__(
        self,
        text_embeddings: np.ndarray,
        audio_features: np.ndarray,
        labels: List[int],
    ):
        assert len(text_embeddings) == len(audio_features) == len(labels)
        self.text_emb = torch.tensor(text_embeddings, dtype=torch.float32)
        self.audio_feat = torch.tensor(audio_features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "text": self.text_emb[idx],
            "audio": self.audio_feat[idx],
            "label": self.labels[idx],
        }


# ──────────────────────────────────────────────────────────────
# Early Fusion — Concatenate embeddings, train joint classifier
# ──────────────────────────────────────────────────────────────

class EarlyFusionModel(nn.Module):
    """
    Early Fusion: concatenate text + audio embeddings, then classify.

    Architecture:
    [text_emb (768)] ++ [audio_feat (163)] → (931,) → Dense layers → output

    Pros: Joint training, captures cross-modal correlations
    Cons: Requires aligned paired data; modality imbalance issues

    Expected improvement over unimodal: +3–6% F1
    """

    def __init__(
        self,
        text_dim: int = 768,
        audio_dim: int = 163,
        hidden_dims: List[int] = (512, 256, 128),
        num_classes: int = 2,
        dropout: float = 0.4,
    ):
        super().__init__()
        input_dim = text_dim + audio_dim
        layers = []
        prev_dim = input_dim

        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout),
            ])
            prev_dim = h_dim

        layers.append(nn.Linear(prev_dim, num_classes))
        self.network = nn.Sequential(*layers)

    def forward(self, text: torch.Tensor, audio: torch.Tensor) -> torch.Tensor:
        fused = torch.cat([text, audio], dim=-1)
        return self.network(fused)

    def get_embedding(self, text: torch.Tensor, audio: torch.Tensor) -> torch.Tensor:
        """Return penultimate layer embedding."""
        fused = torch.cat([text, audio], dim=-1)
        for layer in list(self.network.children())[:-1]:
            fused = layer(fused) if not isinstance(layer, nn.BatchNorm1d) else fused
        return fused


# ──────────────────────────────────────────────────────────────
# Late Fusion — Combine output probabilities
# ──────────────────────────────────────────────────────────────

class LateFusionModel:
    """
    Late Fusion: combine text and audio model probabilities.

    Methods:
    1. Weighted average: final = w_text * p_text + w_audio * p_audio
    2. Learned meta-classifier: Logistic regression on [p_text, p_audio]
    3. Stacking: XGBoost on probability stack

    Pros: Each modality can be trained independently; robust to missing modality
    Cons: No cross-modal feature learning

    Expected improvement: +2–4% F1 over best single model
    """

    def __init__(
        self,
        text_weight: float = 0.5,
        audio_weight: float = 0.5,
        method: str = "weighted_average",
    ):
        self.text_weight = text_weight
        self.audio_weight = audio_weight
        self.method = method
        self.meta_classifier = None

    def fit_meta(
        self,
        text_probs: np.ndarray,
        audio_probs: np.ndarray,
        labels: np.ndarray,
    ) -> None:
        """Fit meta-classifier on validation set probabilities."""
        from sklearn.linear_model import LogisticRegression
        X_meta = np.column_stack([text_probs[:, 1], audio_probs[:, 1]])
        self.meta_classifier = LogisticRegression(max_iter=500, C=1.0)
        self.meta_classifier.fit(X_meta, labels)
        logger.info("Meta-classifier fitted for late fusion.")

    def predict(
        self,
        text_probs: np.ndarray,
        audio_probs: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Combine predictions from text and audio models.

        Args:
            text_probs: (N, 2) class probabilities from text model
            audio_probs: (N, 2) class probabilities from audio model

        Returns:
            (predictions, combined_probabilities) of shape (N,) and (N, 2)
        """
        if self.method == "weighted_average":
            combined = (
                self.text_weight * text_probs + self.audio_weight * audio_probs
            )
        elif self.method == "learned" and self.meta_classifier is not None:
            X_meta = np.column_stack([text_probs[:, 1], audio_probs[:, 1]])
            combined_lie_prob = self.meta_classifier.predict_proba(X_meta)
            combined = combined_lie_prob
        elif self.method == "max":
            combined = np.maximum(text_probs, audio_probs)
        else:
            combined = (text_probs + audio_probs) / 2.0

        preds = combined.argmax(axis=1)
        return preds, combined

    def evaluate(
        self,
        text_probs: np.ndarray,
        audio_probs: np.ndarray,
        labels: np.ndarray,
    ) -> Dict:
        preds, combined = self.predict(text_probs, audio_probs)
        report = classification_report(labels, preds, output_dict=True, zero_division=0)
        auc = roc_auc_score(labels, combined[:, 1]) if len(set(labels)) > 1 else 0.5
        return {"classification_report": report, "roc_auc": auc, "method": self.method}


# ──────────────────────────────────────────────────────────────
# Hybrid Attention Fusion — Cross-modal attention
# ──────────────────────────────────────────────────────────────

class HybridAttentionFusion(nn.Module):
    """
    Hybrid Attention Fusion: cross-modal attention between text and audio.

    Architecture:
    text_proj → Q  (query from audio → attend over text)
    audio_proj → K/V
    MultiheadAttention → attended_text
    Concat(attended_text, audio_proj) → Dense → output

    Also reverses: audio attends over text for symmetric fusion.

    Pros: Learns which audio frames / text features matter for each modality
    Cons: More complex, requires more data to train

    Expected improvement: +4–8% F1 over late fusion
    """

    def __init__(
        self,
        text_dim: int = 768,
        audio_dim: int = 163,
        projection_dim: int = 256,
        num_heads: int = 8,
        hidden_dims: List[int] = (256, 128),
        num_classes: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()

        # Project both modalities to same dimension
        self.text_proj = nn.Sequential(
            nn.Linear(text_dim, projection_dim),
            nn.LayerNorm(projection_dim),
            nn.ReLU(inplace=True),
        )
        self.audio_proj = nn.Sequential(
            nn.Linear(audio_dim, projection_dim),
            nn.LayerNorm(projection_dim),
            nn.ReLU(inplace=True),
        )

        # Cross-modal attention: text attends to audio and vice-versa
        self.text_to_audio_attention = nn.MultiheadAttention(
            embed_dim=projection_dim, num_heads=num_heads,
            dropout=dropout, batch_first=True,
        )
        self.audio_to_text_attention = nn.MultiheadAttention(
            embed_dim=projection_dim, num_heads=num_heads,
            dropout=dropout, batch_first=True,
        )

        # Gating mechanism
        self.text_gate = nn.Sequential(nn.Linear(projection_dim, projection_dim), nn.Sigmoid())
        self.audio_gate = nn.Sequential(nn.Linear(projection_dim, projection_dim), nn.Sigmoid())

        # Final classifier
        classifier_in = projection_dim * 2
        layers = []
        prev = classifier_in
        for h in hidden_dims:
            layers.extend([nn.Linear(prev, h), nn.ReLU(inplace=True), nn.Dropout(dropout)])
            prev = h
        layers.append(nn.Linear(prev, num_classes))
        self.classifier = nn.Sequential(*layers)

        self.dropout = nn.Dropout(dropout)

    def forward(self, text: torch.Tensor, audio: torch.Tensor) -> torch.Tensor:
        """
        Args:
            text: (B, text_dim) text embeddings
            audio: (B, audio_dim) audio feature vectors

        Returns:
            logits: (B, num_classes)
        """
        # Project to common space — add sequence dim for attention: (B, 1, D)
        t = self.text_proj(text).unsqueeze(1)   # (B, 1, D)
        a = self.audio_proj(audio).unsqueeze(1)  # (B, 1, D)

        # Cross-modal attention
        t_attn, _ = self.text_to_audio_attention(query=t, key=a, value=a)
        a_attn, _ = self.audio_to_text_attention(query=a, key=t, value=t)

        # Gating + residual
        t_out = t.squeeze(1) * self.text_gate(t_attn.squeeze(1)) + t.squeeze(1)
        a_out = a.squeeze(1) * self.audio_gate(a_attn.squeeze(1)) + a.squeeze(1)

        # Fuse and classify
        fused = torch.cat([t_out, a_out], dim=-1)  # (B, D*2)
        fused = self.dropout(fused)
        return self.classifier(fused)

    def get_embedding(self, text: torch.Tensor, audio: torch.Tensor) -> torch.Tensor:
        """Return fused representation before final linear layer."""
        t = self.text_proj(text).unsqueeze(1)
        a = self.audio_proj(audio).unsqueeze(1)
        t_attn, _ = self.text_to_audio_attention(query=t, key=a, value=a)
        a_attn, _ = self.audio_to_text_attention(query=a, key=t, value=t)
        t_out = t.squeeze(1) * self.text_gate(t_attn.squeeze(1)) + t.squeeze(1)
        a_out = a.squeeze(1) * self.audio_gate(a_attn.squeeze(1)) + a.squeeze(1)
        return torch.cat([t_out, a_out], dim=-1)


# ──────────────────────────────────────────────────────────────
# Fusion Comparison Summary
# ──────────────────────────────────────────────────────────────

FUSION_COMPARISON = {
    "early_fusion": {
        "description": "Concatenate embeddings, train single joint model",
        "pros": ["Captures cross-modal correlations", "End-to-end training"],
        "cons": ["Requires paired data", "Modality imbalance risk"],
        "expected_f1_gain": "+3–6% over best unimodal",
        "complexity": "Medium",
    },
    "late_fusion": {
        "description": "Independently trained models, combine output probabilities",
        "pros": ["Handles missing modality", "Modular training"],
        "cons": ["No cross-modal feature learning"],
        "expected_f1_gain": "+2–4% over best unimodal",
        "complexity": "Low",
    },
    "hybrid_attention_fusion": {
        "description": "Cross-modal attention + gating mechanism",
        "pros": ["Learns which features matter per modality", "Symmetric attention"],
        "cons": ["Requires more data", "Slower training"],
        "expected_f1_gain": "+4–8% over late fusion",
        "complexity": "High",
    },
}
