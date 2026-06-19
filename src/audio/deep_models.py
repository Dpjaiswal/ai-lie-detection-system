"""
Deep Learning Audio Models — AI-Powered Lie Detection System
CNN on Spectrograms, LSTM on features, CNN-LSTM Hybrid.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from src.utils.logger import logger
from src.utils.config import config
from src.utils.helpers import set_seed


# ──────────────────────────────────────────────────────────────
# Datasets
# ──────────────────────────────────────────────────────────────

class SpectrogramDataset(Dataset):
    """Dataset for CNN — expects (N, H, W) mel spectrograms."""

    def __init__(self, spectrograms: np.ndarray, labels: List[int], target_size: Tuple[int,int] = (128, 128)):
        self.spectrograms = spectrograms
        self.labels = labels
        self.target_size = target_size

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        spec = self.spectrograms[idx]
        # Resize to fixed target size
        spec_tensor = torch.tensor(spec, dtype=torch.float32).unsqueeze(0)  # (1, H, W)
        spec_tensor = F.interpolate(spec_tensor.unsqueeze(0), size=self.target_size, mode="bilinear", align_corners=False).squeeze(0)
        # Normalize to [0, 1]
        spec_min, spec_max = spec_tensor.min(), spec_tensor.max()
        if spec_max > spec_min:
            spec_tensor = (spec_tensor - spec_min) / (spec_max - spec_min)
        return {
            "spectrogram": spec_tensor,
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
        }


class FeatureSequenceDataset(Dataset):
    """Dataset for LSTM — expects (N, T, F) feature sequences."""

    def __init__(self, features: np.ndarray, labels: List[int]):
        self.features = features
        self.labels = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "features": torch.tensor(self.features[idx], dtype=torch.float32),
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
        }


# ──────────────────────────────────────────────────────────────
# CNN Model (Spectrogram Input)
# ──────────────────────────────────────────────────────────────

class AudioCNN(nn.Module):
    """
    CNN for mel spectrogram deception detection.

    Architecture:
    Input: (B, 1, 128, 128) — single-channel mel spectrogram
    4 Conv blocks (Conv2D → BatchNorm → ReLU → MaxPool)
    Global Average Pooling → Dense → Dropout → Classifier

    Why CNN: Local pattern recognition in time-frequency domain.
    Captures formant transitions, energy patterns, micropauses.
    """

    def __init__(
        self,
        num_classes: int = 2,
        filters: List[int] = (32, 64, 128, 256),
        dropout: float = 0.3,
    ):
        super().__init__()
        in_channels = 1
        conv_layers = []
        for out_channels in filters:
            conv_layers.extend([
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Dropout2d(p=0.1),
            ])
            in_channels = out_channels
        self.conv_block = nn.Sequential(*conv_layers)
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(filters[-1], 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_block(x)
        x = self.global_avg_pool(x)
        return self.classifier(x)

    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """Return 128-dim embedding before final classifier."""
        x = self.conv_block(x)
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)
        for i, layer in enumerate(self.classifier):
            if isinstance(layer, nn.Linear) and i >= 6:  # Last Linear
                return x
            x = layer(x)
        return x


# ──────────────────────────────────────────────────────────────
# LSTM Model (Feature Sequence Input)
# ──────────────────────────────────────────────────────────────

class AudioLSTM(nn.Module):
    """
    Bidirectional LSTM for temporal audio feature modeling.

    Input: (B, T, input_size) — MFCC sequence frames
    Bidirectional LSTM stacks → attention pooling → dense → classifier

    Why BiLSTM: Captures forward (building stress) and backward
    (resolution patterns) temporal dependencies in speech.
    """

    def __init__(
        self,
        input_size: int = 40,
        hidden_size: int = 128,
        num_layers: int = 2,
        num_classes: int = 2,
        dropout: float = 0.3,
        bidirectional: bool = True,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )

        # Attention mechanism
        attn_size = hidden_size * self.num_directions
        self.attention = nn.Linear(attn_size, 1)

        self.classifier = nn.Sequential(
            nn.Linear(attn_size, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, F)
        lstm_out, _ = self.lstm(x)  # (B, T, H*dir)
        # Attention pooling
        attn_weights = torch.softmax(self.attention(lstm_out), dim=1)  # (B, T, 1)
        context = (lstm_out * attn_weights).sum(dim=1)  # (B, H*dir)
        return self.classifier(context)

    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        lstm_out, _ = self.lstm(x)
        attn_weights = torch.softmax(self.attention(lstm_out), dim=1)
        return (lstm_out * attn_weights).sum(dim=1)


# ──────────────────────────────────────────────────────────────
# CNN-LSTM Hybrid
# ──────────────────────────────────────────────────────────────

class CNNLSTMHybrid(nn.Module):
    """
    CNN-LSTM hybrid model.

    Architecture:
    CNN encodes local spectrogram patterns → sequence fed into BiLSTM
    → attention pooling → binary classifier

    Why hybrid: CNN captures short-term spectral patterns, LSTM models
    long-range temporal dynamics. Outperforms both standalone CNN and LSTM.

    Input: (B, 1, n_mels, T) mel spectrogram
    """

    def __init__(
        self,
        n_mels: int = 128,
        cnn_channels: List[int] = (64, 128),
        lstm_hidden: int = 128,
        lstm_layers: int = 2,
        num_classes: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()

        # ── CNN frontend ──────────────────────────────────────
        cnn_layers = []
        in_ch = 1
        for ch in cnn_channels:
            cnn_layers.extend([
                nn.Conv2d(in_ch, ch, kernel_size=(3, 3), padding=(1, 1)),
                nn.BatchNorm2d(ch),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=(2, 1)),  # pool freq, keep time
            ])
            in_ch = ch
        self.cnn = nn.Sequential(*cnn_layers)

        # Compute LSTM input size after CNN
        freq_out = n_mels // (2 ** len(cnn_channels))
        lstm_input_size = cnn_channels[-1] * freq_out

        # ── BiLSTM backend ────────────────────────────────────
        self.lstm = nn.LSTM(
            input_size=lstm_input_size,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
            bidirectional=True,
        )
        self.attention = nn.Linear(lstm_hidden * 2, 1)

        self.classifier = nn.Sequential(
            nn.Linear(lstm_hidden * 2, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 1, n_mels, T)
        cnn_out = self.cnn(x)  # (B, C, F', T)
        B, C, F, T = cnn_out.shape
        # Reshape to (B, T, C*F) for LSTM
        cnn_out = cnn_out.permute(0, 3, 1, 2).contiguous().view(B, T, C * F)
        lstm_out, _ = self.lstm(cnn_out)  # (B, T, H*2)
        attn = torch.softmax(self.attention(lstm_out), dim=1)
        context = (lstm_out * attn).sum(dim=1)
        return self.classifier(context)


# ──────────────────────────────────────────────────────────────
# Deep Audio Model Trainer
# ──────────────────────────────────────────────────────────────

class DeepAudioTrainer:
    """Trains deep learning audio models with standard training loop."""

    def __init__(self, model: nn.Module, device: Optional[str] = None):
        if device:
            self.device = torch.device(device)
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        self.model = model.to(self.device)
        logger.info(f"Model on {self.device}: {model.__class__.__name__}")

    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        num_epochs: int = 30,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        output_dir: str = "models/audio",
        patience: int = 5,
    ) -> Dict:
        """Train the model and return training history."""
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)
        criterion = nn.CrossEntropyLoss()
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        history = {"train_loss": [], "val_loss": [], "val_acc": []}
        best_val_acc = 0.0
        patience_counter = 0

        for epoch in range(1, num_epochs + 1):
            # Training
            self.model.train()
            total_loss = 0.0
            for batch in train_loader:
                optimizer.zero_grad()
                inputs = {k: v.to(self.device) for k, v in batch.items() if k != "label"}
                labels = batch["label"].to(self.device)
                key = "spectrogram" if "spectrogram" in inputs else "features"
                logits = self.model(inputs[key])
                loss = criterion(logits, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                total_loss += loss.item()

            avg_loss = total_loss / len(train_loader)
            history["train_loss"].append(avg_loss)

            # Validation
            if val_loader:
                val_metrics = self._validate(val_loader, criterion)
                history["val_loss"].append(val_metrics["loss"])
                history["val_acc"].append(val_metrics["accuracy"])
                scheduler.step(val_metrics["loss"])
                logger.info(f"Epoch {epoch}/{num_epochs} — Loss: {avg_loss:.4f}, "
                            f"Val Loss: {val_metrics['loss']:.4f}, Val Acc: {val_metrics['accuracy']:.4f}")

                if val_metrics["accuracy"] > best_val_acc:
                    best_val_acc = val_metrics["accuracy"]
                    patience_counter = 0
                    torch.save(self.model.state_dict(), f"{output_dir}/best_model.pth")
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        logger.info(f"Early stopping at epoch {epoch}")
                        break
            else:
                logger.info(f"Epoch {epoch}/{num_epochs} — Train Loss: {avg_loss:.4f}")

        return history

    @torch.no_grad()
    def _validate(self, loader: DataLoader, criterion: nn.Module) -> Dict:
        self.model.eval()
        total_loss, correct, total = 0.0, 0, 0
        for batch in loader:
            inputs = {k: v.to(self.device) for k, v in batch.items() if k != "label"}
            labels = batch["label"].to(self.device)
            key = "spectrogram" if "spectrogram" in inputs else "features"
            logits = self.model(inputs[key])
            loss = criterion(logits, labels)
            total_loss += loss.item()
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
        return {"loss": total_loss / len(loader), "accuracy": correct / max(total, 1)}

    @torch.no_grad()
    def predict(self, loader: DataLoader) -> Tuple[np.ndarray, np.ndarray]:
        self.model.eval()
        all_preds, all_probs = [], []
        for batch in loader:
            inputs = {k: v.to(self.device) for k, v in batch.items() if k != "label"}
            key = "spectrogram" if "spectrogram" in inputs else "features"
            logits = self.model(inputs[key])
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            preds = np.argmax(probs, axis=1)
            all_preds.extend(preds)
            all_probs.extend(probs)
        return np.array(all_preds), np.array(all_probs)

    def load_checkpoint(self, path: str) -> None:
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        self.model.eval()
        logger.info(f"Checkpoint loaded from {path}")
