"""
Audio Training Orchestrator — AI-Powered Lie Detection System
Trains classical ML and deep learning models on audio features.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from src.audio.feature_extractor import AudioFeaturePipeline, load_audio
from src.audio.classical_models import AudioRandomForest, AudioXGBoost, AudioSVM
from src.audio.deep_models import (
    AudioCNN, AudioLSTM, CNNLSTMHybrid,
    SpectrogramDataset, FeatureSequenceDataset, DeepAudioTrainer
)
from src.utils.logger import logger
from src.utils.config import config
from src.utils.helpers import set_seed, save_json


class AudioTrainer:
    """
    Orchestrates audio model training pipeline.

    Pipeline:
    1. Load audio files and extract features (163-dim vectors + mel spectrograms)
    2. Train classical models (RF, XGBoost, SVM)
    3. Train deep models (CNN, LSTM, CNN-LSTM)
    4. Save best models and evaluation results
    """

    def __init__(
        self,
        output_dir: str = "models/audio",
        results_dir: str = "reports",
        seed: int = 42,
        sample_rate: int = 22050,
    ):
        set_seed(seed)
        self.output_dir = Path(output_dir)
        self.results_dir = Path(results_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.pipeline = AudioFeaturePipeline(sample_rate=sample_rate)
        self.results: Dict = {}

    def load_features(
        self,
        df: pd.DataFrame,
        audio_path_col: str = "filepath",
        label_col: str = "stress_label",
        val_size: float = 0.15,
        test_size: float = 0.15,
    ) -> Dict:
        """
        Load audio files, extract features, and split into train/val/test.

        Returns:
            Dict with split feature matrices and mel spectrograms.
        """
        paths = df[audio_path_col].tolist()
        labels = df[label_col].astype(int).tolist()

        logger.info(f"Extracting features from {len(paths)} audio files...")

        feature_vectors = []
        mel_spectrograms = []
        valid_labels = []

        for path, label in zip(paths, labels):
            try:
                y, _ = load_audio(path, target_sr=self.pipeline.sr)
                features = self.pipeline.extract_from_array(y)
                feature_vectors.append(features["feature_vector"])
                mel_spectrograms.append(features["mel_spectrogram"])
                valid_labels.append(label)
            except Exception as e:
                logger.warning(f"Skipping {path}: {e}")

        X = np.vstack(feature_vectors)  # (N, 163)
        labels = valid_labels

        logger.info(f"Extracted features for {len(X)} files. Shape: {X.shape}")

        # Split
        indices = list(range(len(X)))
        idx_trainval, idx_test = train_test_split(indices, test_size=test_size, stratify=labels, random_state=42)
        val_ratio = val_size / (1 - test_size)
        labels_trainval = [labels[i] for i in idx_trainval]
        idx_train, idx_val = train_test_split(idx_trainval, test_size=val_ratio, stratify=labels_trainval, random_state=42)

        return {
            "X_train": X[idx_train],
            "X_val": X[idx_val],
            "X_test": X[np.array(idx_test)],
            "y_train": np.array([labels[i] for i in idx_train]),
            "y_val": np.array([labels[i] for i in idx_val]),
            "y_test": np.array([labels[i] for i in idx_test]),
            "mel_train": [mel_spectrograms[i] for i in idx_train],
            "mel_val": [mel_spectrograms[i] for i in idx_val],
            "mel_test": [mel_spectrograms[i] for i in idx_test],
        }

    def train_classical(self, splits: Dict) -> Dict:
        """Train RF, XGBoost, SVM on feature vectors."""
        results = {}
        feature_names = AudioFeaturePipeline.feature_names()

        # Random Forest
        rf = AudioRandomForest()
        rf.fit(splits["X_train"], splits["y_train"])
        results["random_forest"] = rf.evaluate(splits["X_test"], splits["y_test"], feature_names)
        rf.save(str(self.output_dir / "random_forest.pkl"))
        logger.info(f"Random Forest AUC: {results['random_forest']['roc_auc']:.4f}")

        # XGBoost
        xgb = AudioXGBoost()
        xgb.fit(splits["X_train"], splits["y_train"], splits["X_val"], splits["y_val"])
        results["xgboost"] = xgb.evaluate(splits["X_test"], splits["y_test"])
        xgb.save(str(self.output_dir / "xgboost.pkl"))
        logger.info(f"XGBoost AUC: {results['xgboost']['roc_auc']:.4f}")

        # SVM
        svm = AudioSVM()
        svm.fit(splits["X_train"], splits["y_train"])
        results["svm"] = svm.evaluate(splits["X_test"], splits["y_test"])
        svm.save(str(self.output_dir / "svm.pkl"))
        logger.info(f"SVM AUC: {results['svm']['roc_auc']:.4f}")

        return results

    def train_deep(self, splits: Dict, num_epochs: int = 30, batch_size: int = 32) -> Dict:
        """Train CNN, LSTM, CNN-LSTM on spectrograms and feature sequences."""
        results = {}

        # ── CNN on Spectrograms ──────────────────────────────
        logger.info("Training Audio CNN on mel spectrograms...")
        train_ds = SpectrogramDataset(splits["mel_train"], list(splits["y_train"]))
        val_ds = SpectrogramDataset(splits["mel_val"], list(splits["y_val"]))
        test_ds = SpectrogramDataset(splits["mel_test"], list(splits["y_test"]))

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size)
        test_loader = DataLoader(test_ds, batch_size=batch_size)

        cnn_model = AudioCNN(num_classes=2)
        cnn_trainer = DeepAudioTrainer(cnn_model)
        cnn_history = cnn_trainer.train(train_loader, val_loader, num_epochs=num_epochs,
                                         output_dir=str(self.output_dir / "cnn"))
        preds, probas = cnn_trainer.predict(test_loader)
        from sklearn.metrics import classification_report, roc_auc_score
        results["cnn"] = {
            "history": cnn_history,
            "test_report": classification_report(splits["y_test"], preds, output_dict=True, zero_division=0),
            "test_roc_auc": roc_auc_score(splits["y_test"], probas[:, 1]) if len(set(splits["y_test"])) > 1 else 0.5,
        }
        logger.info(f"CNN AUC: {results['cnn']['test_roc_auc']:.4f}")

        # ── LSTM on MFCC sequences ────────────────────────────
        logger.info("Training Audio LSTM on feature sequences...")
        # Use MFCC (first 40 features) as sequence — reshape to (T, 40)
        # For simplicity, we use the feature vector split into chunks
        T = 20
        F = 40

        def reshape_for_lstm(X: np.ndarray) -> np.ndarray:
            return X[:, :T*F].reshape(-1, T, F)

        lstm_train = reshape_for_lstm(splits["X_train"])
        lstm_val = reshape_for_lstm(splits["X_val"])
        lstm_test = reshape_for_lstm(splits["X_test"])

        lstm_train_ds = FeatureSequenceDataset(lstm_train, list(splits["y_train"]))
        lstm_val_ds = FeatureSequenceDataset(lstm_val, list(splits["y_val"]))
        lstm_test_ds = FeatureSequenceDataset(lstm_test, list(splits["y_test"]))

        lstm_model = AudioLSTM(input_size=F)
        lstm_trainer = DeepAudioTrainer(lstm_model)
        lstm_history = lstm_trainer.train(
            DataLoader(lstm_train_ds, batch_size=batch_size, shuffle=True),
            DataLoader(lstm_val_ds, batch_size=batch_size),
            num_epochs=num_epochs, output_dir=str(self.output_dir / "lstm")
        )
        preds, probas = lstm_trainer.predict(DataLoader(lstm_test_ds, batch_size=batch_size))
        results["lstm"] = {
            "history": lstm_history,
            "test_report": classification_report(splits["y_test"], preds, output_dict=True, zero_division=0),
            "test_roc_auc": roc_auc_score(splits["y_test"], probas[:, 1]) if len(set(splits["y_test"])) > 1 else 0.5,
        }
        logger.info(f"LSTM AUC: {results['lstm']['test_roc_auc']:.4f}")

        return results

    def run_full_pipeline(self, df: pd.DataFrame, train_deep: bool = True) -> Dict:
        """Execute full audio training pipeline."""
        splits = self.load_features(df)

        logger.info("=== Audio Phase 1: Classical Models ===")
        classical_results = self.train_classical(splits)
        self.results.update(classical_results)

        if train_deep:
            logger.info("=== Audio Phase 2: Deep Learning Models ===")
            deep_results = self.train_deep(splits)
            self.results.update(deep_results)

        save_json(self.results, str(self.results_dir / "audio_results.json"))
        logger.info(f"Audio results saved to {self.results_dir / 'audio_results.json'}")
        return self.results
