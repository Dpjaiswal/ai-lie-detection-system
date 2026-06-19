"""
Dataset Loader — AI-Powered Lie Detection System
Loaders for LIAR, RAVDESS, CREMA-D, and custom datasets.
"""
from __future__ import annotations

import os
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
from tqdm import tqdm

from src.utils.logger import logger
from src.utils.config import config


# ──────────────────────────────────────────────────────────────
# LIAR Dataset Loader (Text Deception)
# ──────────────────────────────────────────────────────────────

LIAR_COLUMNS = [
    "id", "label", "statement", "subject", "speaker",
    "speaker_job", "state", "party_affiliation",
    "barely_true_count", "false_count", "half_true_count",
    "mostly_true_count", "pants_on_fire_count", "context"
]

LIAR_LABEL_MAP = {
    "true": 0,
    "mostly-true": 0,
    "half-true": 0,
    "barely-true": 1,
    "false": 1,
    "pants-fire": 1,
}


class LIARDatasetLoader:
    """
    Loader for the LIAR dataset (Wang, 2017).
    Source: https://www.cs.ucsb.edu/~william/data/liar_dataset.zip
    Labels: 6-class -> mapped to binary (truth=0, lie=1)
    Size: ~12,836 statements from PolitiFact
    """

    DOWNLOAD_URL = "https://www.cs.ucsb.edu/~william/data/liar_dataset.zip"

    def __init__(self, data_dir: str = "data/raw/liar"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def download(self) -> None:
        """Download and extract the LIAR dataset."""
        zip_path = self.data_dir / "liar_dataset.zip"
        if zip_path.exists():
            logger.info("LIAR dataset already downloaded.")
            return

        logger.info(f"Downloading LIAR dataset from {self.DOWNLOAD_URL}")
        response = requests.get(self.DOWNLOAD_URL, stream=True)
        total = int(response.headers.get("content-length", 0))

        with open(zip_path, "wb") as f, tqdm(total=total, unit="B", unit_scale=True) as bar:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                bar.update(len(chunk))

        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(self.data_dir)
        logger.info(f"LIAR dataset extracted to {self.data_dir}")

    def load_split(self, split: str = "train") -> pd.DataFrame:
        """
        Load a split of LIAR dataset.

        Args:
            split: 'train', 'val', or 'test'

        Returns:
            DataFrame with columns: text, label, binary_label, speaker, party
        """
        file_map = {"train": "train.tsv", "val": "valid.tsv", "test": "test.tsv"}
        if split not in file_map:
            raise ValueError(f"split must be one of {list(file_map.keys())}")

        path = self.data_dir / file_map[split]
        if not path.exists():
            raise FileNotFoundError(
                f"LIAR {split} file not found at {path}. Run download() first."
            )

        df = pd.read_csv(path, sep="\t", header=None, names=LIAR_COLUMNS)
        df["text"] = df["statement"]
        df["raw_label"] = df["label"]
        df["label"] = df["label"].map(LIAR_LABEL_MAP).fillna(1).astype(int)
        df["split"] = split

        logger.info(f"Loaded LIAR {split}: {len(df)} samples, "
                    f"label distribution: {df['label'].value_counts().to_dict()}")
        return df[["text", "label", "raw_label", "speaker", "party_affiliation", "split"]]

    def load_all(self) -> pd.DataFrame:
        """Load and concatenate all splits."""
        dfs = [self.load_split(s) for s in ["train", "val", "test"]]
        return pd.concat(dfs, ignore_index=True)

    @staticmethod
    def get_info() -> Dict:
        return {
            "name": "LIAR Dataset",
            "authors": "William Yang Wang (2017)",
            "size": 12836,
            "classes": 6,
            "binary_mapping": LIAR_LABEL_MAP,
            "language": "English",
            "domain": "Political statements",
            "pros": [
                "Large, well-curated benchmark",
                "Multi-class labels with speaker metadata",
                "Widely used in academic research",
            ],
            "cons": [
                "Political domain only (limited generalization)",
                "Short statements (avg ~18 words)",
                "No audio modality",
            ],
            "url": "https://www.cs.ucsb.edu/~william/data/liar_dataset.zip",
        }


# ──────────────────────────────────────────────────────────────
# RAVDESS Dataset Loader (Audio Emotion/Stress)
# ──────────────────────────────────────────────────────────────

class RAVDESSLoader:
    """
    Loader for RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song).
    Source: https://zenodo.org/record/1188976
    24 professional actors, 8 emotions, 2452 audio files.
    Stress/deception proxy: emotions mapped to stress indicators.
    """

    EMOTION_MAP = {
        "01": "neutral", "02": "calm", "03": "happy",
        "04": "sad", "05": "angry", "06": "fearful",
        "07": "disgust", "08": "surprised"
    }

    # Deception proxy: high-arousal negative emotions → stress=1
    STRESS_MAP = {
        "01": 0, "02": 0, "03": 0, "04": 1,
        "05": 1, "06": 1, "07": 1, "08": 0
    }

    def __init__(self, data_dir: str = "data/raw/ravdess"):
        self.data_dir = Path(data_dir)

    def parse_filename(self, filepath: Path) -> Dict:
        """Parse RAVDESS filename convention into metadata."""
        parts = filepath.stem.split("-")
        if len(parts) != 7:
            return {}
        return {
            "filepath": str(filepath),
            "modality": parts[0],
            "vocal_channel": parts[1],
            "emotion_code": parts[2],
            "emotion": self.EMOTION_MAP.get(parts[2], "unknown"),
            "intensity": parts[3],
            "statement": parts[4],
            "repetition": parts[5],
            "actor": parts[6],
            "stress_label": self.STRESS_MAP.get(parts[2], 0),
        }

    def load(self) -> pd.DataFrame:
        """Load all RAVDESS audio metadata into a DataFrame."""
        records = []
        for wav in self.data_dir.rglob("*.wav"):
            meta = self.parse_filename(wav)
            if meta:
                records.append(meta)

        df = pd.DataFrame(records)
        if df.empty:
            logger.warning(f"No RAVDESS files found in {self.data_dir}")
            return df

        logger.info(f"Loaded RAVDESS: {len(df)} files, "
                    f"stress distribution: {df['stress_label'].value_counts().to_dict()}")
        return df

    @staticmethod
    def get_info() -> Dict:
        return {
            "name": "RAVDESS",
            "full_name": "Ryerson Audio-Visual Database of Emotional Speech and Song",
            "size": 2452,
            "actors": 24,
            "emotions": 8,
            "audio_length": "3–5 seconds per clip",
            "sample_rate": 48000,
            "format": "WAV (16-bit, 48kHz)",
            "suitability": "Good for stress/anxiety feature extraction as deception proxy",
            "download": "https://zenodo.org/record/1188976",
            "pros": ["High quality audio", "Professional actors", "Balanced emotions"],
            "cons": ["Small dataset", "Acted speech (not real deception)", "Limited spontaneity"],
        }


# ──────────────────────────────────────────────────────────────
# CREMA-D Dataset Loader
# ──────────────────────────────────────────────────────────────

class CREMADLoader:
    """
    Loader for CREMA-D (Crowd-sourced Emotional Multimodal Actors Dataset).
    7442 clips from 91 actors (48 male, 43 female).
    6 emotions: Anger, Disgust, Fear, Happy, Neutral, Sad.
    """

    EMOTION_MAP = {
        "ANG": "angry", "DIS": "disgust", "FEA": "fear",
        "HAP": "happy", "NEU": "neutral", "SAD": "sad"
    }
    STRESS_MAP = {"ANG": 1, "DIS": 1, "FEA": 1, "HAP": 0, "NEU": 0, "SAD": 1}

    def __init__(self, data_dir: str = "data/raw/cremad"):
        self.data_dir = Path(data_dir)

    def load(self) -> pd.DataFrame:
        """Load CREMA-D audio metadata."""
        records = []
        for wav in self.data_dir.rglob("*.wav"):
            parts = wav.stem.split("_")
            if len(parts) >= 3:
                emotion_code = parts[2]
                records.append({
                    "filepath": str(wav),
                    "actor_id": parts[0],
                    "sentence_id": parts[1],
                    "emotion_code": emotion_code,
                    "emotion": self.EMOTION_MAP.get(emotion_code, "unknown"),
                    "stress_label": self.STRESS_MAP.get(emotion_code, 0),
                })

        df = pd.DataFrame(records)
        logger.info(f"Loaded CREMA-D: {len(df)} files")
        return df

    @staticmethod
    def get_info() -> Dict:
        return {
            "name": "CREMA-D",
            "size": 7442,
            "actors": 91,
            "emotions": 6,
            "audio_length": "~3 seconds",
            "download": "https://github.com/CheyneyComputerScience/CREMA-D",
            "suitability": "Larger and more diverse than RAVDESS for stress modeling",
        }


# ──────────────────────────────────────────────────────────────
# Combined Dataset Factory
# ──────────────────────────────────────────────────────────────

class DatasetFactory:
    """Factory to load and merge multiple datasets."""

    @staticmethod
    def load_text_dataset(name: str = "liar", **kwargs) -> pd.DataFrame:
        loaders = {"liar": LIARDatasetLoader}
        if name not in loaders:
            raise ValueError(f"Unknown text dataset: {name}. Available: {list(loaders.keys())}")
        loader = loaders[name](**kwargs)
        if hasattr(loader, "download"):
            loader.download()
        return loader.load_all()

    @staticmethod
    def load_audio_dataset(name: str = "ravdess", **kwargs) -> pd.DataFrame:
        loaders = {"ravdess": RAVDESSLoader, "cremad": CREMADLoader}
        if name not in loaders:
            raise ValueError(f"Unknown audio dataset: {name}. Available: {list(loaders.keys())}")
        return loaders[name](**kwargs).load()

    @staticmethod
    def dataset_summary() -> Dict:
        return {
            "text_datasets": {
                "liar": LIARDatasetLoader.get_info(),
            },
            "audio_datasets": {
                "ravdess": RAVDESSLoader.get_info(),
                "cremad": CREMADLoader.get_info(),
            },
        }
