"""
Audio Feature Extractor — AI-Powered Lie Detection System
Extracts spectral, voice stress, and emotional features from audio.
Uses Librosa, SoundFile, NumPy, and Parselmouth (Praat).
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import librosa
import soundfile as sf

from src.utils.logger import logger
from src.utils.config import config


# ──────────────────────────────────────────────────────────────
# Audio Loader
# ──────────────────────────────────────────────────────────────

def load_audio(
    path: str,
    target_sr: int = 22050,
    mono: bool = True,
    max_duration: float = 30.0,
    normalize: bool = True,
) -> Tuple[np.ndarray, int]:
    """
    Load audio file with resampling and normalization.

    Args:
        path: Path to audio file (WAV, MP3, FLAC, OGG, M4A)
        target_sr: Target sample rate
        mono: Convert to mono if True
        max_duration: Clip to max_duration seconds
        normalize: Peak normalize to [-1, 1]

    Returns:
        (waveform array, sample_rate)
    """
    y, sr = librosa.load(path, sr=target_sr, mono=mono, duration=max_duration)
    if normalize and y.max() != 0:
        y = y / np.abs(y).max()
    return y, sr


# ──────────────────────────────────────────────────────────────
# Spectral Feature Extractor
# ──────────────────────────────────────────────────────────────

class SpectralFeatureExtractor:
    """
    Extracts MFCC, Chroma, Spectral Contrast, and Mel Spectrogram features.

    Feature dimensions:
    - MFCC: (n_mfcc,) — captures vocal tract shape
    - Chroma: (12,) — pitch class profile
    - Spectral Contrast: (7,) — valley-peak contrast per octave band
    - Mel Spectrogram: (n_mels, T) — full time-frequency representation
    - ZCR: (1,) — zero crossing rate (roughness proxy)
    - Spectral Centroid: (1,) — brightness
    - Spectral Bandwidth: (1,) — frequency spread
    - Spectral Rolloff: (1,) — spectral energy cutoff
    """

    def __init__(
        self,
        sample_rate: int = 22050,
        n_mfcc: int = 40,
        n_chroma: int = 12,
        n_mels: int = 128,
        hop_length: int = 512,
        win_length: int = 2048,
        fmax: float = 8000.0,
    ):
        cfg = config.get_feature_config()
        self.sr = cfg.get("sample_rate", sample_rate)
        self.n_mfcc = cfg.get("n_mfcc", n_mfcc)
        self.n_chroma = cfg.get("n_chroma", n_chroma)
        self.n_mels = cfg.get("n_mels", n_mels)
        self.hop_length = cfg.get("hop_length", hop_length)
        self.win_length = cfg.get("win_length", win_length)
        self.fmax = cfg.get("fmax", fmax)

    def extract_mfcc(
        self, y: np.ndarray, include_delta: bool = True, include_delta2: bool = True
    ) -> np.ndarray:
        """
        Extract MFCC + Delta + Delta-Delta coefficients.
        Returns flattened mean vector of shape (n_mfcc * 3,) if deltas included.
        """
        mfcc = librosa.feature.mfcc(
            y=y, sr=self.sr, n_mfcc=self.n_mfcc,
            hop_length=self.hop_length, win_length=self.win_length,
            fmax=self.fmax,
        )
        features = [mfcc.mean(axis=1)]
        if include_delta:
            features.append(librosa.feature.delta(mfcc).mean(axis=1))
        if include_delta2:
            features.append(librosa.feature.delta(mfcc, order=2).mean(axis=1))
        return np.concatenate(features)

    def extract_chroma(self, y: np.ndarray) -> np.ndarray:
        """Extract Chroma features (pitch class) — mean over time."""
        chroma = librosa.feature.chroma_stft(
            y=y, sr=self.sr, n_chroma=self.n_chroma,
            hop_length=self.hop_length,
        )
        return chroma.mean(axis=1)  # shape: (12,)

    def extract_spectral_contrast(self, y: np.ndarray, n_bands: int = 6) -> np.ndarray:
        """Extract Spectral Contrast — mean over time."""
        contrast = librosa.feature.spectral_contrast(
            y=y, sr=self.sr, n_bands=n_bands, hop_length=self.hop_length
        )
        return contrast.mean(axis=1)  # shape: (n_bands+1,)

    def extract_mel_spectrogram(self, y: np.ndarray) -> np.ndarray:
        """Extract Mel Spectrogram — returns (n_mels, T) for CNN input."""
        S = librosa.feature.melspectrogram(
            y=y, sr=self.sr, n_mels=self.n_mels,
            hop_length=self.hop_length, win_length=self.win_length,
            fmax=self.fmax,
        )
        return librosa.power_to_db(S, ref=np.max)

    def extract_statistical_features(self, y: np.ndarray) -> np.ndarray:
        """
        Extract zero-crossing rate, spectral centroid, bandwidth, rolloff.
        Returns vector of shape (4,).
        """
        zcr = librosa.feature.zero_crossing_rate(y, hop_length=self.hop_length).mean()
        centroid = librosa.feature.spectral_centroid(
            y=y, sr=self.sr, hop_length=self.hop_length
        ).mean()
        bandwidth = librosa.feature.spectral_bandwidth(
            y=y, sr=self.sr, hop_length=self.hop_length
        ).mean()
        rolloff = librosa.feature.spectral_rolloff(
            y=y, sr=self.sr, hop_length=self.hop_length
        ).mean()
        return np.array([zcr, centroid, bandwidth, rolloff], dtype=np.float32)

    def extract_all(self, y: np.ndarray) -> np.ndarray:
        """
        Extract all spectral features and concatenate into a single vector.
        Total size: 40*3 + 12 + 7 + 4 = 143 features
        """
        mfcc = self.extract_mfcc(y)            # 120
        chroma = self.extract_chroma(y)         # 12
        contrast = self.extract_spectral_contrast(y)  # 7
        stat = self.extract_statistical_features(y)    # 4
        return np.concatenate([mfcc, chroma, contrast, stat]).astype(np.float32)


# ──────────────────────────────────────────────────────────────
# Voice Stress Feature Extractor (Praat-based)
# ──────────────────────────────────────────────────────────────

class VoiceStressExtractor:
    """
    Extracts prosodic and voice quality features using Praat via Parselmouth.

    Features:
    - Pitch (F0): mean, std, min, max, range
    - Jitter: cycle-to-cycle period variation (voice roughness)
    - Shimmer: cycle-to-cycle amplitude variation
    - HNR: Harmonics-to-Noise Ratio (breathiness)
    - RMS Energy: loudness proxy
    - Speaking rate: syllables per second estimate
    - Pause ratio: proportion of silent frames

    These features are well-studied stress indicators in forensic phonetics.
    """

    def __init__(self, sample_rate: int = 22050):
        self.sr = sample_rate

    def _get_parselmouth_sound(self, y: np.ndarray):
        """Convert numpy array to Parselmouth Sound object."""
        try:
            import parselmouth
            return parselmouth.Sound(y, self.sr)
        except ImportError:
            logger.warning("Parselmouth not installed. Pitch/Jitter/Shimmer unavailable.")
            return None

    def extract_pitch_features(self, y: np.ndarray) -> np.ndarray:
        """
        Extract pitch (F0) statistics.
        Returns: [mean_f0, std_f0, min_f0, max_f0, range_f0, voiced_fraction]
        """
        sound = self._get_parselmouth_sound(y)
        if sound is None:
            return np.zeros(6, dtype=np.float32)

        import parselmouth
        from parselmouth.praat import call

        pitch = call(sound, "To Pitch", 0.0, 75, 600)
        pitch_values = pitch.selected_array["frequency"]
        voiced = pitch_values[pitch_values > 0]

        if len(voiced) == 0:
            return np.zeros(6, dtype=np.float32)

        return np.array([
            voiced.mean(),
            voiced.std(),
            voiced.min(),
            voiced.max(),
            voiced.max() - voiced.min(),
            len(voiced) / max(len(pitch_values), 1),
        ], dtype=np.float32)

    def extract_jitter(self, y: np.ndarray) -> np.ndarray:
        """
        Extract jitter (period perturbation) features.
        Returns: [jitter_local, jitter_local_abs, jitter_rap, jitter_ppq5]
        """
        sound = self._get_parselmouth_sound(y)
        if sound is None:
            return np.zeros(4, dtype=np.float32)

        from parselmouth.praat import call
        try:
            point_process = call(sound, "To PointProcess (periodic, cc)", 75, 600)
            jitter_local = call(point_process, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
            jitter_local_abs = call(point_process, "Get jitter (local, absolute)", 0, 0, 0.0001, 0.02, 1.3)
            jitter_rap = call(point_process, "Get jitter (rap)", 0, 0, 0.0001, 0.02, 1.3)
            jitter_ppq5 = call(point_process, "Get jitter (ppq5)", 0, 0, 0.0001, 0.02, 1.3)
            return np.array([jitter_local, jitter_local_abs, jitter_rap, jitter_ppq5], dtype=np.float32)
        except Exception as e:
            logger.debug(f"Jitter extraction failed: {e}")
            return np.zeros(4, dtype=np.float32)

    def extract_shimmer(self, y: np.ndarray) -> np.ndarray:
        """
        Extract shimmer (amplitude perturbation) features.
        Returns: [shimmer_local, shimmer_local_db, shimmer_apq3, shimmer_apq5]
        """
        sound = self._get_parselmouth_sound(y)
        if sound is None:
            return np.zeros(4, dtype=np.float32)

        from parselmouth.praat import call
        try:
            point_process = call(sound, "To PointProcess (periodic, cc)", 75, 600)
            shimmer_local = call([sound, point_process], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
            shimmer_db = call([sound, point_process], "Get shimmer (local_dB)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
            shimmer_apq3 = call([sound, point_process], "Get shimmer (apq3)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
            shimmer_apq5 = call([sound, point_process], "Get shimmer (apq5)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
            return np.array([shimmer_local, shimmer_db, shimmer_apq3, shimmer_apq5], dtype=np.float32)
        except Exception as e:
            logger.debug(f"Shimmer extraction failed: {e}")
            return np.zeros(4, dtype=np.float32)

    def extract_rms_energy(self, y: np.ndarray, hop_length: int = 512) -> np.ndarray:
        """
        Extract RMS energy statistics.
        Returns: [mean_rms, std_rms, max_rms, min_rms]
        """
        rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]
        return np.array([
            rms.mean(), rms.std(), rms.max(), rms.min()
        ], dtype=np.float32)

    def extract_speaking_rate(self, y: np.ndarray) -> np.ndarray:
        """
        Estimate speaking rate via onset detection.
        Returns: [onset_rate, pause_ratio]
        """
        hop_length = 512
        onsets = librosa.onset.onset_detect(y=y, sr=self.sr, hop_length=hop_length)
        duration = len(y) / self.sr
        onset_rate = len(onsets) / max(duration, 1e-6)

        # Pause ratio using energy threshold
        rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]
        threshold = 0.01 * rms.max()
        pause_frames = np.sum(rms < threshold)
        pause_ratio = pause_frames / max(len(rms), 1)

        return np.array([onset_rate, pause_ratio], dtype=np.float32)

    def extract_all(self, y: np.ndarray) -> np.ndarray:
        """
        Extract all voice stress features.
        Total: 6 (pitch) + 4 (jitter) + 4 (shimmer) + 4 (rms) + 2 (rate) = 20 features
        """
        pitch = self.extract_pitch_features(y)
        jitter = self.extract_jitter(y)
        shimmer = self.extract_shimmer(y)
        rms = self.extract_rms_energy(y)
        rate = self.extract_speaking_rate(y)
        return np.concatenate([pitch, jitter, shimmer, rms, rate]).astype(np.float32)


# ──────────────────────────────────────────────────────────────
# Combined Audio Feature Pipeline
# ──────────────────────────────────────────────────────────────

class AudioFeaturePipeline:
    """
    Complete audio feature extraction pipeline combining spectral and stress features.
    Total feature vector: 143 (spectral) + 20 (stress) = 163 features
    """

    def __init__(self, sample_rate: int = 22050):
        self.sr = sample_rate
        self.spectral = SpectralFeatureExtractor(sample_rate=sample_rate)
        self.stress = VoiceStressExtractor(sample_rate=sample_rate)

    def extract_from_array(self, y: np.ndarray) -> Dict[str, np.ndarray]:
        """Extract all features from a waveform array."""
        spectral_features = self.spectral.extract_all(y)
        stress_features = self.stress.extract_all(y)
        mel_spec = self.spectral.extract_mel_spectrogram(y)
        combined = np.concatenate([spectral_features, stress_features])
        return {
            "feature_vector": combined,        # (163,) — for classical ML
            "spectral_features": spectral_features,  # (143,)
            "stress_features": stress_features,      # (20,)
            "mel_spectrogram": mel_spec,        # (n_mels, T) — for CNN
        }

    def extract_from_file(self, path: str) -> Dict[str, np.ndarray]:
        """Load audio file and extract all features."""
        y, _ = load_audio(path, target_sr=self.sr)
        return self.extract_from_array(y)

    def extract_batch(self, paths: List[str]) -> np.ndarray:
        """
        Extract feature vectors for a list of audio files.
        Returns (N, 163) array for classical ML models.
        """
        vectors = []
        for path in paths:
            try:
                features = self.extract_from_file(path)
                vectors.append(features["feature_vector"])
            except Exception as e:
                logger.warning(f"Failed to extract features from {path}: {e}")
                vectors.append(np.zeros(163, dtype=np.float32))
        return np.vstack(vectors)

    @staticmethod
    def feature_names() -> List[str]:
        """Return ordered feature names for interpretability."""
        names = []
        # MFCC (120)
        for i in range(40):
            names.extend([f"mfcc_{i}", f"mfcc_delta_{i}", f"mfcc_delta2_{i}"])
        # Chroma (12)
        names.extend([f"chroma_{i}" for i in range(12)])
        # Spectral Contrast (7)
        names.extend([f"spec_contrast_{i}" for i in range(7)])
        # Statistical (4)
        names.extend(["zcr", "spectral_centroid", "spectral_bandwidth", "spectral_rolloff"])
        # Pitch (6)
        names.extend(["pitch_mean", "pitch_std", "pitch_min", "pitch_max", "pitch_range", "voiced_fraction"])
        # Jitter (4)
        names.extend(["jitter_local", "jitter_local_abs", "jitter_rap", "jitter_ppq5"])
        # Shimmer (4)
        names.extend(["shimmer_local", "shimmer_db", "shimmer_apq3", "shimmer_apq5"])
        # RMS (4)
        names.extend(["rms_mean", "rms_std", "rms_max", "rms_min"])
        # Rate (2)
        names.extend(["onset_rate", "pause_ratio"])
        return names
