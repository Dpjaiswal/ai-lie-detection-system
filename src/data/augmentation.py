"""
Audio Data Augmentation — AI-Powered Lie Detection System
Augmentation strategies to expand limited audio datasets.
"""
from __future__ import annotations

import random
from typing import List, Optional, Tuple

import numpy as np
import librosa

from src.utils.logger import logger


class AudioAugmenter:
    """
    Audio augmentation pipeline for training data expansion.

    Augmentation types:
    - Time stretching: slow down or speed up speech without changing pitch
    - Pitch shifting: raise/lower fundamental frequency
    - Noise injection: add Gaussian noise at controlled SNR
    - Gain variation: random amplitude scaling
    - Time masking: random silence insertions (SpecAugment-style)
    - Frequency masking: zero out frequency bands (SpecAugment-style)

    Usage:
        augmenter = AudioAugmenter(seed=42)
        y_aug = augmenter.augment(y, sr=22050)
    """

    def __init__(
        self,
        time_stretch_range: Tuple[float, float] = (0.8, 1.2),
        pitch_shift_range: Tuple[int, int] = (-2, 2),
        noise_level: float = 0.005,
        gain_range: Tuple[float, float] = (0.8, 1.2),
        p_time_stretch: float = 0.4,
        p_pitch_shift: float = 0.4,
        p_noise: float = 0.5,
        p_gain: float = 0.3,
        seed: Optional[int] = None,
    ):
        self.time_stretch_range = time_stretch_range
        self.pitch_shift_range = pitch_shift_range
        self.noise_level = noise_level
        self.gain_range = gain_range
        self.p_time_stretch = p_time_stretch
        self.p_pitch_shift = p_pitch_shift
        self.p_noise = p_noise
        self.p_gain = p_gain
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

    def time_stretch(self, y: np.ndarray) -> np.ndarray:
        """Randomly stretch or compress audio tempo."""
        rate = random.uniform(*self.time_stretch_range)
        return librosa.effects.time_stretch(y, rate=rate)

    def pitch_shift(self, y: np.ndarray, sr: int) -> np.ndarray:
        """Randomly shift pitch by n semitones."""
        n_steps = random.randint(*self.pitch_shift_range)
        return librosa.effects.pitch_shift(y, sr=sr, n_steps=n_steps)

    def add_noise(self, y: np.ndarray) -> np.ndarray:
        """Add Gaussian white noise."""
        noise = np.random.normal(0, self.noise_level, len(y))
        return np.clip(y + noise, -1.0, 1.0)

    def random_gain(self, y: np.ndarray) -> np.ndarray:
        """Apply random gain scaling."""
        gain = random.uniform(*self.gain_range)
        return np.clip(y * gain, -1.0, 1.0)

    def augment(self, y: np.ndarray, sr: int = 22050) -> np.ndarray:
        """Apply a random combination of augmentations."""
        if random.random() < self.p_time_stretch:
            try:
                y = self.time_stretch(y)
            except Exception as e:
                logger.debug(f"Time stretch failed: {e}")

        if random.random() < self.p_pitch_shift:
            try:
                y = self.pitch_shift(y, sr)
            except Exception as e:
                logger.debug(f"Pitch shift failed: {e}")

        if random.random() < self.p_noise:
            y = self.add_noise(y)

        if random.random() < self.p_gain:
            y = self.random_gain(y)

        return y

    def augment_batch(
        self,
        waveforms: List[np.ndarray],
        sr: int = 22050,
        n_augmentations: int = 2,
    ) -> Tuple[List[np.ndarray], List[int]]:
        """
        Augment a batch of waveforms.

        Returns:
            (augmented_waveforms, original_indices) — indices map augmented back to original
        """
        augmented = list(waveforms)
        indices = list(range(len(waveforms)))

        for orig_idx, y in enumerate(waveforms):
            for _ in range(n_augmentations):
                try:
                    y_aug = self.augment(y, sr)
                    augmented.append(y_aug)
                    indices.append(orig_idx)
                except Exception as e:
                    logger.warning(f"Augmentation failed for sample {orig_idx}: {e}")

        logger.info(f"Augmented {len(waveforms)} → {len(augmented)} samples "
                    f"({n_augmentations}x expansion)")
        return augmented, indices
