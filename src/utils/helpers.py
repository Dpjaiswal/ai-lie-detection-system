"""
Helpers — AI-Powered Lie Detection System
Common utility functions shared across modules.
"""
from __future__ import annotations

import hashlib
import json
import random
import time
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, TypeVar, Union

import numpy as np

F = TypeVar("F", bound=Callable[..., Any])


# ── Reproducibility ───────────────────────────────────────────
def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except ImportError:
        pass


# ── Timing ────────────────────────────────────────────────────
def timer(func: F) -> F:
    """Decorator to measure and log function execution time."""
    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        print(f"[timer] {func.__qualname__} executed in {elapsed:.4f}s")
        return result
    return wrapper  # type: ignore


# ── File utilities ────────────────────────────────────────────
def ensure_dir(path: Union[str, Path]) -> Path:
    """Create directory if it doesn't exist, return Path."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def file_hash(path: Union[str, Path], algorithm: str = "md5") -> str:
    """Compute hash of a file for integrity checks."""
    h = hashlib.new(algorithm)
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def load_json(path: Union[str, Path]) -> Any:
    """Load JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: Any, path: Union[str, Path], indent: int = 2) -> None:
    """Save data to JSON file."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, ensure_ascii=False, default=str)


# ── Data utilities ────────────────────────────────────────────
def flatten_list(nested: List[List[Any]]) -> List[Any]:
    """Flatten a list of lists."""
    return [item for sublist in nested for item in sublist]


def chunk_list(lst: List[Any], size: int) -> List[List[Any]]:
    """Split a list into fixed-size chunks."""
    return [lst[i : i + size] for i in range(0, len(lst), size)]


def class_weights_from_labels(labels: List[int]) -> Dict[int, float]:
    """Compute inverse-frequency class weights for imbalanced datasets."""
    from collections import Counter
    counts = Counter(labels)
    total = len(labels)
    n_classes = len(counts)
    return {
        cls: total / (n_classes * count)
        for cls, count in counts.items()
    }


# ── Text utilities ────────────────────────────────────────────
def truncate_text(text: str, max_chars: int = 1000) -> str:
    """Truncate text to max_chars with ellipsis."""
    return text[:max_chars] + "..." if len(text) > max_chars else text


def word_count(text: str) -> int:
    """Count words in text."""
    return len(text.split())


# ── Prediction utilities ──────────────────────────────────────
def softmax(logits: np.ndarray) -> np.ndarray:
    """Numerically stable softmax."""
    e = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def label_to_str(label: int) -> str:
    """Convert binary label to human-readable string."""
    return "lie" if label == 1 else "truth"


def confidence_to_verdict(lie_prob: float) -> str:
    """Convert lie probability to human-readable verdict string."""
    if lie_prob >= 0.75:
        return "likely_lie"
    elif lie_prob >= 0.55:
        return "possibly_lie"
    elif lie_prob >= 0.45:
        return "uncertain"
    else:
        return "likely_truth"


# ── Formatting ────────────────────────────────────────────────
def format_confidence(score: float) -> str:
    """Format confidence score as percentage string."""
    return f"{score * 100:.1f}%"


def seconds_to_hms(seconds: float) -> str:
    """Convert seconds to HH:MM:SS string."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"
