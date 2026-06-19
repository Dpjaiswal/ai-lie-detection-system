"""
NLP Feature Extractor — AI-Powered Lie Detection System
TF-IDF, Word2Vec, and Transformer-based embeddings.
"""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.utils.logger import logger
from src.utils.config import config


# ──────────────────────────────────────────────────────────────
# TF-IDF Feature Extractor
# ──────────────────────────────────────────────────────────────

class TFIDFExtractor:
    """
    TF-IDF feature extraction with configurable n-gram range.
    Produces sparse matrix features for classical ML models.

    Why TF-IDF: Fast, interpretable baseline. Term frequencies
    capture vocabulary-level deception signals (hedge words,
    negations, certainty markers).
    """

    def __init__(
        self,
        max_features: int = 50000,
        ngram_range: Tuple[int, int] = (1, 3),
        min_df: int = 2,
        max_df: float = 0.95,
        sublinear_tf: bool = True,
    ):
        cfg = config.get_nlp_config("baseline").get("tfidf", {})
        self.vectorizer = TfidfVectorizer(
            max_features=cfg.get("max_features", max_features),
            ngram_range=tuple(cfg.get("ngram_range", list(ngram_range))),
            min_df=cfg.get("min_df", min_df),
            max_df=cfg.get("max_df", max_df),
            sublinear_tf=cfg.get("sublinear_tf", sublinear_tf),
            analyzer="word",
        )
        self.is_fitted = False

    def fit(self, texts: List[str]) -> "TFIDFExtractor":
        """Fit the TF-IDF vectorizer on training texts."""
        self.vectorizer.fit(texts)
        self.is_fitted = True
        logger.info(f"TF-IDF fitted: vocabulary size = {len(self.vectorizer.vocabulary_):,}")
        return self

    def transform(self, texts: List[str]) -> np.ndarray:
        """Transform texts to TF-IDF feature matrix."""
        if not self.is_fitted:
            raise RuntimeError("Call fit() before transform()")
        return self.vectorizer.transform(texts)

    def fit_transform(self, texts: List[str]) -> np.ndarray:
        self.fit(texts)
        return self.transform(texts)

    def get_feature_names(self) -> List[str]:
        return self.vectorizer.get_feature_names_out().tolist()

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self.vectorizer, f)
        logger.info(f"TF-IDF vectorizer saved to {path}")

    def load(self, path: str) -> "TFIDFExtractor":
        with open(path, "rb") as f:
            self.vectorizer = pickle.load(f)
        self.is_fitted = True
        return self


# ──────────────────────────────────────────────────────────────
# Word2Vec Feature Extractor
# ──────────────────────────────────────────────────────────────

class Word2VecExtractor:
    """
    Word2Vec sentence embeddings via mean/max pooling.

    Why Word2Vec: Captures semantic similarity beyond term frequency.
    Effective with XGBoost for intermediate performance step.
    Uses Gensim's Word2Vec implementation.
    """

    def __init__(
        self,
        vector_size: int = 300,
        window: int = 5,
        min_count: int = 1,
        workers: int = 4,
        epochs: int = 10,
        pooling: str = "mean",  # mean | max | concat
    ):
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.workers = workers
        self.epochs = epochs
        self.pooling = pooling
        self.model = None
        self.is_fitted = False

    def fit(self, token_lists: List[List[str]]) -> "Word2VecExtractor":
        """Train Word2Vec on tokenized texts."""
        from gensim.models import Word2Vec
        self.model = Word2Vec(
            sentences=token_lists,
            vector_size=self.vector_size,
            window=self.window,
            min_count=self.min_count,
            workers=self.workers,
            epochs=self.epochs,
        )
        self.is_fitted = True
        logger.info(f"Word2Vec trained: vocab={len(self.model.wv):,}, dim={self.vector_size}")
        return self

    def _embed_tokens(self, tokens: List[str]) -> np.ndarray:
        """Embed a list of tokens into a sentence vector."""
        vectors = []
        for token in tokens:
            if token in self.model.wv:
                vectors.append(self.model.wv[token])

        if not vectors:
            return np.zeros(self.vector_size if self.pooling != "concat" else self.vector_size * 2)

        vstack = np.vstack(vectors)
        if self.pooling == "mean":
            return vstack.mean(axis=0)
        elif self.pooling == "max":
            return vstack.max(axis=0)
        elif self.pooling == "concat":
            return np.concatenate([vstack.mean(axis=0), vstack.max(axis=0)])
        return vstack.mean(axis=0)

    def transform(self, token_lists: List[List[str]]) -> np.ndarray:
        """Transform tokenized texts to embedding matrix."""
        if not self.is_fitted:
            raise RuntimeError("Call fit() before transform()")
        return np.vstack([self._embed_tokens(tokens) for tokens in token_lists])

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.model.save(path)

    def load(self, path: str) -> "Word2VecExtractor":
        from gensim.models import Word2Vec
        self.model = Word2Vec.load(path)
        self.is_fitted = True
        return self


# ──────────────────────────────────────────────────────────────
# Transformer Embedding Extractor (BERT / RoBERTa / DeBERTa)
# ──────────────────────────────────────────────────────────────

class TransformerEmbeddingExtractor:
    """
    Extract [CLS] token embeddings from pretrained transformer models.

    Why transformers:
    - BERT: Strong bidirectional contextual understanding
    - RoBERTa: Improved BERT training, better downstream performance
    - DeBERTa: Disentangled attention, SOTA on many NLU benchmarks

    Computational requirements:
    - BERT-base: ~110M params, ~3GB GPU RAM
    - RoBERTa-base: ~125M params, ~3.5GB GPU RAM
    - DeBERTa-v3-base: ~184M params, ~5GB GPU RAM
    Expected F1 improvement over TF-IDF: +8–15%
    """

    def __init__(
        self,
        model_name: str = "roberta-base",
        max_length: int = 512,
        batch_size: int = 16,
        device: Optional[str] = None,
        pooling: str = "cls",  # cls | mean | max
    ):
        self.model_name = model_name
        self.max_length = max_length
        self.batch_size = batch_size
        self.pooling = pooling
        self.tokenizer = None
        self.model = None
        self._device = device

    def _get_device(self):
        import torch
        if self._device:
            return torch.device(self._device)
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    def load_model(self) -> "TransformerEmbeddingExtractor":
        """Load tokenizer and model from HuggingFace Hub."""
        from transformers import AutoTokenizer, AutoModel
        logger.info(f"Loading transformer model: {self.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name)
        self.device = self._get_device()
        self.model.to(self.device)
        self.model.eval()
        logger.info(f"Model loaded on {self.device}")
        return self

    def _pool(self, outputs, attention_mask) -> np.ndarray:
        import torch
        last_hidden = outputs.last_hidden_state  # (B, L, H)
        if self.pooling == "cls":
            return last_hidden[:, 0, :].cpu().numpy()
        elif self.pooling == "mean":
            mask = attention_mask.unsqueeze(-1).float()
            summed = (last_hidden * mask).sum(dim=1)
            counts = mask.sum(dim=1).clamp(min=1e-9)
            return (summed / counts).cpu().numpy()
        elif self.pooling == "max":
            mask = attention_mask.unsqueeze(-1).float()
            last_hidden[mask == 0] = -1e9
            return last_hidden.max(dim=1).values.cpu().numpy()
        return last_hidden[:, 0, :].cpu().numpy()

    def extract(self, texts: List[str]) -> np.ndarray:
        """Extract embeddings for a list of texts, returns (N, H) array."""
        import torch
        if self.model is None:
            self.load_model()

        all_embeddings = []
        chunks = [texts[i : i + self.batch_size] for i in range(0, len(texts), self.batch_size)]

        with torch.no_grad():
            for chunk in chunks:
                encoded = self.tokenizer(
                    chunk,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                )
                encoded = {k: v.to(self.device) for k, v in encoded.items()}
                outputs = self.model(**encoded)
                embeddings = self._pool(outputs, encoded["attention_mask"])
                all_embeddings.append(embeddings)

        return np.vstack(all_embeddings)

    @property
    def embedding_dim(self) -> int:
        """Return the embedding dimension of this model."""
        dims = {
            "bert-base-uncased": 768,
            "roberta-base": 768,
            "microsoft/deberta-v3-base": 768,
            "bert-large-uncased": 1024,
        }
        return dims.get(self.model_name, 768)
