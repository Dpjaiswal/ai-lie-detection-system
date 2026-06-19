"""
Configuration Manager — AI-Powered Lie Detection System
Centralized config loading from YAML files and environment variables.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent.parent
CONFIGS_DIR = BASE_DIR / "configs"


def load_yaml(path: Path) -> Dict[str, Any]:
    """Load a YAML configuration file."""
    with open(path, "r") as f:
        return yaml.safe_load(f)


class Config:
    """Singleton configuration container."""

    _instance: Optional["Config"] = None
    _loaded: bool = False

    def __new__(cls) -> "Config":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        if not self._loaded:
            self._model = load_yaml(CONFIGS_DIR / "model_config.yaml")
            self._training = load_yaml(CONFIGS_DIR / "training_config.yaml")
            self._api = load_yaml(CONFIGS_DIR / "api_config.yaml")
            Config._loaded = True

    # ── Environment helpers ────────────────────────────────────
    @property
    def app_name(self) -> str:
        return os.getenv("APP_NAME", "AI Lie Detection System")

    @property
    def app_env(self) -> str:
        return os.getenv("APP_ENV", "development")

    @property
    def debug(self) -> bool:
        return os.getenv("DEBUG", "false").lower() == "true"

    @property
    def secret_key(self) -> str:
        return os.getenv("SECRET_KEY", "change-me-in-production")

    # ── Model paths ───────────────────────────────────────────
    @property
    def nlp_model_path(self) -> Path:
        return BASE_DIR / os.getenv("NLP_MODEL_PATH", "models/nlp/best_model")

    @property
    def audio_model_path(self) -> Path:
        return BASE_DIR / os.getenv("AUDIO_MODEL_PATH", "models/audio/best_model")

    @property
    def fusion_model_path(self) -> Path:
        return BASE_DIR / os.getenv("FUSION_MODEL_PATH", "models/fusion/best_model")

    # ── HuggingFace ───────────────────────────────────────────
    @property
    def hf_model_name(self) -> str:
        return os.getenv("HF_MODEL_NAME", "roberta-base")

    # ── API settings ──────────────────────────────────────────
    @property
    def api_host(self) -> str:
        return os.getenv("API_HOST", "0.0.0.0")

    @property
    def api_port(self) -> int:
        return int(os.getenv("API_PORT", "8000"))

    # ── Audio settings ────────────────────────────────────────
    @property
    def sample_rate(self) -> int:
        return int(os.getenv("AUDIO_SAMPLE_RATE", "22050"))

    @property
    def audio_upload_dir(self) -> Path:
        return BASE_DIR / os.getenv("AUDIO_UPLOAD_DIR", "data/audio/uploads")

    # ── Passthrough to YAML sections ──────────────────────────
    @property
    def model(self) -> Dict[str, Any]:
        return self._model

    @property
    def training(self) -> Dict[str, Any]:
        return self._training

    @property
    def api(self) -> Dict[str, Any]:
        return self._api

    # ── Convenience accessors ─────────────────────────────────
    def get_nlp_config(self, model_type: str = "transformer") -> Dict[str, Any]:
        return self._model["nlp"].get(model_type, {})

    def get_audio_config(self, model_type: str = "deep_learning") -> Dict[str, Any]:
        return self._model["audio"].get(model_type, {})

    def get_fusion_config(self, method: str = "hybrid_fusion") -> Dict[str, Any]:
        return self._model["fusion"].get(method, {})

    def get_feature_config(self) -> Dict[str, Any]:
        return self._model["audio"]["feature_extraction"]

    def __repr__(self) -> str:
        return f"<Config env={self.app_env} debug={self.debug}>"


# Global config singleton
config = Config()
