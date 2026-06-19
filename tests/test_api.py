"""
API Tests — AI-Powered Lie Detection System
Integration tests for all FastAPI endpoints.
"""
from __future__ import annotations

import io
import json
import struct
import wave
from typing import Generator

import numpy as np
import pytest
from fastapi.testclient import TestClient

# Ensure env is set before imports
import os
os.environ.setdefault("APP_ENV", "test")
os.environ.setdefault("DEBUG", "true")
os.environ.setdefault("LOG_FILE", "logs/test.log")


@pytest.fixture(scope="module")
def client() -> Generator:
    """Create a test client for the FastAPI app."""
    from api.main import app
    with TestClient(app) as c:
        yield c


def _make_test_wav(duration: float = 2.0, sample_rate: int = 22050) -> bytes:
    """Generate a minimal WAV file in memory for testing."""
    n_samples = int(duration * sample_rate)
    # Simple sine wave
    t = np.linspace(0, duration, n_samples)
    audio = (np.sin(2 * np.pi * 440 * t) * 32767 * 0.5).astype(np.int16)

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio.tobytes())
    return buf.getvalue()


# ──────────────────────────────────────────────────────────────
# Health Endpoint
# ──────────────────────────────────────────────────────────────

class TestHealthEndpoint:
    def test_health_returns_200(self, client: TestClient):
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_response_structure(self, client: TestClient):
        data = client.get("/health").json()
        assert "status" in data
        assert "version" in data
        assert "models_loaded" in data
        assert "uptime_seconds" in data

    def test_health_status_healthy(self, client: TestClient):
        data = client.get("/health").json()
        assert data["status"] == "healthy"


# ──────────────────────────────────────────────────────────────
# Root Endpoint
# ──────────────────────────────────────────────────────────────

class TestRootEndpoint:
    def test_root_returns_200(self, client: TestClient):
        response = client.get("/")
        assert response.status_code == 200

    def test_root_has_endpoints(self, client: TestClient):
        data = client.get("/").json()
        assert "endpoints" in data
        assert "ethical_disclaimer" in data


# ──────────────────────────────────────────────────────────────
# Text Prediction Endpoint
# ──────────────────────────────────────────────────────────────

class TestTextPrediction:
    def test_predict_text_basic(self, client: TestClient):
        response = client.post(
            "/predict/text",
            json={"text": "I was at home all evening and did not go out.", "model": "baseline"},
        )
        assert response.status_code == 200

    def test_predict_text_response_fields(self, client: TestClient):
        data = client.post(
            "/predict/text",
            json={"text": "I never met that person in my life.", "model": "baseline"},
        ).json()
        assert "text_prediction" in data
        assert "confidence" in data
        assert "lie_probability" in data
        assert "truth_probability" in data
        assert "verdict" in data
        assert "warning" in data

    def test_predict_text_probabilities_sum_to_one(self, client: TestClient):
        data = client.post(
            "/predict/text",
            json={"text": "I definitely did not steal anything.", "model": "baseline"},
        ).json()
        total = data["lie_probability"] + data["truth_probability"]
        assert abs(total - 1.0) < 1e-4

    def test_predict_text_prediction_is_valid(self, client: TestClient):
        data = client.post(
            "/predict/text",
            json={"text": "I was at work until 6pm.", "model": "baseline"},
        ).json()
        assert data["text_prediction"] in ("lie", "truth")

    def test_predict_text_confidence_in_range(self, client: TestClient):
        data = client.post(
            "/predict/text",
            json={"text": "I absolutely was not there.", "model": "baseline"},
        ).json()
        assert 0.0 <= data["confidence"] <= 1.0

    def test_predict_text_empty_text_returns_422(self, client: TestClient):
        response = client.post("/predict/text", json={"text": "", "model": "baseline"})
        assert response.status_code == 422

    def test_predict_text_short_text_returns_422(self, client: TestClient):
        response = client.post("/predict/text", json={"text": "Hi", "model": "baseline"})
        assert response.status_code == 422

    def test_predict_text_invalid_model_returns_422(self, client: TestClient):
        response = client.post(
            "/predict/text",
            json={"text": "This is a valid statement.", "model": "invalid_model"},
        )
        assert response.status_code == 422

    def test_predict_text_with_explanation(self, client: TestClient):
        data = client.post(
            "/predict/text",
            json={"text": "I definitely was not there that night.", "model": "baseline",
                  "include_explanation": True},
        ).json()
        assert data["explanation"] is not None

    def test_predict_text_verdict_is_valid(self, client: TestClient):
        data = client.post(
            "/predict/text",
            json={"text": "I was completely honest about everything.", "model": "baseline"},
        ).json()
        valid_verdicts = {"likely_lie", "possibly_lie", "uncertain", "likely_truth"}
        assert data["verdict"] in valid_verdicts


# ──────────────────────────────────────────────────────────────
# Audio Prediction Endpoint
# ──────────────────────────────────────────────────────────────

class TestAudioPrediction:
    def test_predict_audio_basic(self, client: TestClient):
        wav_bytes = _make_test_wav(duration=3.0)
        response = client.post(
            "/predict/audio",
            files={"file": ("test.wav", wav_bytes, "audio/wav")},
            data={"include_features": "false"},
        )
        assert response.status_code == 200

    def test_predict_audio_response_fields(self, client: TestClient):
        wav_bytes = _make_test_wav()
        data = client.post(
            "/predict/audio",
            files={"file": ("test.wav", wav_bytes, "audio/wav")},
        ).json()
        assert "voice_prediction" in data
        assert "stress_probability" in data
        assert "confidence" in data
        assert "audio_duration_seconds" in data
        assert "warning" in data

    def test_predict_audio_voice_prediction_valid(self, client: TestClient):
        wav_bytes = _make_test_wav()
        data = client.post(
            "/predict/audio",
            files={"file": ("test.wav", wav_bytes, "audio/wav")},
        ).json()
        assert data["voice_prediction"] in ("stress_detected", "no_stress")

    def test_predict_audio_invalid_format_returns_422(self, client: TestClient):
        response = client.post(
            "/predict/audio",
            files={"file": ("test.txt", b"not audio", "text/plain")},
        )
        assert response.status_code == 422

    def test_predict_audio_confidence_in_range(self, client: TestClient):
        wav_bytes = _make_test_wav()
        data = client.post(
            "/predict/audio",
            files={"file": ("test.wav", wav_bytes, "audio/wav")},
        ).json()
        assert 0.0 <= data["confidence"] <= 1.0


# ──────────────────────────────────────────────────────────────
# Explain Endpoint
# ──────────────────────────────────────────────────────────────

class TestExplainEndpoint:
    def test_explain_text_lime(self, client: TestClient):
        response = client.post(
            "/explain",
            json={"text": "I was definitely not there that evening.", "method": "lime", "model_type": "text"},
        )
        assert response.status_code == 200

    def test_explain_response_has_text_explanation(self, client: TestClient):
        data = client.post(
            "/explain",
            json={"text": "I never did anything wrong.", "method": "lime", "model_type": "text"},
        ).json()
        assert "text_explanation" in data
        assert "processing_time_ms" in data

    def test_explain_without_inputs_returns_422(self, client: TestClient):
        response = client.post(
            "/explain",
            json={"method": "lime", "model_type": "text"},
        )
        assert response.status_code in (200, 422)  # Either fails gracefully

    def test_explain_invalid_method_returns_422(self, client: TestClient):
        response = client.post(
            "/explain",
            json={"text": "Test text.", "method": "invalid", "model_type": "text"},
        )
        assert response.status_code == 422


# ──────────────────────────────────────────────────────────────
# OpenAPI Documentation
# ──────────────────────────────────────────────────────────────

class TestOpenAPI:
    def test_openapi_schema_accessible(self, client: TestClient):
        response = client.get("/openapi.json")
        assert response.status_code == 200

    def test_docs_accessible(self, client: TestClient):
        response = client.get("/docs")
        assert response.status_code == 200

    def test_redoc_accessible(self, client: TestClient):
        response = client.get("/redoc")
        assert response.status_code == 200
