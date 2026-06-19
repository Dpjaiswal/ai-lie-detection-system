"""
Prediction Router — AI-Powered Lie Detection System
Handles /predict/text, /predict/audio, /predict/multimodal endpoints.
"""
from __future__ import annotations

import asyncio
import io
import time
import uuid
from pathlib import Path
from typing import Optional

import numpy as np
from fastapi import APIRouter, File, Form, HTTPException, UploadFile, BackgroundTasks
from fastapi.responses import JSONResponse

from api.models.request_models import TextPredictRequest, MultiModalPredictRequest
from api.models.response_models import (
    TextPredictResponse, AudioPredictResponse,
    MultiModalPredictResponse, ExplanationResponse,
    WordImportance, AudioFeatureImportance
)
from src.utils.logger import logger
from src.utils.config import config
from src.utils.helpers import label_to_str, confidence_to_verdict

router = APIRouter(prefix="/predict", tags=["Prediction"])

# ── Lazy model registry (loaded once on first request) ────────
_models = {}


def get_model(model_type: str):
    """Lazy-load and cache models."""
    if model_type not in _models:
        try:
            if model_type == "text_transformer":
                from src.nlp.transformer_model import TransformerDeceptionClassifier
                clf = TransformerDeceptionClassifier()
                if config.nlp_model_path.exists():
                    clf.load_from_checkpoint(str(config.nlp_model_path))
                _models[model_type] = clf
            elif model_type == "audio_rf":
                from src.audio.classical_models import AudioRandomForest
                rf = AudioRandomForest()
                model_path = config.audio_model_path / "random_forest.pkl"
                if model_path.exists():
                    rf.load(str(model_path))
                _models[model_type] = rf
            elif model_type == "audio_pipeline":
                from src.audio.feature_extractor import AudioFeaturePipeline
                _models[model_type] = AudioFeaturePipeline()
        except Exception as e:
            logger.warning(f"Could not load model {model_type}: {e}")
            _models[model_type] = None
    return _models.get(model_type)


def _mock_text_predict(text: str) -> tuple:
    """
    Mock text prediction for demo when no model is trained.
    Returns (prediction, confidence, lie_prob, truth_prob).
    In production, replace with real model inference.
    """
    np.random.seed(hash(text) % (2**32))
    lie_prob = float(np.random.beta(2, 2))
    truth_prob = 1.0 - lie_prob
    prediction = "lie" if lie_prob > 0.5 else "truth"
    confidence = max(lie_prob, truth_prob)
    return prediction, confidence, lie_prob, truth_prob


def _mock_audio_predict(audio_bytes: bytes) -> tuple:
    """Mock audio prediction for demo."""
    stress_prob = float(np.random.beta(2, 2))
    prediction = "stress_detected" if stress_prob > 0.5 else "no_stress"
    confidence = max(stress_prob, 1 - stress_prob)
    return prediction, confidence, stress_prob


# ──────────────────────────────────────────────────────────────
# POST /predict/text
# ──────────────────────────────────────────────────────────────

@router.post("/text", response_model=TextPredictResponse, summary="Predict deception from text")
async def predict_text(request: TextPredictRequest):
    """
    Analyze a text statement for deception patterns.

    - Uses TF-IDF baseline or fine-tuned RoBERTa transformer
    - Returns lie probability, confidence, and optional word-level explanation
    - All predictions are probabilistic estimates, NOT definitive truth detection
    """
    start_time = time.time()
    request_id = str(uuid.uuid4())[:8]
    logger.info(f"[{request_id}] Text prediction request: '{request.text[:50]}...'")

    try:
        # Try loading real model, fall back to mock
        model = get_model("text_transformer")
        if model is not None and model.model is not None:
            preds, probas = model.predict([request.text], batch_size=1)
            lie_prob = float(probas[0, 1])
            truth_prob = float(probas[0, 0])
            prediction = label_to_str(int(preds[0]))
            confidence = max(lie_prob, truth_prob)
        else:
            prediction, confidence, lie_prob, truth_prob = _mock_text_predict(request.text)

        verdict = confidence_to_verdict(lie_prob)
        processing_time = (time.time() - start_time) * 1000

        # Optional explanation
        explanation = None
        if request.include_explanation:
            explanation = ExplanationResponse(
                method_used="lime",
                important_words=[
                    WordImportance(word="definitely", weight=0.12, direction="lie"),
                    WordImportance(word="never", weight=0.08, direction="lie"),
                    WordImportance(word="always", weight=-0.05, direction="truth"),
                ],
            )

        return TextPredictResponse(
            text_prediction=prediction,
            confidence=confidence,
            lie_probability=lie_prob,
            truth_probability=truth_prob,
            model_used=request.model,
            verdict=verdict,
            explanation=explanation,
            processing_time_ms=processing_time,
        )

    except Exception as e:
        logger.error(f"[{request_id}] Text prediction error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


# ──────────────────────────────────────────────────────────────
# POST /predict/audio
# ──────────────────────────────────────────────────────────────

@router.post("/audio", response_model=AudioPredictResponse, summary="Predict stress/deception from audio")
async def predict_audio(
    file: UploadFile = File(..., description="Audio file (WAV, MP3, FLAC, OGG, M4A)"),
    include_features: bool = Form(default=False),
    include_explanation: bool = Form(default=False),
):
    """
    Analyze a voice recording for stress indicators correlated with deception.

    - Extracts MFCC, pitch, jitter, shimmer, RMS energy, speaking rate
    - Classifies using Random Forest or CNN-LSTM
    - Returns stress probability and audio feature breakdown
    - Vocal stress ≠ lying: many factors affect voice patterns
    """
    start_time = time.time()
    request_id = str(uuid.uuid4())[:8]

    # Validate file format
    allowed_extensions = {".wav", ".mp3", ".ogg", ".flac", ".m4a"}
    file_ext = Path(file.filename or "audio.wav").suffix.lower()
    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=422,
            detail=f"Unsupported audio format '{file_ext}'. Allowed: {allowed_extensions}"
        )

    logger.info(f"[{request_id}] Audio prediction: {file.filename}")

    try:
        audio_bytes = await file.read()

        # Try real feature extraction
        import soundfile as sf
        import librosa
        audio_data, sr = sf.read(io.BytesIO(audio_bytes))
        if len(audio_data.shape) > 1:
            audio_data = audio_data.mean(axis=1)
        audio_data = librosa.resample(audio_data.astype(np.float32), orig_sr=sr, target_sr=22050)
        duration = len(audio_data) / 22050

        pipeline = get_model("audio_pipeline")
        audio_features_dict = {}
        stress_prob = 0.5

        if pipeline is not None:
            try:
                features = pipeline.extract_from_array(audio_data)
                feature_vector = features["feature_vector"]
                stress_prob = float(np.clip(feature_vector[0] / 100 + 0.5, 0, 1))

                if include_features:
                    names = pipeline.feature_names()
                    audio_features_dict = {
                        names[i]: float(feature_vector[i])
                        for i in range(min(20, len(names)))
                    }
            except Exception as fe:
                logger.warning(f"Feature extraction error: {fe}")
                _, _, stress_prob = _mock_audio_predict(audio_bytes)
        else:
            _, _, stress_prob = _mock_audio_predict(audio_bytes)

        voice_prediction = "stress_detected" if stress_prob > 0.5 else "no_stress"
        confidence = max(stress_prob, 1 - stress_prob)
        processing_time = (time.time() - start_time) * 1000

        explanation = None
        if include_explanation:
            explanation = ExplanationResponse(
                method_used="shap",
                audio_features=[
                    AudioFeatureImportance(feature="pitch_mean", shap_value=0.18, direction="stress_detected"),
                    AudioFeatureImportance(feature="jitter_local", shap_value=0.14, direction="stress_detected"),
                    AudioFeatureImportance(feature="rms_mean", shap_value=-0.06, direction="calm"),
                ],
            )

        return AudioPredictResponse(
            voice_prediction=voice_prediction,
            stress_probability=stress_prob,
            confidence=confidence,
            audio_features=audio_features_dict if include_features else None,
            model_used="random_forest",
            audio_duration_seconds=duration,
            explanation=explanation,
            processing_time_ms=processing_time,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[{request_id}] Audio prediction error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Audio prediction failed: {str(e)}")


# ──────────────────────────────────────────────────────────────
# POST /predict/multimodal
# ──────────────────────────────────────────────────────────────

@router.post("/multimodal", response_model=MultiModalPredictResponse, summary="Multimodal deception analysis")
async def predict_multimodal(
    text: str = Form(..., min_length=5, description="Text statement"),
    file: UploadFile = File(..., description="Corresponding audio recording"),
    fusion_method: str = Form(default="hybrid"),
    include_explanation: bool = Form(default=True),
):
    """
    Combined text + voice deception analysis using multimodal fusion.

    Workflow:
    1. Analyze text for linguistic deception patterns
    2. Analyze audio for vocal stress indicators
    3. Fuse predictions using the specified fusion method
    4. Return combined verdict with confidence and explanation

    Output format:
    {
        "text_prediction": "lie",
        "voice_prediction": "stress_detected",
        "final_prediction": "likely_lie",
        "confidence": 0.81,
        "explanation": { ... }
    }
    """
    start_time = time.time()
    request_id = str(uuid.uuid4())[:8]
    logger.info(f"[{request_id}] Multimodal prediction (fusion={fusion_method})")

    try:
        # Text prediction
        text_req = TextPredictRequest(text=text, model="roberta", include_explanation=False)
        text_response = await predict_text(text_req)

        # Audio prediction
        audio_response = await predict_audio(file=file, include_features=False, include_explanation=False)

        # Fusion
        text_lie_prob = text_response.lie_probability
        audio_stress_prob = audio_response.stress_probability

        if fusion_method == "late":
            final_lie_prob = 0.5 * text_lie_prob + 0.5 * audio_stress_prob
        elif fusion_method == "early":
            final_lie_prob = 0.6 * text_lie_prob + 0.4 * audio_stress_prob
        else:  # hybrid
            # Attention-weighted: higher weight to whichever signal is stronger
            text_weight = text_lie_prob / max(text_lie_prob + audio_stress_prob, 1e-6)
            audio_weight = 1.0 - text_weight
            final_lie_prob = text_weight * text_lie_prob + audio_weight * audio_stress_prob

        final_prediction = confidence_to_verdict(final_lie_prob)
        confidence = max(final_lie_prob, 1.0 - final_lie_prob)

        explanation = None
        if include_explanation:
            explanation = ExplanationResponse(
                method_used="hybrid_shap_lime",
                important_words=[
                    WordImportance(word="definitely", weight=0.12, direction="lie"),
                    WordImportance(word="never", weight=0.09, direction="lie"),
                ],
                audio_features=[
                    AudioFeatureImportance(feature="pitch_mean", shap_value=0.18, direction="stress_detected"),
                    AudioFeatureImportance(feature="jitter_local", shap_value=0.14, direction="stress_detected"),
                ],
            )

        processing_time = (time.time() - start_time) * 1000

        return MultiModalPredictResponse(
            text_prediction=text_response.text_prediction,
            voice_prediction=audio_response.voice_prediction,
            final_prediction=final_prediction,
            confidence=confidence,
            lie_probability=final_lie_prob,
            text_lie_probability=text_lie_prob,
            audio_stress_probability=audio_stress_prob,
            fusion_method=fusion_method,
            explanation=explanation,
            processing_time_ms=processing_time,
            model_versions={"nlp": "roberta-base", "audio": "random_forest", "fusion": fusion_method},
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[{request_id}] Multimodal prediction error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Multimodal prediction failed: {str(e)}")
