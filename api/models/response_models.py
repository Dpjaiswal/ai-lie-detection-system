"""
FastAPI Response Schemas — AI-Powered Lie Detection System
"""
from __future__ import annotations

from typing import Dict, List, Optional
from pydantic import BaseModel, Field


class WordImportance(BaseModel):
    """LIME word-level contribution score."""
    word: str = Field(..., description="The word analyzed.")
    weight: float = Field(..., description="Importance weight score.")
    direction: str = Field(..., description="Direction of contribution: 'lie' or 'truth'.")


class AudioFeatureImportance(BaseModel):
    """SHAP audio feature-level contribution score."""
    feature: str = Field(..., description="Vocal feature name.")
    shap_value: float = Field(..., description="SHAP value indicating contribution magnitude.")
    direction: str = Field(..., description="Direction of contribution: 'stress_detected' or 'calm'.")


class ExplanationResponse(BaseModel):
    """Unified wrapper for LIME/SHAP explainability outputs."""
    method_used: str = Field(..., description="Explanation method name (e.g., lime, shap).")
    important_words: Optional[List[WordImportance]] = Field(
        default=None,
        description="LIME word-level breakdown for text."
    )
    audio_features: Optional[List[AudioFeatureImportance]] = Field(
        default=None,
        description="SHAP feature breakdown for audio."
    )


class TextPredictResponse(BaseModel):
    """Response schema for text lie prediction."""
    text_prediction: str = Field(..., description="Linguistic verdict: 'lie' or 'truth'.")
    confidence: float = Field(..., description="Model confidence score (0.0 to 1.0).")
    lie_probability: float = Field(..., description="Deception probability score.")
    truth_probability: float = Field(..., description="Truthfulness probability score.")
    model_used: str = Field(..., description="NLP model used for inference.")
    verdict: str = Field(..., description="Friendly classification (likely_lie, possibly_lie, uncertain, likely_truth).")
    explanation: Optional[ExplanationResponse] = Field(
        default=None,
        description="On-demand explainability breakdown."
    )
    processing_time_ms: float = Field(..., description="Server processing duration.")
    warning: str = Field(
        default="Ethical Notice: This system provides probabilistic estimates based on linguistic patterns. Results are not definitive.",
        description="General warning and ethical disclaimer."
    )


class AudioPredictResponse(BaseModel):
    """Response schema for voice lie prediction."""
    voice_prediction: str = Field(..., description="Voice stress verdict: 'stress_detected' or 'no_stress'.")
    stress_probability: float = Field(..., description="Voice stress probability score.")
    confidence: float = Field(..., description="Model confidence score.")
    audio_features: Optional[Dict[str, float]] = Field(
        default=None,
        description="Extracted acoustic features."
    )
    model_used: str = Field(..., description="Acoustic classifier model used.")
    audio_duration_seconds: float = Field(..., description="Duration of processed audio in seconds.")
    explanation: Optional[ExplanationResponse] = Field(
        default=None,
        description="On-demand explainability breakdown."
    )
    processing_time_ms: float = Field(..., description="Server processing duration.")
    warning: str = Field(
        default="Ethical Notice: Vocal stress does not equate to lying. Results are probabilistic.",
        description="General warning and ethical disclaimer."
    )


class MultiModalPredictResponse(BaseModel):
    """Response schema for combined multimodal lie prediction."""
    text_prediction: str = Field(..., description="Linguistic verdict: 'lie' or 'truth'.")
    voice_prediction: str = Field(..., description="Voice stress verdict: 'stress_detected' or 'no_stress'.")
    final_prediction: str = Field(..., description="Combined decision label.")
    confidence: float = Field(..., description="Fusion model confidence score.")
    lie_probability: float = Field(..., description="Fused deception probability score.")
    text_lie_probability: float = Field(..., description="Linguistic-only lie probability.")
    audio_stress_probability: float = Field(..., description="Acoustic-only stress probability.")
    fusion_method: str = Field(..., description="Fusion algorithm used.")
    explanation: Optional[ExplanationResponse] = Field(
        default=None,
        description="Combined explainability breakdown."
    )
    processing_time_ms: float = Field(..., description="Server processing duration.")
    model_versions: Dict[str, str] = Field(..., description="Versions of NLP, Audio, and Fusion models used.")
    warning: str = Field(
        default="Ethical Notice: Fused multimodal estimates are statistical. Results are not definitive.",
        description="General warning and ethical disclaimer."
    )


class ExplainResponse(BaseModel):
    """Response schema for model explanation queries."""
    text_explanation: Optional[ExplanationResponse] = Field(
        default=None,
        description="Word-level text explanations."
    )
    audio_explanation: Optional[ExplanationResponse] = Field(
        default=None,
        description="Feature-level audio explanations."
    )
    methods_used: List[str] = Field(..., description="Explainability methods executed.")
    processing_time_ms: float = Field(..., description="Server processing duration.")


class HealthResponse(BaseModel):
    """Response schema for system health check."""
    status: str = Field(..., description="Health status (e.g., 'healthy').")
    version: str = Field(..., description="Backend system version.")
    models_loaded: Dict[str, bool] = Field(..., description="Loading status of various backend models.")
    uptime_seconds: float = Field(..., description="System uptime duration.")
    environment: str = Field(..., description="System execution environment (development, test, production).")
