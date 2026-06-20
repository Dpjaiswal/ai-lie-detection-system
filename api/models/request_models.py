"""
FastAPI Request Schemas — AI-Powered Lie Detection System
"""
from __future__ import annotations

from typing import Dict, Literal, Optional
from pydantic import BaseModel, Field


class TextPredictRequest(BaseModel):
    """Schema for text-only lie prediction requests."""
    text: str = Field(
        ...,
        min_length=5,
        description="The statement text to analyze. Must be at least 5 characters.",
        json_schema_extra={"example": "I was at home the entire evening and never left."}
    )
    model: Literal["roberta", "bert", "deberta", "baseline"] = Field(
        default="roberta",
        description="Linguistic model to use for analysis.",
        json_schema_extra={"example": "roberta"}
    )
    include_explanation: bool = Field(
        default=False,
        description="Whether to return LIME/SHAP feature importances."
    )


class MultiModalPredictRequest(BaseModel):
    """Schema for multimodal lie prediction requests (JSON body version if used)."""
    text: str = Field(
        ...,
        min_length=5,
        description="The spoken statement text."
    )
    fusion_method: Literal["hybrid", "late", "early"] = Field(
        default="hybrid",
        description="Method to combine text and audio signals."
    )
    include_explanation: bool = Field(
        default=True,
        description="Whether to generate word-level and audio feature importances."
    )


class ExplainRequest(BaseModel):
    """Schema for explanation-only requests."""
    method: Literal["lime", "shap", "both"] = Field(
        default="lime",
        description="Explainability method to use."
    )
    model_type: Literal["text", "audio", "fusion"] = Field(
        default="text",
        description="Modality to explain."
    )
    text: Optional[str] = Field(
        default=None,
        description="Input text statement. Required if model_type is 'text' or 'fusion'."
    )
    audio_features: Optional[Dict[str, float]] = Field(
        default=None,
        description="Pre-extracted audio feature dictionary. Required if model_type is 'audio' or 'fusion'."
    )
