"""
Health Router — AI-Powered Lie Detection System
GET /health endpoint for service monitoring.
"""
from __future__ import annotations

import time
from typing import Dict

from fastapi import APIRouter
from api.models.response_models import HealthResponse
from src.utils.config import config

router = APIRouter(tags=["Health"])
_start_time = time.time()


@router.get("/health", response_model=HealthResponse, summary="Health check")
async def health_check() -> HealthResponse:
    """
    Returns service health status, loaded models, and uptime.
    Use this endpoint for liveness and readiness probes in Kubernetes.
    """
    models_loaded: Dict[str, bool] = {
        "nlp_model": config.nlp_model_path.exists(),
        "audio_model": config.audio_model_path.exists(),
        "fusion_model": config.fusion_model_path.exists(),
    }

    return HealthResponse(
        status="healthy",
        version="1.0.0",
        models_loaded=models_loaded,
        uptime_seconds=round(time.time() - _start_time, 2),
        environment=config.app_env,
    )
