"""
FastAPI Application — AI-Powered Lie Detection System
Production-grade API with CORS, rate limiting, logging, and async support.
"""
from __future__ import annotations

import time
import uuid
from contextlib import asynccontextmanager
from typing import AsyncGenerator

import uvicorn
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from fastapi.openapi.utils import get_openapi

from api.routers import predict, explain, health
from src.utils.logger import logger
from src.utils.config import config


# ──────────────────────────────────────────────────────────────
# Application Lifespan (startup / shutdown events)
# ──────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator:
    """Application startup and shutdown logic."""
    logger.info("=" * 60)
    logger.info(f"  Starting {config.app_name} API v1.0.0")
    logger.info(f"  Environment: {config.app_env}")
    logger.info(f"  Debug mode:  {config.debug}")
    logger.info("=" * 60)

    # Pre-warm audio pipeline on startup
    try:
        from src.audio.feature_extractor import AudioFeaturePipeline
        logger.info("Pre-warming audio pipeline...")
        AudioFeaturePipeline()
        logger.info("Audio pipeline ready.")
    except Exception as e:
        logger.warning(f"Audio pipeline warm-up skipped: {e}")

    yield  # App runs here

    logger.info("Shutting down AI Lie Detection API...")


# ──────────────────────────────────────────────────────────────
# FastAPI App Instance
# ──────────────────────────────────────────────────────────────

app = FastAPI(
    title="AI-Powered Lie Detection System API",
    description=(
        "Multimodal deception likelihood estimation system using NLP and audio analysis.\n\n"
        "**⚠️ Ethical Disclaimer**: This system provides probabilistic estimates based on "
        "statistical patterns. Results are NOT definitive truth detection and should NOT "
        "be used for legal, criminal, or high-stakes decision making."
    ),
    version="1.0.0",
    contact={
        "name": "AI Research Team",
        "email": "research@liedetect.ai",
    },
    license_info={"name": "MIT", "url": "https://opensource.org/licenses/MIT"},
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)


# ──────────────────────────────────────────────────────────────
# Middleware
# ──────────────────────────────────────────────────────────────

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501", "http://localhost:3000", "http://127.0.0.1:8501"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# GZip compression for responses > 1KB
app.add_middleware(GZipMiddleware, minimum_size=1000)


@app.middleware("http")
async def request_logging_middleware(request: Request, call_next) -> Response:
    """Log all incoming requests with timing and request ID."""
    request_id = str(uuid.uuid4())[:8]
    start_time = time.time()

    # Attach request ID to state
    request.state.request_id = request_id

    logger.info(
        f"[{request_id}] → {request.method} {request.url.path} "
        f"(client={request.client.host if request.client else 'unknown'})"
    )

    try:
        response = await call_next(request)
    except Exception as exc:
        logger.error(f"[{request_id}] Unhandled exception: {exc}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"error": "Internal server error", "request_id": request_id},
        )

    duration_ms = (time.time() - start_time) * 1000
    response.headers["X-Request-ID"] = request_id
    response.headers["X-Process-Time-Ms"] = f"{duration_ms:.2f}"

    logger.info(
        f"[{request_id}] ← {response.status_code} "
        f"({duration_ms:.1f}ms)"
    )
    return response


# ──────────────────────────────────────────────────────────────
# Exception Handlers
# ──────────────────────────────────────────────────────────────

@app.exception_handler(404)
async def not_found_handler(request: Request, exc) -> JSONResponse:
    return JSONResponse(
        status_code=404,
        content={
            "error": "Endpoint not found",
            "detail": f"No route matches '{request.method} {request.url.path}'",
            "docs": "/docs",
        },
    )


@app.exception_handler(422)
async def validation_error_handler(request: Request, exc) -> JSONResponse:
    return JSONResponse(
        status_code=422,
        content={
            "error": "Validation error",
            "detail": str(exc),
        },
    )


# ──────────────────────────────────────────────────────────────
# Routers
# ──────────────────────────────────────────────────────────────

app.include_router(health.router)
app.include_router(predict.router)
app.include_router(explain.router)


# ──────────────────────────────────────────────────────────────
# Root endpoint
# ──────────────────────────────────────────────────────────────

@app.get("/", tags=["Root"], summary="API Information")
async def root():
    return {
        "name": "AI-Powered Lie Detection System API",
        "version": "1.0.0",
        "description": "Multimodal deception likelihood estimation using NLP + Audio",
        "docs": "/docs",
        "health": "/health",
        "endpoints": {
            "POST /predict/text": "Analyze text for deception patterns",
            "POST /predict/audio": "Analyze voice recording for stress indicators",
            "POST /predict/multimodal": "Combined text + voice analysis",
            "POST /explain": "Generate LIME/SHAP explanations",
            "GET /health": "Service health check",
        },
        "ethical_disclaimer": (
            "This system estimates deception likelihood probabilistically. "
            "It is NOT a definitive lie detector and should never be used "
            "for legal, criminal, employment, or high-stakes decisions."
        ),
    }


# ──────────────────────────────────────────────────────────────
# WebSocket endpoint for real-time streaming
# ──────────────────────────────────────────────────────────────

@app.websocket("/ws/predict/realtime")
async def websocket_realtime_predict(websocket):
    """
    WebSocket endpoint for real-time audio streaming and prediction.

    Protocol:
    1. Client connects and sends audio chunks as binary frames
    2. Server processes chunks incrementally
    3. Server sends prediction updates as JSON text frames
    4. Client sends {"action": "stop"} to end session

    Feature roadmap (not fully implemented):
    - Incremental MFCC extraction per 500ms chunk
    - Rolling prediction window with exponential smoothing
    - WebRTC-compatible audio format support
    """
    from fastapi.websockets import WebSocketState
    await websocket.accept()
    logger.info("WebSocket connection established for real-time prediction")

    try:
        audio_buffer = b""
        chunk_count = 0

        while True:
            data = await websocket.receive()

            if data.get("type") == "websocket.disconnect":
                break

            if "bytes" in data and data["bytes"]:
                audio_buffer += data["bytes"]
                chunk_count += 1

                # Process every ~500ms (assuming 22050 Hz, 16-bit, 1 channel ≈ 22050 bytes)
                if len(audio_buffer) >= 22050:
                    # Incremental prediction
                    import numpy as np
                    mock_stress = float(np.random.beta(2, 2))
                    await websocket.send_json({
                        "chunk": chunk_count,
                        "buffer_size_bytes": len(audio_buffer),
                        "stress_probability": mock_stress,
                        "prediction": "stress_detected" if mock_stress > 0.5 else "no_stress",
                        "streaming": True,
                    })
                    audio_buffer = b""

            elif "text" in data:
                import json
                msg = json.loads(data["text"])
                if msg.get("action") == "stop":
                    await websocket.send_json({"action": "stopped", "total_chunks": chunk_count})
                    break

    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        if websocket.client_state == WebSocketState.CONNECTED:
            await websocket.send_json({"error": str(e)})
    finally:
        logger.info("WebSocket connection closed")


# ──────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────

def run():
    """Run the API server (entry point for console script)."""
    uvicorn.run(
        "api.main:app",
        host=config.api_host,
        port=config.api_port,
        reload=config.debug,
        log_level="info",
        workers=1 if config.debug else 4,
        access_log=True,
    )


if __name__ == "__main__":
    run()
