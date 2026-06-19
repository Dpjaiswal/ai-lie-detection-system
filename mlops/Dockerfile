# ─────────────────────────────────────────────────────────────
# Dockerfile — AI-Powered Lie Detection System (FastAPI Backend)
# Multi-stage build for minimal production image
# ─────────────────────────────────────────────────────────────

# ── Stage 1: Builder ─────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    libsndfile1-dev \
    libffi-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (Docker layer cache optimization)
COPY requirements.txt .

# Install Python dependencies into a prefix directory
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir --prefix=/install \
        fastapi \
        uvicorn[standard] \
        pydantic \
        python-multipart \
        httpx \
        websockets \
        aiofiles \
        python-dotenv \
        pyyaml \
        loguru \
        numpy \
        scikit-learn \
        xgboost \
        librosa \
        soundfile \
        scipy \
        joblib \
        rich


# ── Stage 2: Production Image ─────────────────────────────────
FROM python:3.11-slim AS production

WORKDIR /app

# Runtime OS dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    libgomp1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy application code
COPY src/ ./src/
COPY api/ ./api/
COPY configs/ ./configs/
COPY .env.example .env

# Create runtime directories
RUN mkdir -p \
    models/nlp \
    models/audio \
    models/fusion \
    data/audio/uploads \
    logs \
    reports

# Non-root user for security
RUN useradd --create-home --shell /bin/bash liedetect && \
    chown -R liedetect:liedetect /app
USER liedetect

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# Environment variables
ENV APP_ENV=production \
    DEBUG=false \
    API_HOST=0.0.0.0 \
    API_PORT=8000 \
    LOG_LEVEL=INFO \
    LOG_FORMAT=json \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Start server
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", \
     "--workers", "2", "--log-level", "info", "--access-log"]
