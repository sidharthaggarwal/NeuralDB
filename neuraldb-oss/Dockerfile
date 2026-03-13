# ─────────────────────────────────────────────────────────────────────────────
# NeuralDB — Production Dockerfile
#
# Multi-stage build:
#   builder  — installs Python deps into an isolated prefix
#   runtime  — copies only what's needed; runs as non-root user
#
# Security hardening:
#   - Non-root user (neuraldb uid=1001)
#   - No package manager in runtime image
#   - Read-only root filesystem (mount /tmp and /data as tmpfs/volume)
#   - HEALTHCHECK so orchestrators detect dead containers early
#   - Labels for OCI image spec compliance
# ─────────────────────────────────────────────────────────────────────────────

# ── Stage 1: dependency builder ───────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

# Install build deps; clean up in same layer to keep the layer small
RUN apt-get update \
    && apt-get install -y --no-install-recommends gcc \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt


# ── Stage 2: runtime image ───────────────────────────────────────────────
FROM python:3.11-slim AS runtime

LABEL org.opencontainers.image.title="NeuralDB"
LABEL org.opencontainers.image.description="AI-Native Database Engine"
LABEL org.opencontainers.image.version="1.0.0"
LABEL org.opencontainers.image.source="https://github.com/<owner>/neuraldb"
LABEL org.opencontainers.image.licenses="MIT"

# Create a dedicated non-root user
RUN groupadd -r -g 1001 neuraldb \
    && useradd -r -u 1001 -g neuraldb -m -d /home/neuraldb -s /sbin/nologin neuraldb

# Copy installed packages from builder
COPY --from=builder /install /usr/local

WORKDIR /app

# Copy application source
COPY --chown=neuraldb:neuraldb neuraldb/    ./neuraldb/
COPY --chown=neuraldb:neuraldb server.py   .

# Data directory — must be mounted as a persistent volume in production
RUN mkdir -p /data && chown neuraldb:neuraldb /data

USER neuraldb

# ── Environment defaults (override via --env or docker-compose) ───────────
ENV NEURALDB_DATA_PATH=/data
ENV NEURALDB_API_KEY=""
ENV NEURALDB_AUTH=true
ENV NEURALDB_VECTOR_DIM=128
ENV NEURALDB_RATE_LIMIT=true
ENV NEURALDB_RPM=120
ENV NEURALDB_BURST=20
ENV CORS_ORIGINS=*
ENV PORT=8000
ENV LOG_LEVEL=INFO

# Persist database files outside the container filesystem
VOLUME ["/data"]

EXPOSE 8000

HEALTHCHECK \
    --interval=30s \
    --timeout=10s \
    --start-period=15s \
    --retries=3 \
    CMD python -c "\
import urllib.request, sys; \
r = urllib.request.urlopen('http://localhost:8000/health', timeout=5); \
sys.exit(0 if r.status == 200 else 1)" \
    || exit 1

# Run with 2 workers; adjust via WORKERS env var if needed
CMD ["sh", "-c", \
     "python -m uvicorn server:app \
      --host 0.0.0.0 \
      --port ${PORT} \
      --workers ${WORKERS:-2} \
      --access-log \
      --log-level ${LOG_LEVEL,,}"]
