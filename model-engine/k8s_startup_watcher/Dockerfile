# K8s Startup Watcher - Generic pod startup metrics collection
#
# Watches K8s pod events and emits startup metrics via OTLP.
# Uses deterministic trace IDs for correlation with in-container metrics.
#
# Build:
#   docker build -f k8s_startup_watcher/Dockerfile -t k8s-startup-watcher .
#
# Multi-stage: -dev builder → distroless runtime (0 OS CVEs)
ARG ECR_REGISTRY=022465994601.dkr.ecr.us-west-2.amazonaws.com

# --- Builder stage ---
FROM ${ECR_REGISTRY}/golden/chainguard/python:3.11-dev AS builder
USER root
ENTRYPOINT []
WORKDIR /app

RUN pip install --no-cache-dir \
    kubernetes~=29.0.0 \
    opentelemetry-api~=1.27.0 \
    opentelemetry-sdk~=1.27.0 \
    opentelemetry-exporter-otlp~=1.27.0

# --- Runtime stage (distroless) ---
FROM ${ECR_REGISTRY}/golden/chainguard/python:3.11

WORKDIR /app

# Copy installed site-packages from builder (Chainguard uses /usr/lib/pythonX.Y)
COPY --from=builder /usr/lib/python3.11/site-packages /usr/lib/python3.11/site-packages

# Copy shared startup_tracing library (for deterministic trace correlation)
COPY model_engine_server/common/startup_tracing /app/startup_tracing

# Copy watcher modules
COPY k8s_startup_watcher/config.py /app/config.py
COPY k8s_startup_watcher/watcher.py /app/watcher.py

CMD ["python", "/app/watcher.py"]
