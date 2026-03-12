# K8s Startup Watcher - Generic pod startup metrics collection
#
# Watches K8s pod events and emits startup metrics via OTLP.
# Uses deterministic trace IDs for correlation with in-container metrics.
#
# Multi-stage build: -dev builder (has pip) → distroless runtime (no shell/pkg mgr)
#
# Build:
#   docker build -f k8s_startup_watcher/Dockerfile -t k8s-startup-watcher .
#
ARG ECR_REGISTRY=022465994601.dkr.ecr.us-west-2.amazonaws.com

# --- Builder: install pip dependencies ---
FROM ${ECR_REGISTRY}/golden/chainguard/python:3.11-dev AS builder

USER root
ENTRYPOINT []
WORKDIR /app

RUN pip install --no-cache-dir \
    kubernetes~=29.0.0 \
    opentelemetry-api~=1.27.0 \
    opentelemetry-sdk~=1.27.0 \
    opentelemetry-exporter-otlp~=1.27.0

# --- Runtime: distroless (no shell, no pip, no build tools) ---
FROM ${ECR_REGISTRY}/golden/chainguard/python:3.11

# Distroless image defaults: ENTRYPOINT ["/usr/bin/python"], USER nonroot (65532)
WORKDIR /app

# Copy installed site-packages from builder
COPY --from=builder /usr/lib/python3.11/site-packages /usr/lib/python3.11/site-packages

# Copy shared startup_tracing library (for deterministic trace correlation)
COPY model_engine_server/common/startup_tracing /app/startup_tracing

# Copy watcher modules
COPY k8s_startup_watcher/config.py /app/config.py
COPY k8s_startup_watcher/watcher.py /app/watcher.py

# ENTRYPOINT is already /usr/bin/python, so CMD is just the script args
CMD ["/app/watcher.py"]
