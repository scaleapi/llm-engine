# K8s Startup Watcher - Generic pod startup metrics collection
#
# Watches K8s pod events and emits startup metrics via OTLP.
# Uses deterministic trace IDs for correlation with in-container metrics.
#
# Multi-stage build: builder (has pip) → slim runtime
#
# Build:
#   docker build -f k8s_startup_watcher/Dockerfile -t k8s-startup-watcher .
#
# --- Builder: install pip dependencies ---
FROM python:3.13-slim AS builder

WORKDIR /app

RUN pip install --no-cache-dir \
    kubernetes~=29.0.0 \
    opentelemetry-api~=1.27.0 \
    opentelemetry-sdk~=1.27.0 \
    opentelemetry-exporter-otlp~=1.27.0

# --- Runtime ---
FROM python:3.13-slim

WORKDIR /app

# Copy installed site-packages from builder
COPY --from=builder /usr/local/lib/python3.13/site-packages /usr/local/lib/python3.13/site-packages

# Copy shared startup_tracing library (for deterministic trace correlation)
COPY model_engine_server/common/startup_tracing /app/startup_tracing

# Copy watcher modules
COPY k8s_startup_watcher/config.py /app/config.py
COPY k8s_startup_watcher/watcher.py /app/watcher.py

RUN useradd --create-home --shell /bin/bash nonroot && \
    chmod -R 777 /app
USER nonroot
ENTRYPOINT ["python"]
CMD ["/app/watcher.py"]
