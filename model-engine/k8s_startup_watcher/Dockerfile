# K8s Startup Watcher - Generic pod startup metrics collection
#
# Watches K8s pod events and emits startup metrics via OTLP.
# Uses deterministic trace IDs for correlation with in-container metrics.
#
# Build:
#   docker build -f k8s_startup_watcher/Dockerfile -t k8s-startup-watcher .
#
FROM python:3.11-slim

WORKDIR /app

# Install dependencies
RUN pip install --no-cache-dir \
    kubernetes~=29.0.0 \
    opentelemetry-api~=1.27.0 \
    opentelemetry-sdk~=1.27.0 \
    opentelemetry-exporter-otlp~=1.27.0

# Copy shared startup_tracing library (for deterministic trace correlation)
COPY model_engine_server/common/startup_tracing /app/startup_tracing

# Copy watcher modules
COPY k8s_startup_watcher/config.py /app/config.py
COPY k8s_startup_watcher/watcher.py /app/watcher.py

CMD ["python", "/app/watcher.py"]
