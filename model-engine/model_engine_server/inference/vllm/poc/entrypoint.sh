#!/bin/bash
# Instrumented entrypoint that captures startup phase timings

set -e

# Helper function to log startup metrics (use stderr for immediate flush)
log_startup() {
    echo "[STARTUP METRICS] $1" >&2
}

# Construct OTLP endpoint from DD_AGENT_HOST (Datadog Agent's OTLP receiver)
# Using gRPC on port 4317 (our Python code uses grpc exporters)
# IPv6 addresses must be wrapped in brackets for URLs
if [ -n "$DD_AGENT_HOST" ]; then
    if [[ "$DD_AGENT_HOST" == *:* ]]; then
        # IPv6 address - wrap in brackets
        export OTEL_EXPORTER_OTLP_ENDPOINT="http://[${DD_AGENT_HOST}]:4317"
    else
        # IPv4 address or hostname
        export OTEL_EXPORTER_OTLP_ENDPOINT="http://${DD_AGENT_HOST}:4317"
    fi
    log_startup "OTLP endpoint set to ${OTEL_EXPORTER_OTLP_ENDPOINT}"
fi

# Record container start time
export CONTAINER_START_TS=$(date +%s.%N)
log_startup "Container start at $(date -u +%Y-%m-%dT%H:%M:%S.%3NZ)"

# ============================================
# Phase 1: Model Download (s5cmd)
# ============================================
log_startup "Starting model download from S3..."
export DOWNLOAD_START_TS=$(date +%s.%N)

# Run s5cmd with the provided arguments
# MODEL_S3_PATH and MODEL_LOCAL_PATH are set via environment
s5cmd --numworkers 512 cp --concurrency 10 \
    --include "*.model" \
    --include "*.model.v*" \
    --include "*.json" \
    --include "*.safetensors" \
    --include "*.txt" \
    --exclude "optimizer*" \
    "${MODEL_S3_PATH}/*" "${MODEL_LOCAL_PATH}"

export DOWNLOAD_END_TS=$(date +%s.%N)
DOWNLOAD_DURATION=$(echo "$DOWNLOAD_END_TS - $DOWNLOAD_START_TS" | bc)
log_startup "Model download complete: ${DOWNLOAD_DURATION}s"

# Calculate download size
DOWNLOAD_SIZE_MB=$(du -sm "${MODEL_LOCAL_PATH}" | cut -f1)
log_startup "Downloaded ${DOWNLOAD_SIZE_MB}MB"

# Export timing info for the Python server
export STARTUP_DOWNLOAD_DURATION_S="${DOWNLOAD_DURATION}"
export STARTUP_DOWNLOAD_SIZE_MB="${DOWNLOAD_SIZE_MB}"

# ============================================
# Phase 2: Start vLLM Server (instrumented)
# ============================================
log_startup "Starting vLLM server..."

# Run the instrumented vLLM server with all remaining arguments
exec python /workspace/vllm_server_instrumented.py \
    --model "${MODEL_LOCAL_PATH}" \
    --served-model-name "${SERVED_MODEL_NAME}" \
    --port "${VLLM_PORT:-5005}" \
    --host "::" \
    ${VLLM_EXTRA_ARGS}
