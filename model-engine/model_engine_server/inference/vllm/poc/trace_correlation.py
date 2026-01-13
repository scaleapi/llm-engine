"""
Shared trace correlation utilities for vLLM startup metrics.

This module provides deterministic trace ID and span ID generation from pod_uid,
enabling both in-container telemetry and K8s event watchers to emit spans
that appear in the same trace.

Algorithm:
- Trace ID (128-bit): SHA256(pod_uid)[:32] as int
- Span ID (64-bit): SHA256(pod_uid:suffix)[:16] as int
"""

import hashlib
from typing import Optional

# OTel imports - gracefully handle missing dependencies
try:
    from opentelemetry import trace
    from opentelemetry.context import Context
    from opentelemetry.trace import NonRecordingSpan, SpanContext, TraceFlags

    OTEL_AVAILABLE = True
except ImportError:
    OTEL_AVAILABLE = False
    Context = None  # type: ignore


def derive_trace_id(pod_uid: str) -> int:
    """Derive deterministic 128-bit trace ID from pod UID.

    Uses SHA256 hash truncated to 128 bits (32 hex chars).
    Both K8s event watcher and in-container telemetry use this
    same algorithm to correlate spans into a single trace.

    Args:
        pod_uid: Kubernetes pod UID (e.g., from metadata.uid)

    Returns:
        128-bit integer trace ID
    """
    hex_id = hashlib.sha256(pod_uid.encode()).hexdigest()[:32]
    return int(hex_id, 16)


def derive_span_id(pod_uid: str, suffix: str = "") -> int:
    """Derive deterministic 64-bit span ID for a given span name.

    Uses SHA256 hash of pod_uid:suffix truncated to 64 bits (16 hex chars).

    Args:
        pod_uid: Kubernetes pod UID
        suffix: Span identifier suffix (e.g., "root", "download", "init")

    Returns:
        64-bit integer span ID
    """
    data = f"{pod_uid}:{suffix}"
    hex_id = hashlib.sha256(data.encode()).hexdigest()[:16]
    return int(hex_id, 16)


def create_deterministic_context(pod_uid: str) -> Optional[Context]:
    """Create a trace context with deterministic trace ID from pod_uid.

    This allows the K8s event watcher and in-container telemetry to
    emit spans that appear in the same trace.

    The returned context can be used as the parent context when starting
    new spans, ensuring they share the same trace ID.

    Args:
        pod_uid: Kubernetes pod UID

    Returns:
        OpenTelemetry Context with deterministic trace/span IDs,
        or None if OpenTelemetry is not available
    """
    if not OTEL_AVAILABLE:
        return None

    trace_id = derive_trace_id(pod_uid)
    # Use "root" suffix to create consistent parent span ID
    span_id = derive_span_id(pod_uid, "root")

    span_context = SpanContext(
        trace_id=trace_id,
        span_id=span_id,
        is_remote=True,  # Treat as remote so new spans become children
        trace_flags=TraceFlags(TraceFlags.SAMPLED),
    )

    # Create a non-recording span with this context to use as parent
    parent_span = NonRecordingSpan(span_context)
    return trace.set_span_in_context(parent_span)


def format_trace_id(pod_uid: str) -> str:
    """Format trace ID as hex string for logging.

    Args:
        pod_uid: Kubernetes pod UID

    Returns:
        32-character hex string trace ID
    """
    trace_id = derive_trace_id(pod_uid)
    return format(trace_id, "032x")
