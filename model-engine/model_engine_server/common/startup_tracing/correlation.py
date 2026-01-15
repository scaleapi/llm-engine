"""
Core trace correlation utilities.

Provides deterministic trace ID and span ID generation from a unique identifier
(typically pod_uid), enabling distributed tracing across processes.
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


def derive_trace_id(unique_id: str) -> int:
    """Derive deterministic 128-bit trace ID from a unique identifier.

    Uses SHA256 hash truncated to 128 bits (32 hex chars).

    Args:
        unique_id: Unique identifier (e.g., pod_uid, request_id)

    Returns:
        128-bit integer trace ID
    """
    hex_id = hashlib.sha256(unique_id.encode()).hexdigest()[:32]
    return int(hex_id, 16)


def derive_span_id(unique_id: str, suffix: str = "") -> int:
    """Derive deterministic 64-bit span ID.

    Uses SHA256 hash of unique_id:suffix truncated to 64 bits.

    Args:
        unique_id: Unique identifier (e.g., pod_uid)
        suffix: Span identifier (e.g., "root", "download", "init")

    Returns:
        64-bit integer span ID
    """
    data = f"{unique_id}:{suffix}"
    hex_id = hashlib.sha256(data.encode()).hexdigest()[:16]
    return int(hex_id, 16)


def create_parent_context(unique_id: str, parent_suffix: str) -> Optional["Context"]:
    """Create a trace context with a deterministic parent span.

    This enables creating child spans that link to a known parent,
    even if the parent span is created by a different process.

    Args:
        unique_id: Unique identifier for trace correlation
        parent_suffix: Suffix for deriving the parent span ID

    Returns:
        OpenTelemetry Context with deterministic parent span,
        or None if OpenTelemetry is not available
    """
    if not OTEL_AVAILABLE:
        return None

    trace_id = derive_trace_id(unique_id)
    span_id = derive_span_id(unique_id, parent_suffix)

    span_context = SpanContext(
        trace_id=trace_id,
        span_id=span_id,
        is_remote=True,
        trace_flags=TraceFlags(TraceFlags.SAMPLED),
    )

    parent_span = NonRecordingSpan(span_context)
    return trace.set_span_in_context(parent_span)


def format_trace_id(unique_id: str) -> str:
    """Format trace ID as 32-character hex string for logging."""
    return format(derive_trace_id(unique_id), "032x")
