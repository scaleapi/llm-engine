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
from typing import Any, Dict, Optional, Sequence, Tuple

# OTel imports - gracefully handle missing dependencies
try:
    from opentelemetry import trace
    from opentelemetry.context import Context
    from opentelemetry.sdk.trace import ReadableSpan, Resource
    from opentelemetry.sdk.util.instrumentation import InstrumentationScope
    from opentelemetry.trace import (
        Link,
        NonRecordingSpan,
        SpanContext,
        SpanKind,
        Status,
        StatusCode,
        TraceFlags,
    )

    OTEL_AVAILABLE = True
except ImportError:
    OTEL_AVAILABLE = False
    Context = None  # type: ignore
    ReadableSpan = None  # type: ignore


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


# Well-known span suffixes for deterministic span IDs
SPAN_K8S_TOTAL_TO_RUNNING = "k8s_total_to_running"
SPAN_IN_CONTAINER_STARTUP = "in_container_startup"


def create_k8s_parent_context(pod_uid: str) -> Optional[Context]:
    """Create a trace context where k8s_total_to_running is the parent span.

    This allows in-container telemetry spans to appear as children of the
    k8s_total_to_running span in the trace waterfall.

    The k8s_total_to_running span covers the entire time from pod creation
    to container running, making it the natural parent for all startup phases.

    Args:
        pod_uid: Kubernetes pod UID

    Returns:
        OpenTelemetry Context with k8s_total_to_running as parent,
        or None if OpenTelemetry is not available
    """
    if not OTEL_AVAILABLE:
        return None

    trace_id = derive_trace_id(pod_uid)
    # Use k8s_total_to_running as the parent span ID
    span_id = derive_span_id(pod_uid, SPAN_K8S_TOTAL_TO_RUNNING)

    span_context = SpanContext(
        trace_id=trace_id,
        span_id=span_id,
        is_remote=True,  # Treat as remote so new spans become children
        trace_flags=TraceFlags(TraceFlags.SAMPLED),
    )

    # Create a non-recording span with this context to use as parent
    parent_span = NonRecordingSpan(span_context)
    return trace.set_span_in_context(parent_span)


class DeterministicSpan:
    """A span implementation with deterministic span_id for manual export.

    OTel SDK auto-generates span_id, which breaks our correlation scheme.
    This class creates a span-like object with a deterministic span_id that
    can be exported directly via the span processor.

    Usage:
        span = DeterministicSpan(
            name="k8s_total_to_running",
            pod_uid="abc-123",
            span_suffix="k8s_total_to_running",
            parent_suffix="root",
            start_time_ns=...,
            end_time_ns=...,
            attributes={...},
            resource=resource,
        )
        span_processor.on_end(span)
    """

    def __init__(
        self,
        name: str,
        pod_uid: str,
        span_suffix: str,
        parent_suffix: str,
        start_time_ns: int,
        end_time_ns: int,
        attributes: Optional[Dict[str, Any]] = None,
        resource: Optional["Resource"] = None,
        instrumentation_scope: Optional["InstrumentationScope"] = None,
    ):
        if not OTEL_AVAILABLE:
            raise RuntimeError("OpenTelemetry not available")

        self._name = name
        self._start_time = start_time_ns
        self._end_time = end_time_ns
        self._attributes = dict(attributes) if attributes else {}
        self._resource = resource or Resource.create({})
        self._instrumentation_scope = instrumentation_scope or InstrumentationScope(
            "vllm-event-watcher"
        )
        self._status = Status(StatusCode.OK)
        self._kind = SpanKind.INTERNAL
        self._events: Tuple = ()
        self._links: Tuple = ()

        # Create deterministic span context
        trace_id = derive_trace_id(pod_uid)
        span_id = derive_span_id(pod_uid, span_suffix)
        parent_span_id = derive_span_id(pod_uid, parent_suffix)

        self._context = SpanContext(
            trace_id=trace_id,
            span_id=span_id,
            is_remote=False,
            trace_flags=TraceFlags(TraceFlags.SAMPLED),
        )

        self._parent = SpanContext(
            trace_id=trace_id,
            span_id=parent_span_id,
            is_remote=True,
            trace_flags=TraceFlags(TraceFlags.SAMPLED),
        )

    # ReadableSpan interface implementation
    @property
    def name(self) -> str:
        return self._name

    @property
    def context(self) -> "SpanContext":
        return self._context

    def get_span_context(self) -> "SpanContext":
        return self._context

    @property
    def parent(self) -> Optional["SpanContext"]:
        return self._parent

    @property
    def start_time(self) -> int:
        return self._start_time

    @property
    def end_time(self) -> int:
        return self._end_time

    @property
    def attributes(self) -> Dict[str, Any]:
        return self._attributes

    @property
    def events(self) -> Sequence:
        return self._events

    @property
    def links(self) -> Sequence["Link"]:
        return self._links

    @property
    def kind(self) -> "SpanKind":
        return self._kind

    @property
    def status(self) -> "Status":
        return self._status

    @property
    def resource(self) -> "Resource":
        return self._resource

    @property
    def instrumentation_scope(self) -> Optional["InstrumentationScope"]:
        return self._instrumentation_scope

    # Alias for older SDK versions
    @property
    def instrumentation_info(self) -> Optional["InstrumentationScope"]:
        return self._instrumentation_scope

    # Additional ReadableSpan properties required by exporters
    @property
    def dropped_attributes(self) -> int:
        return 0

    @property
    def dropped_events(self) -> int:
        return 0

    @property
    def dropped_links(self) -> int:
        return 0
