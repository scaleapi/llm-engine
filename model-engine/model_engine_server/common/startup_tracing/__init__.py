"""
Startup tracing library for deterministic trace correlation.

This library enables multiple processes (K8s watchers, in-container code) to emit
spans that appear in the same trace by using deterministic IDs derived from pod_uid.

Usage for endpoint creators:

    from model_engine_server.common.startup_tracing import (
        StartupTracer,
        SPAN_IN_CONTAINER_STARTUP,
    )

    # Initialize once at container start
    tracer = StartupTracer.from_env()

    # Create child spans under in_container_startup
    with tracer.span("my_phase") as span:
        do_work()

    # Or with explicit timestamps
    tracer.create_span("download", start_ns, end_ns, {"size_mb": 100})

    # When startup is complete
    tracer.complete()
"""

from .correlation import (
    OTEL_AVAILABLE,
    create_parent_context,
    derive_span_id,
    derive_trace_id,
    format_trace_id,
)
from .deterministic_span import DeterministicSpan
from .tracer import StartupTracer

# Well-known span suffixes - these form the standard span hierarchy
SPAN_POD_STARTUP = "pod_startup"  # Root span for entire pod lifecycle
SPAN_IN_CONTAINER_STARTUP = "in_container_startup"  # Parent for all in-container phases

__all__ = [
    # Core utilities
    "derive_trace_id",
    "derive_span_id",
    "create_parent_context",
    "format_trace_id",
    "DeterministicSpan",
    "OTEL_AVAILABLE",
    # Main interface
    "StartupTracer",
    # Well-known span names
    "SPAN_POD_STARTUP",
    "SPAN_IN_CONTAINER_STARTUP",
]
