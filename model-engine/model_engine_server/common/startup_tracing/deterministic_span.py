"""
DeterministicSpan - A span with predetermined trace/span IDs for manual export.

OTel SDK auto-generates span IDs, which breaks deterministic correlation.
This class creates a ReadableSpan-compatible object with controlled IDs.
"""

from typing import Any, Dict, Optional, Sequence, Tuple

from .correlation import OTEL_AVAILABLE, derive_span_id, derive_trace_id

if OTEL_AVAILABLE:
    from opentelemetry.sdk.trace import Resource
    from opentelemetry.sdk.util.instrumentation import InstrumentationScope
    from opentelemetry.trace import Link, SpanContext, SpanKind, Status, StatusCode, TraceFlags


class DeterministicSpan:
    """A ReadableSpan with deterministic span_id for manual export.

    Usage:
        span = DeterministicSpan(
            name="my_span",
            unique_id="pod-uid-123",
            span_suffix="my_span",
            parent_suffix="root",
            start_time_ns=start,
            end_time_ns=end,
            attributes={"key": "value"},
        )
        span_processor.on_end(span)
    """

    def __init__(
        self,
        name: str,
        unique_id: str,
        span_suffix: str,
        parent_suffix: str,
        start_time_ns: int,
        end_time_ns: int,
        attributes: Optional[Dict[str, Any]] = None,
        resource: Optional["Resource"] = None,
        service_name: str = "startup-tracing",
    ):
        if not OTEL_AVAILABLE:
            raise RuntimeError("OpenTelemetry not available")

        self._name = name
        self._start_time = start_time_ns
        self._end_time = end_time_ns
        self._attributes = dict(attributes) if attributes else {}
        self._resource = resource or Resource.create({"service.name": service_name})
        self._instrumentation_scope = InstrumentationScope(service_name)
        self._status = Status(StatusCode.OK)
        self._kind = SpanKind.INTERNAL
        self._events: Tuple = ()
        self._links: Tuple = ()

        # Create deterministic span context
        trace_id = derive_trace_id(unique_id)
        span_id = derive_span_id(unique_id, span_suffix)
        parent_span_id = derive_span_id(unique_id, parent_suffix)

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

    # ReadableSpan interface
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

    @property
    def instrumentation_info(self) -> Optional["InstrumentationScope"]:
        return self._instrumentation_scope

    @property
    def dropped_attributes(self) -> int:
        return 0

    @property
    def dropped_events(self) -> int:
        return 0

    @property
    def dropped_links(self) -> int:
        return 0
