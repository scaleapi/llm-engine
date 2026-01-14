"""
OpenTelemetry instrumentation for vLLM startup metrics.

This module provides lightweight instrumentation to capture startup phase timings
and emit them as OTel spans and metrics for analysis in Datadog.

All startup spans are linked to a common parent "pod_startup" span so they appear
in a single trace waterfall view in Datadog.
"""

import os

# For explicit span timestamps
import time
import time as time_module
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Generator, Optional

# Shared trace correlation utilities
from trace_correlation import (
    OTEL_AVAILABLE,
    create_deterministic_context,
    derive_span_id,
    derive_trace_id,
)

# OTel imports - gracefully handle missing dependencies
if OTEL_AVAILABLE:
    from opentelemetry import metrics, trace
    from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.trace import SpanKind, Status, StatusCode


@dataclass
class StartupContext:
    """Runtime context for startup telemetry."""

    endpoint_name: str
    model_name: str
    gpu_type: str
    num_gpus: int
    region: str
    pod_uid: str
    pod_name: str
    node_name: str

    @classmethod
    def from_env(cls) -> "StartupContext":
        """Create context from environment variables."""
        return cls(
            endpoint_name=os.environ.get("ENDPOINT_NAME", "unknown"),
            model_name=os.environ.get("MODEL_NAME", "unknown"),
            gpu_type=os.environ.get("GPU_TYPE", "unknown"),
            num_gpus=int(os.environ.get("NUM_GPUS", "1")),
            region=os.environ.get("AWS_REGION", os.environ.get("AWS_DEFAULT_REGION", "unknown")),
            pod_uid=os.environ.get("POD_UID", "unknown"),
            pod_name=os.environ.get("POD_NAME", "unknown"),
            node_name=os.environ.get("NODE_NAME", "unknown"),
        )


class StartupTelemetry:
    """Manages OpenTelemetry instrumentation for startup metrics."""

    def __init__(self):
        self._tracer: Optional[trace.Tracer] = None
        self._meter: Optional[metrics.Meter] = None
        self._histograms: dict = {}
        self._gauges: dict = {}
        self._context: Optional[StartupContext] = None
        self._start_time: float = time.perf_counter()
        self._start_time_ns: int = time_module.time_ns()  # Wall-clock for span timestamps
        self._initialized: bool = False
        # Root span for trace correlation - all child spans will be linked to this
        self._root_span = None
        self._root_span_context = None

    def init(
        self, ctx: Optional[StartupContext] = None, container_start_time_ns: Optional[int] = None
    ) -> None:
        """Initialize OTel SDK for startup instrumentation.

        Args:
            ctx: Optional startup context with endpoint/model metadata
            container_start_time_ns: Optional container start time in nanoseconds.
                If provided, the root span will use this as its start time so it
                properly encapsulates all child spans including s5cmd download.
                If not provided, falls back to CONTAINER_START_TS env var or current time.
        """
        if not OTEL_AVAILABLE:
            print("Skipping OTel init - dependencies not available")
            return

        endpoint = os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT")
        if not endpoint:
            print("Skipping OTel init - OTEL_EXPORTER_OTLP_ENDPOINT not set")
            return

        self._context = ctx or StartupContext.from_env()

        # Determine root span start time - use container start if available
        if container_start_time_ns:
            self._root_start_time_ns = container_start_time_ns
        else:
            # Try to get from environment (set by entrypoint.sh)
            container_start_ts = os.environ.get("CONTAINER_START_TS")
            if container_start_ts:
                self._root_start_time_ns = int(float(container_start_ts) * 1_000_000_000)
            else:
                self._root_start_time_ns = self._start_time_ns

        resource = Resource.create(
            {
                "service.name": "vllm-startup",
                "k8s.pod.uid": self._context.pod_uid,
                "k8s.pod.name": self._context.pod_name,
                "k8s.node.name": self._context.node_name,
                "endpoint_name": self._context.endpoint_name,
                "model_name": self._context.model_name,
            }
        )

        # Traces
        provider = TracerProvider(resource=resource)
        provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint)))
        trace.set_tracer_provider(provider)
        self._tracer = trace.get_tracer("vllm-startup")

        # Metrics
        reader = PeriodicExportingMetricReader(
            OTLPMetricExporter(endpoint=endpoint),
            export_interval_millis=5000,  # Export quickly for startup
        )
        meter_provider = MeterProvider(resource=resource, metric_readers=[reader])
        metrics.set_meter_provider(meter_provider)
        self._meter = metrics.get_meter("vllm-startup")

        # Create histograms for aggregate analysis
        # Namespaced under "vllm.startup" for Datadog organization
        self._histograms["download_duration"] = self._meter.create_histogram(
            "vllm.startup.download.duration_seconds",
            description="Model download duration (s5cmd from S3)",
            unit="s",
        )
        self._histograms["vllm_init_duration"] = self._meter.create_histogram(
            "vllm.startup.vllm_init.duration_seconds",
            description="vLLM initialization duration",
            unit="s",
        )
        self._histograms["python_init_duration"] = self._meter.create_histogram(
            "vllm.startup.python_init.duration_seconds",
            description="Python startup, module imports, arg parsing",
            unit="s",
        )
        self._histograms["total_duration"] = self._meter.create_histogram(
            "vllm.startup.total.duration_seconds",
            description="Total startup duration from container start to server ready",
            unit="s",
        )

        # Also create gauges for simpler querying in Datadog
        self._gauges["download_duration"] = self._meter.create_gauge(
            "vllm.startup.download.duration",
            description="Model download duration (s5cmd from S3)",
            unit="s",
        )
        self._gauges["vllm_init_duration"] = self._meter.create_gauge(
            "vllm.startup.vllm_init.duration",
            description="vLLM initialization duration",
            unit="s",
        )
        self._gauges["python_init_duration"] = self._meter.create_gauge(
            "vllm.startup.python_init.duration",
            description="Python startup, module imports, arg parsing",
            unit="s",
        )
        self._gauges["total_duration"] = self._meter.create_gauge(
            "vllm.startup.total.duration",
            description="Total startup duration",
            unit="s",
        )

        self._initialized = True

        # Create root span for trace correlation - all child spans will be linked to this
        # Use deterministic trace ID from pod_uid so K8s event watcher spans
        # appear in the same trace
        deterministic_ctx = self._get_deterministic_context()

        self._root_span = self._tracer.start_span(
            "in_container_startup",
            kind=SpanKind.INTERNAL,
            context=deterministic_ctx,  # Use deterministic trace ID
            start_time=self._root_start_time_ns,  # Use container start time to encapsulate all phases
        )
        for k, v in self._get_common_attributes().items():
            self._root_span.set_attribute(k, v)
        self._root_span.set_attribute("pod_uid", self._context.pod_uid)
        self._root_span.set_attribute("pod_name", self._context.pod_name)
        # Store context for child span creation
        self._root_span_context = trace.set_span_in_context(self._root_span)

        trace_id = self.get_trace_id()
        trace_id_hex = format(trace_id, "032x") if trace_id else "unknown"
        print(f"OTel startup telemetry initialized, endpoint={endpoint}, trace_id={trace_id_hex}")

    def get_trace_id(self) -> Optional[int]:
        """Get deterministic trace ID from pod UID for correlation.

        Returns a 128-bit integer trace ID derived from pod UID.
        Uses shared trace_correlation module for consistency with K8s event watcher.
        """
        if not self._context or self._context.pod_uid == "unknown":
            return None
        return derive_trace_id(self._context.pod_uid)

    def get_span_id(self, suffix: str = "") -> int:
        """Get deterministic span ID for a given span name.

        Returns a 64-bit integer span ID.
        Uses shared trace_correlation module for consistency with K8s event watcher.
        """
        if not self._context:
            return 0
        return derive_span_id(self._context.pod_uid, suffix)

    def _get_deterministic_context(self):
        """Get trace context with deterministic trace ID from pod_uid.

        Uses shared trace_correlation module for consistency with K8s event watcher.
        """
        if not self._context or self._context.pod_uid == "unknown":
            return None
        return create_deterministic_context(self._context.pod_uid)

    def _get_common_attributes(self) -> dict:
        """Get common attributes for spans and metrics."""
        if not self._context:
            return {}
        return {
            "endpoint_name": self._context.endpoint_name,
            "model_name": self._context.model_name,
            "gpu_type": self._context.gpu_type,
            "num_gpus": self._context.num_gpus,
            "region": self._context.region,
            "node_name": self._context.node_name,
        }

    def create_child_span(
        self, name: str, start_time_ns: int, end_time_ns: int, attributes: Optional[dict] = None
    ):
        """Create a child span with explicit timestamps, linked to the root pod_startup span.

        This ensures all startup phase spans appear in the same trace waterfall.
        """
        if not self._tracer or not self._initialized or not self._root_span_context:
            return None

        attrs = {**self._get_common_attributes(), **(attributes or {})}

        # Create span with root span as parent
        span = self._tracer.start_span(
            name,
            kind=SpanKind.INTERNAL,
            context=self._root_span_context,
            start_time=start_time_ns,
        )
        for k, v in attrs.items():
            span.set_attribute(k, v)
        span.set_status(Status(StatusCode.OK))
        span.end(end_time=end_time_ns)
        return span

    @contextmanager
    def span(self, name: str, attributes: Optional[dict] = None) -> Generator:
        """Context manager for startup phase spans, linked to root span."""
        if not self._tracer or not self._initialized:
            yield None
            return

        attrs = {**self._get_common_attributes(), **(attributes or {})}

        # Use root span context as parent if available
        ctx = self._root_span_context if self._root_span_context else None
        with self._tracer.start_as_current_span(name, kind=SpanKind.INTERNAL, context=ctx) as span:
            for k, v in attrs.items():
                span.set_attribute(k, v)
            start = time.perf_counter()
            try:
                yield span
                span.set_status(Status(StatusCode.OK))
            except Exception as e:
                span.set_status(Status(StatusCode.ERROR, str(e)))
                span.record_exception(e)
                raise
            finally:
                duration = time.perf_counter() - start
                span.set_attribute("duration_seconds", duration)

    def record_metric(self, name: str, value: float, extra_attrs: Optional[dict] = None) -> None:
        """Record a histogram metric and gauge."""
        if not self._initialized:
            return

        attrs = {**self._get_common_attributes(), **(extra_attrs or {})}
        # Filter to low-cardinality attributes only
        metric_attrs = {
            k: v
            for k, v in attrs.items()
            if k in ("endpoint_name", "model_name", "gpu_type", "region")
        }

        # Record to histogram
        if name in self._histograms:
            self._histograms[name].record(value, metric_attrs)

        # Also record to gauge for simpler Datadog queries
        if name in self._gauges:
            self._gauges[name].set(value, metric_attrs)

    def record_startup_complete(self) -> float:
        """Record that startup is complete, return total duration.

        Also closes the root pod_startup span to finalize the trace.
        """
        end_time_ns = time_module.time_ns()
        # Calculate total duration from container start (not Python start)
        if hasattr(self, "_root_start_time_ns"):
            total_duration = (end_time_ns - self._root_start_time_ns) / 1_000_000_000
        else:
            total_duration = time.perf_counter() - self._start_time

        if self._initialized:
            self.record_metric("total_duration", total_duration)

            # Create "startup_complete" as a child span marking the end
            self.create_child_span(
                "startup_complete",
                start_time_ns=end_time_ns - 1_000_000,  # 1ms before end
                end_time_ns=end_time_ns,
                attributes={"total_duration_seconds": total_duration},
            )

            # Close the root span to finalize the trace waterfall
            if self._root_span:
                self._root_span.set_attribute("total_duration_seconds", total_duration)
                self._root_span.set_status(Status(StatusCode.OK))
                self._root_span.end(end_time=end_time_ns)

            print(f"Startup complete: total_duration={total_duration:.2f}s")

        return total_duration

    def flush(self) -> None:
        """Force flush all telemetry data."""
        if not self._initialized:
            return

        try:
            # Force flush traces
            provider = trace.get_tracer_provider()
            if hasattr(provider, "force_flush"):
                provider.force_flush(timeout_millis=5000)

            # Force flush metrics
            meter_provider = metrics.get_meter_provider()
            if hasattr(meter_provider, "force_flush"):
                meter_provider.force_flush(timeout_millis=5000)
        except Exception as e:
            print(f"Error flushing telemetry: {e}")


# Global instance for convenience
_telemetry = StartupTelemetry()


def init_startup_telemetry(
    ctx: Optional[StartupContext] = None, container_start_time_ns: Optional[int] = None
) -> StartupTelemetry:
    """Initialize the global startup telemetry instance."""
    _telemetry.init(ctx, container_start_time_ns)
    return _telemetry


def get_telemetry() -> StartupTelemetry:
    """Get the global startup telemetry instance."""
    return _telemetry
