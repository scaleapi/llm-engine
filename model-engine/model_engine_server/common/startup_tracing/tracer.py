"""
StartupTracer - Main interface for endpoint startup instrumentation.

Provides a simple API for creating spans that correlate with K8s events
and other processes using deterministic trace/span IDs.
"""

import os
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Dict, Generator, Optional

from .correlation import OTEL_AVAILABLE, create_parent_context, format_trace_id
from .deterministic_span import DeterministicSpan

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

# Well-known span suffix for in-container startup
SPAN_IN_CONTAINER_STARTUP = "in_container_startup"


@dataclass
class StartupContext:
    """Context for startup tracing - identifies the pod and endpoint."""

    pod_uid: str
    pod_name: str = "unknown"
    node_name: str = "unknown"
    endpoint_name: str = "unknown"
    model_name: str = "unknown"
    gpu_type: str = "unknown"
    num_gpus: int = 1
    region: str = "unknown"

    @classmethod
    def from_env(cls) -> "StartupContext":
        """Create context from standard environment variables."""
        return cls(
            pod_uid=os.environ.get("POD_UID", "unknown"),
            pod_name=os.environ.get("POD_NAME", "unknown"),
            node_name=os.environ.get("NODE_NAME", "unknown"),
            endpoint_name=os.environ.get("ENDPOINT_NAME", "unknown"),
            model_name=os.environ.get("MODEL_NAME", "unknown"),
            gpu_type=os.environ.get("GPU_TYPE", "unknown"),
            num_gpus=int(os.environ.get("NUM_GPUS", "1")),
            region=os.environ.get("AWS_REGION", os.environ.get("AWS_DEFAULT_REGION", "unknown")),
        )


class StartupTracer:
    """Main interface for startup tracing.

    Handles OTel SDK initialization and provides methods for creating
    spans that correlate with K8s events via deterministic IDs.

    Usage:
        tracer = StartupTracer.from_env(service_name="vllm-startup")

        # Create spans as children of in_container_startup
        tracer.create_span("download", start_ns, end_ns, {"size_mb": 100})

        # Or use context manager
        with tracer.span("init_phase"):
            do_initialization()

        # Record metrics
        tracer.record_metric("download_duration", 25.5)

        # When startup completes
        tracer.complete()
    """

    def __init__(
        self,
        context: StartupContext,
        service_name: str = "startup-tracing",
        container_start_time_ns: Optional[int] = None,
    ):
        self._context = context
        self._service_name = service_name
        self._initialized = False
        self._tracer = None
        self._meter = None
        self._resource = None
        self._gauges: Dict[str, Any] = {}
        self._parent_context = None

        # Container start time for root span
        if container_start_time_ns:
            self._start_time_ns = container_start_time_ns
        else:
            # Check env var (set by wrapper scripts)
            env_start = os.environ.get("CONTAINER_START_TS")
            if env_start:
                self._start_time_ns = int(float(env_start) * 1_000_000_000)
            else:
                self._start_time_ns = time.time_ns()

    @classmethod
    def from_env(
        cls,
        service_name: str = "startup-tracing",
        container_start_time_ns: Optional[int] = None,
    ) -> "StartupTracer":
        """Create tracer from environment variables."""
        ctx = StartupContext.from_env()
        tracer = cls(ctx, service_name, container_start_time_ns)
        tracer.init()
        return tracer

    def init(self) -> bool:
        """Initialize OTel SDK. Returns True if successful."""
        if not OTEL_AVAILABLE:
            print(f"[{self._service_name}] Skipping OTel init - dependencies not available")
            return False

        endpoint = os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT")
        if not endpoint:
            # Fallback: construct from DD_AGENT_HOST (Datadog Agent's OTLP receiver)
            dd_agent_host = os.environ.get("DD_AGENT_HOST")
            if dd_agent_host:
                # Handle IPv6 addresses (need brackets)
                if ":" in dd_agent_host and not dd_agent_host.startswith("["):
                    endpoint = f"http://[{dd_agent_host}]:4317"
                else:
                    endpoint = f"http://{dd_agent_host}:4317"
                print(f"[{self._service_name}] Using DD_AGENT_HOST for OTLP endpoint: {endpoint}")
            else:
                print(
                    f"[{self._service_name}] Skipping OTel init - OTEL_EXPORTER_OTLP_ENDPOINT not set"
                )
                return False

        self._resource = Resource.create(
            {
                "service.name": self._service_name,
                "k8s.pod.uid": self._context.pod_uid,
                "k8s.pod.name": self._context.pod_name,
                "k8s.node.name": self._context.node_name,
                "endpoint_name": self._context.endpoint_name,
                "model_name": self._context.model_name,
            }
        )

        # Determine if we need insecure mode (http:// vs https://)
        use_insecure = endpoint.startswith("http://")

        # Traces
        provider = TracerProvider(resource=self._resource)
        provider.add_span_processor(
            BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint, insecure=use_insecure))
        )
        trace.set_tracer_provider(provider)
        self._tracer = trace.get_tracer(self._service_name)

        # Metrics
        reader = PeriodicExportingMetricReader(
            OTLPMetricExporter(endpoint=endpoint, insecure=use_insecure),
            export_interval_millis=5000,
        )
        meter_provider = MeterProvider(resource=self._resource, metric_readers=[reader])
        metrics.set_meter_provider(meter_provider)
        self._meter = meter_provider.get_meter(self._service_name)

        # Create parent context - all spans will be children of in_container_startup
        self._parent_context = create_parent_context(
            self._context.pod_uid, SPAN_IN_CONTAINER_STARTUP
        )

        self._initialized = True
        trace_id = format_trace_id(self._context.pod_uid)
        print(f"[{self._service_name}] OTel initialized, trace_id={trace_id}")
        return True

    @property
    def trace_id(self) -> str:
        """Get the trace ID as a hex string."""
        return format_trace_id(self._context.pod_uid)

    def _common_attributes(self) -> Dict[str, Any]:
        """Get common attributes for all spans/metrics."""
        return {
            "endpoint_name": self._context.endpoint_name,
            "model_name": self._context.model_name,
            "gpu_type": self._context.gpu_type,
            "num_gpus": self._context.num_gpus,
            "region": self._context.region,
            "node_name": self._context.node_name,
            "pod_name": self._context.pod_name,
        }

    def create_span(
        self,
        name: str,
        start_time_ns: int,
        end_time_ns: int,
        attributes: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Create a child span with explicit timestamps."""
        if not self._initialized or not self._tracer:
            return

        attrs = {**self._common_attributes(), **(attributes or {})}

        span = self._tracer.start_span(
            name,
            kind=SpanKind.INTERNAL,
            context=self._parent_context,
            start_time=start_time_ns,
        )
        for k, v in attrs.items():
            span.set_attribute(k, v)
        span.set_status(Status(StatusCode.OK))
        span.end(end_time=end_time_ns)

    @contextmanager
    def span(self, name: str, attributes: Optional[Dict[str, Any]] = None) -> Generator:
        """Context manager for creating spans."""
        if not self._initialized or not self._tracer:
            yield None
            return

        attrs = {**self._common_attributes(), **(attributes or {})}

        with self._tracer.start_as_current_span(
            name, kind=SpanKind.INTERNAL, context=self._parent_context
        ) as span:
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

    def create_gauge(self, name: str, description: str = "", unit: str = "s") -> None:
        """Create a gauge metric for later recording."""
        if not self._initialized or not self._meter:
            return
        self._gauges[name] = self._meter.create_gauge(name, description=description, unit=unit)

    def record_metric(
        self, name: str, value: float, extra_attrs: Optional[Dict[str, Any]] = None
    ) -> None:
        """Record a metric value."""
        if not self._initialized:
            return

        # Auto-create gauge if needed
        if name not in self._gauges and self._meter:
            self._gauges[name] = self._meter.create_gauge(name)

        if name in self._gauges:
            attrs = {**self._common_attributes(), **(extra_attrs or {})}
            # Filter to low-cardinality for metrics
            metric_attrs = {
                k: v
                for k, v in attrs.items()
                if k in ("endpoint_name", "model_name", "gpu_type", "region", "pod_name")
            }
            self._gauges[name].set(value, metric_attrs)

    def complete(self) -> float:
        """Mark startup complete. Emits the in_container_startup root span.

        Returns total duration in seconds.
        """
        end_time_ns = time.time_ns()
        total_duration = (end_time_ns - self._start_time_ns) / 1_000_000_000

        if not self._initialized:
            return total_duration

        # Create the in_container_startup span that all child spans link to
        root_span = DeterministicSpan(
            name="in_container_startup",
            unique_id=self._context.pod_uid,
            span_suffix=SPAN_IN_CONTAINER_STARTUP,
            parent_suffix="pod_startup",  # Parent is the overall pod_startup span
            start_time_ns=self._start_time_ns,
            end_time_ns=end_time_ns,
            attributes={
                **self._common_attributes(),
                "pod_uid": self._context.pod_uid,
                "pod_name": self._context.pod_name,
                "total_duration_seconds": total_duration,
            },
            resource=self._resource,
            service_name=self._service_name,
        )

        # Export via span processor
        provider = trace.get_tracer_provider()
        if hasattr(provider, "_active_span_processor"):
            provider._active_span_processor.on_end(root_span)

        print(f"[{self._service_name}] Startup complete: {total_duration:.2f}s")
        return total_duration

    def flush(self, timeout_ms: int = 5000) -> None:
        """Force flush all telemetry data."""
        if not self._initialized:
            return

        try:
            provider = trace.get_tracer_provider()
            if hasattr(provider, "force_flush"):
                provider.force_flush(timeout_millis=timeout_ms)

            meter_provider = metrics.get_meter_provider()
            if hasattr(meter_provider, "force_flush"):
                meter_provider.force_flush(timeout_millis=timeout_ms)
        except Exception as e:
            print(f"[{self._service_name}] Error flushing telemetry: {e}")
