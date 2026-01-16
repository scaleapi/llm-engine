"""
vLLM Startup Metrics - uses the generic startup_tracing library.

This module provides vLLM-specific startup instrumentation:
- s3_download: Model download from S3
- python_init: Python startup and imports
- vllm_init: vLLM engine initialization

Usage in vllm_startup_wrapper.py:
    from startup_telemetry import VLLMStartupMetrics

    metrics = VLLMStartupMetrics.init()
    metrics.record_download(start_ns, end_ns, size_mb)
    metrics.flush()  # Before exec

Usage in vllm_server.py:
    from startup_telemetry import VLLMStartupMetrics

    metrics = VLLMStartupMetrics.init()
    metrics.record_python_init(duration_s)

    with metrics.vllm_init():
        engine = AsyncLLMEngine.from_engine_args(...)

    metrics.complete()
"""

import os
import time
from typing import Optional

# The library can be imported from model_engine_server.common.startup_tracing
# For now, we'll keep a simpler import path for the container
try:
    from model_engine_server.common.startup_tracing import StartupTracer
except ImportError:
    # Fallback for container environment where model_engine_server isn't installed
    # In production, we'd copy the library files or install as a package
    from startup_tracing import StartupTracer  # type: ignore


class VLLMStartupMetrics:
    """vLLM-specific startup metrics wrapper.

    Provides a clean API for the three vLLM startup phases:
    - s3_download: Model file download
    - python_init: Python/module initialization
    - vllm_init: vLLM engine initialization
    """

    # Metric names for Datadog
    METRIC_DOWNLOAD = "vllm.startup.download.duration_seconds"
    METRIC_PYTHON_INIT = "vllm.startup.python_init.duration_seconds"
    METRIC_VLLM_INIT = "vllm.startup.vllm_init.duration_seconds"
    METRIC_TOTAL = "vllm.startup.total.duration_seconds"

    def __init__(self, tracer: Optional[StartupTracer] = None):
        self._tracer = tracer
        self._python_init_start_ns: Optional[int] = None

    @classmethod
    def init(cls, container_start_time_ns: Optional[int] = None) -> "VLLMStartupMetrics":
        """Initialize vLLM startup metrics.

        Args:
            container_start_time_ns: Container start time (set by wrapper).
                If not provided, reads from CONTAINER_START_TS env var.

        Returns:
            VLLMStartupMetrics instance (may be a no-op if metrics disabled)
        """
        if os.environ.get("ENABLE_STARTUP_METRICS", "").lower() != "true":
            return cls(tracer=None)

        tracer = StartupTracer.from_env(
            service_name="vllm-startup",
            container_start_time_ns=container_start_time_ns,
        )
        return cls(tracer=tracer)

    @property
    def enabled(self) -> bool:
        """Whether metrics are enabled."""
        return self._tracer is not None

    @property
    def trace_id(self) -> Optional[str]:
        """Get trace ID for logging."""
        return self._tracer.trace_id if self._tracer else None

    # --- Download phase (called from wrapper) ---

    def record_download(
        self,
        start_time_ns: int,
        end_time_ns: int,
        size_mb: int = 0,
    ) -> None:
        """Record model download span and metric."""
        if not self._tracer:
            return

        duration_s = (end_time_ns - start_time_ns) / 1_000_000_000

        self._tracer.create_span(
            "s3_download",
            start_time_ns,
            end_time_ns,
            {"download_size_mb": size_mb, "duration_seconds": duration_s},
        )
        self._tracer.record_metric(self.METRIC_DOWNLOAD, duration_s)

    # --- Python init phase (called from server) ---

    def mark_python_init_start(self) -> None:
        """Mark the start of Python initialization (call at module load)."""
        self._python_init_start_ns = time.time_ns()

    def record_python_init(self, duration_s: Optional[float] = None) -> None:
        """Record Python initialization time.

        Args:
            duration_s: Duration in seconds. If not provided, calculates from
                mark_python_init_start() call.
        """
        if not self._tracer:
            return

        end_time_ns = time.time_ns()

        if duration_s is not None:
            start_time_ns = end_time_ns - int(duration_s * 1_000_000_000)
        elif self._python_init_start_ns:
            start_time_ns = self._python_init_start_ns
            duration_s = (end_time_ns - start_time_ns) / 1_000_000_000
        else:
            return

        self._tracer.create_span(
            "python_init",
            start_time_ns,
            end_time_ns,
            {"duration_seconds": duration_s},
        )
        self._tracer.record_metric(self.METRIC_PYTHON_INIT, duration_s)

    # --- vLLM init phase (called from server) ---

    def record_vllm_init(self, start_time_ns: int, end_time_ns: int) -> None:
        """Record vLLM initialization span with explicit timestamps.

        Args:
            start_time_ns: Start time in nanoseconds (before run_server)
            end_time_ns: End time in nanoseconds (when health check passes)
        """
        if not self._tracer:
            return

        duration_s = (end_time_ns - start_time_ns) / 1_000_000_000

        self._tracer.create_span(
            "vllm_init",
            start_time_ns,
            end_time_ns,
            {"duration_seconds": duration_s},
        )
        self._tracer.record_metric(self.METRIC_VLLM_INIT, duration_s)
        print(f"[STARTUP METRICS] vLLM init: {duration_s:.2f}s")

    # --- Completion ---

    def complete(self) -> float:
        """Mark startup complete. Returns total duration in seconds."""
        if not self._tracer:
            return 0.0

        total_duration = self._tracer.complete()
        self._tracer.record_metric(self.METRIC_TOTAL, total_duration)
        return total_duration

    def flush(self, timeout_ms: int = 5000) -> None:
        """Flush telemetry (call before exec)."""
        if self._tracer:
            self._tracer.flush(timeout_ms)


# --- Module-level convenience for wrapper script ---

_metrics: Optional[VLLMStartupMetrics] = None


def init_metrics(container_start_time_ns: Optional[int] = None) -> VLLMStartupMetrics:
    """Initialize global metrics instance."""
    global _metrics
    _metrics = VLLMStartupMetrics.init(container_start_time_ns)
    return _metrics


def get_metrics() -> Optional[VLLMStartupMetrics]:
    """Get global metrics instance."""
    return _metrics
