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
import threading
import time
from argparse import Namespace
from typing import Awaitable, Callable, Optional, TypeVar

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


def _health_check_loop(metrics, vllm_init_start_ns: int, host: str = "localhost", port: int = 5005):
    """
    Background thread that polls /health endpoint to detect when server is ready.
    Once ready, records the vllm_init span and startup complete metric.
    """
    import urllib.error
    import urllib.request

    url = f"http://{host}:{port}/health"
    max_attempts = 1200  # 20 minutes max
    attempt = 0

    while attempt < max_attempts:
        try:
            with urllib.request.urlopen(url, timeout=1) as response:
                if response.status == 200:
                    # Server is ready - record vllm_init span
                    end_time_ns = time.time_ns()
                    metrics.record_vllm_init(vllm_init_start_ns, end_time_ns)

                    # Record startup complete (emits in_container_startup span)
                    total_duration = metrics.complete()
                    print(f"[STARTUP METRICS] Server ready after {total_duration:.2f}s total")
                    return
        except (urllib.error.URLError, urllib.error.HTTPError, ConnectionRefusedError, OSError):
            pass

        attempt += 1
        time.sleep(1)

    print("[STARTUP METRICS] WARNING: Health check timed out, server may not be ready")


T = TypeVar("T")


async def with_startup_metrics(
    func: Callable[..., Awaitable[T]], args: Namespace, python_start_time: float
) -> T:
    """
    Wrapper to run a function with startup metrics if enabled

    Emits spans for:
    - python_init: From module start to vllm_init start
    - vllm_init: From vllm_init start to server ready (health check passes)

    Download span is emitted by vllm_startup_wrapper.py before exec'ing into this.

    Args:
        func: The async function to wrap (must return an awaitable).

    Returns:
        The result of the function.
    """

    # Startup metrics feature gate (check early to avoid unnecessary imports)
    ENABLE_STARTUP_METRICS = os.environ.get("ENABLE_STARTUP_METRICS", "").lower() == "true"
    if ENABLE_STARTUP_METRICS:
        from .startup_telemetry import VLLMStartupMetrics

        # Initialize metrics (reads CONTAINER_START_TS from env)
        metrics = VLLMStartupMetrics.init()

        if metrics.enabled:
            print(f"[STARTUP METRICS] trace_id={metrics.trace_id}")

        # Record Python init time (from module start to now)
        python_init_duration = time.perf_counter() - python_start_time
        metrics.record_python_init(python_init_duration)
        print(f"[STARTUP METRICS] Python init: {python_init_duration:.2f}s")

        # Mark vllm_init start time
        vllm_init_start_ns = time.time_ns()

        # Start health check thread to detect when server is ready
        # This will record vllm_init span and call metrics.complete() when /health returns 200
        port = getattr(args, "port", 5005)
        health_thread = threading.Thread(
            target=_health_check_loop,
            args=(metrics, vllm_init_start_ns, "localhost", port),
            daemon=True,
        )
        health_thread.start()

    return await func(*args)
