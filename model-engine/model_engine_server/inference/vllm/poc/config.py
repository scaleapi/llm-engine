"""
Configuration for vLLM startup metrics.

Uses Pydantic BaseSettings for environment variable configuration with defaults.
"""

import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class WatcherConfig:
    """Configuration for K8s event watcher pod filtering.

    All values can be set via environment variables:
    - WATCH_NAMESPACE: Kubernetes namespace to watch (default: "scale-deploy")
    - WATCH_LABEL_SELECTOR: Label selector for filtering pods (default: "")
    - WATCH_POD_NAME_PATTERN: Regex pattern for pod names (default: "")
    - WATCH_INCLUDE_LABELS: Comma-separated list of required labels (default: "")
    - WATCH_EXCLUDE_LABELS: Comma-separated list of labels to exclude (default: "")
    """

    namespace: str = field(
        default_factory=lambda: os.environ.get("WATCH_NAMESPACE", "scale-deploy")
    )
    label_selector: str = field(default_factory=lambda: os.environ.get("WATCH_LABEL_SELECTOR", ""))
    pod_name_pattern: str = field(
        default_factory=lambda: os.environ.get("WATCH_POD_NAME_PATTERN", "")
    )
    include_labels: list[str] = field(
        default_factory=lambda: _parse_list(os.environ.get("WATCH_INCLUDE_LABELS", ""))
    )
    exclude_labels: list[str] = field(
        default_factory=lambda: _parse_list(os.environ.get("WATCH_EXCLUDE_LABELS", ""))
    )

    def should_watch_pod(self, pod_name: str, labels: Optional[dict] = None) -> bool:
        """Determine if a pod should be watched based on configuration.

        Args:
            pod_name: Name of the pod
            labels: Pod labels dictionary

        Returns:
            True if pod matches filter criteria
        """
        import re

        labels = labels or {}

        # Check pod name pattern if specified
        if self.pod_name_pattern:
            if not re.match(self.pod_name_pattern, pod_name):
                return False

        # Check required labels
        for label in self.include_labels:
            if "=" in label:
                key, value = label.split("=", 1)
                if labels.get(key) != value:
                    return False
            else:
                if label not in labels:
                    return False

        # Check excluded labels
        for label in self.exclude_labels:
            if "=" in label:
                key, value = label.split("=", 1)
                if labels.get(key) == value:
                    return False
            else:
                if label in labels:
                    return False

        return True


@dataclass
class TelemetryConfig:
    """Configuration for OpenTelemetry export.

    All values can be set via environment variables:
    - OTEL_EXPORTER_OTLP_ENDPOINT: OTLP gRPC endpoint (default: "")
    - OTEL_SERVICE_NAME: Service name for traces/metrics (default: "vllm-startup")
    - DD_ENV: Deployment environment (default: "prod")
    """

    otlp_endpoint: str = field(
        default_factory=lambda: os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT", "")
    )
    service_name: str = field(
        default_factory=lambda: os.environ.get("OTEL_SERVICE_NAME", "vllm-startup")
    )
    environment: str = field(default_factory=lambda: os.environ.get("DD_ENV", "prod"))
    metric_export_interval_ms: int = field(
        default_factory=lambda: int(os.environ.get("OTEL_METRIC_EXPORT_INTERVAL_MS", "5000"))
    )

    @property
    def is_enabled(self) -> bool:
        """Check if telemetry is enabled (endpoint is configured)."""
        return bool(self.otlp_endpoint)


@dataclass
class StartupConfig:
    """Configuration for startup context metadata.

    These are typically set by the deployment/entrypoint:
    - ENDPOINT_NAME: Name of the vLLM endpoint
    - MODEL_NAME: Name of the model being served
    - GPU_TYPE: GPU hardware type
    - NUM_GPUS: Number of GPUs
    - AWS_REGION: AWS region
    - POD_UID: Kubernetes pod UID (from downward API)
    - POD_NAME: Kubernetes pod name
    - NODE_NAME: Kubernetes node name
    - CONTAINER_START_TS: Container start timestamp (set by entrypoint.sh)
    """

    endpoint_name: str = field(default_factory=lambda: os.environ.get("ENDPOINT_NAME", "unknown"))
    model_name: str = field(default_factory=lambda: os.environ.get("MODEL_NAME", "unknown"))
    gpu_type: str = field(default_factory=lambda: os.environ.get("GPU_TYPE", "unknown"))
    num_gpus: int = field(default_factory=lambda: int(os.environ.get("NUM_GPUS", "1")))
    region: str = field(
        default_factory=lambda: os.environ.get(
            "AWS_REGION", os.environ.get("AWS_DEFAULT_REGION", "unknown")
        )
    )
    pod_uid: str = field(default_factory=lambda: os.environ.get("POD_UID", "unknown"))
    pod_name: str = field(default_factory=lambda: os.environ.get("POD_NAME", "unknown"))
    node_name: str = field(default_factory=lambda: os.environ.get("NODE_NAME", "unknown"))

    # Timestamps from entrypoint.sh
    container_start_ts: Optional[float] = field(
        default_factory=lambda: _parse_float(os.environ.get("CONTAINER_START_TS"))
    )
    download_start_ts: Optional[float] = field(
        default_factory=lambda: _parse_float(os.environ.get("DOWNLOAD_START_TS"))
    )
    download_end_ts: Optional[float] = field(
        default_factory=lambda: _parse_float(os.environ.get("DOWNLOAD_END_TS"))
    )
    download_duration_s: Optional[float] = field(
        default_factory=lambda: _parse_float(os.environ.get("STARTUP_DOWNLOAD_DURATION_S"))
    )
    download_size_mb: Optional[int] = field(
        default_factory=lambda: _parse_int(os.environ.get("STARTUP_DOWNLOAD_SIZE_MB"))
    )

    @property
    def container_start_time_ns(self) -> Optional[int]:
        """Get container start time in nanoseconds for span timestamps."""
        if self.container_start_ts:
            return int(self.container_start_ts * 1_000_000_000)
        return None


def _parse_list(value: str) -> list[str]:
    """Parse comma-separated string into list."""
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def _parse_float(value: Optional[str]) -> Optional[float]:
    """Parse string to float, returning None on failure."""
    if not value:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _parse_int(value: Optional[str]) -> Optional[int]:
    """Parse string to int, returning None on failure."""
    if not value:
        return None
    try:
        return int(value)
    except ValueError:
        return None


# Singleton instances for convenience
_watcher_config: Optional[WatcherConfig] = None
_telemetry_config: Optional[TelemetryConfig] = None
_startup_config: Optional[StartupConfig] = None


def get_watcher_config() -> WatcherConfig:
    """Get or create singleton WatcherConfig instance."""
    global _watcher_config
    if _watcher_config is None:
        _watcher_config = WatcherConfig()
    return _watcher_config


def get_telemetry_config() -> TelemetryConfig:
    """Get or create singleton TelemetryConfig instance."""
    global _telemetry_config
    if _telemetry_config is None:
        _telemetry_config = TelemetryConfig()
    return _telemetry_config


def get_startup_config() -> StartupConfig:
    """Get or create singleton StartupConfig instance."""
    global _startup_config
    if _startup_config is None:
        _startup_config = StartupConfig()
    return _startup_config
