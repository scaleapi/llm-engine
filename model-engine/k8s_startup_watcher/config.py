"""
Configuration for K8s Startup Watcher.

Environment variables for configuration:
- WATCH_NAMESPACE: Kubernetes namespace to watch (default: "scale-deploy")
- WATCH_POD_NAME_PATTERN: Regex pattern for pod names (default: "")
- WATCH_INCLUDE_LABELS: Comma-separated list of required labels (default: "")
- WATCH_EXCLUDE_LABELS: Comma-separated list of labels to exclude (default: "")
- OTEL_EXPORTER_OTLP_ENDPOINT: OTLP gRPC endpoint for telemetry export
- OTEL_SERVICE_NAME: Service name for traces/metrics (default: "k8s-startup-watcher")
- OTEL_METRIC_EXPORT_INTERVAL_MS: Metric export interval (default: 5000)
"""

import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class WatcherConfig:
    """Configuration for K8s pod/event watching."""

    namespace: str = field(
        default_factory=lambda: os.environ.get("WATCH_NAMESPACE", "scale-deploy")
    )
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
    """Configuration for OpenTelemetry export."""

    otlp_endpoint: str = field(
        default_factory=lambda: os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT", "")
    )
    service_name: str = field(
        default_factory=lambda: os.environ.get("OTEL_SERVICE_NAME", "k8s-startup-watcher")
    )
    metric_export_interval_ms: int = field(
        default_factory=lambda: int(os.environ.get("OTEL_METRIC_EXPORT_INTERVAL_MS", "5000"))
    )

    @property
    def is_enabled(self) -> bool:
        """Check if telemetry is enabled (endpoint is configured)."""
        return bool(self.otlp_endpoint)


def _parse_list(value: str) -> list[str]:
    """Parse comma-separated string into list."""
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


# Singleton instances
_watcher_config: Optional[WatcherConfig] = None
_telemetry_config: Optional[TelemetryConfig] = None


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
