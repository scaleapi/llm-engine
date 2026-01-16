"""
K8s Startup Watcher - Generic Pod Startup Metrics.

Watches pod events and K8s events to capture granular startup phase timings:
- Scheduling: pod creation -> scheduled to node
- Image pull: pulling started -> pulling completed
- Container create: image pulled -> container created
- Container start: container created -> container started
- Total: pod creation -> container running

Emits metrics via OTLP to Datadog for correlation with in-container startup metrics.

Uses deterministic trace ID derived from pod_uid so spans appear in the same
trace as in-container startup spans (which use the same algorithm).
"""

import threading
import time
from datetime import datetime, timezone
from typing import Optional

from config import WatcherConfig, get_telemetry_config, get_watcher_config
from kubernetes import client, config, watch

# OTel imports
from opentelemetry import metrics, trace
from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.trace import SpanKind, Status, StatusCode

# Use the shared startup_tracing library for deterministic trace correlation
from startup_tracing import (
    SPAN_POD_STARTUP,
    DeterministicSpan,
    create_parent_context,
    format_trace_id,
)


class PodStartupTracker:
    """Tracks pod lifecycle events and emits granular startup metrics."""

    def __init__(self, watcher_config: Optional[WatcherConfig] = None):
        # pod_uid -> {timestamps and metadata}
        self._pods: dict = {}
        self._lock = threading.Lock()
        self._meter: Optional[metrics.Meter] = None
        self._tracer: Optional[trace.Tracer] = None
        self._span_processor: Optional[BatchSpanProcessor] = None
        self._resource: Optional[Resource] = None
        self._gauges: dict = {}
        self._histograms: dict = {}
        self._initialized = False
        self._watcher_config = watcher_config or get_watcher_config()
        self._v1: Optional[client.CoreV1Api] = None
        # Track startup time to avoid re-emitting metrics for old events on restart
        self._startup_time = datetime.now(timezone.utc)

    def set_k8s_client(self, v1: client.CoreV1Api):
        """Set the K8s API client for fetching pod metadata."""
        self._v1 = v1

    def init_telemetry(self):
        """Initialize OTel SDK."""
        telemetry_config = get_telemetry_config()

        if not telemetry_config.is_enabled:
            print("WARNING: OTEL_EXPORTER_OTLP_ENDPOINT not set, metrics disabled")
            return

        endpoint = telemetry_config.otlp_endpoint

        resource = Resource.create(
            {
                "service.name": "k8s-startup-watcher",
            }
        )

        # Metrics
        reader = PeriodicExportingMetricReader(
            OTLPMetricExporter(endpoint=endpoint),
            export_interval_millis=telemetry_config.metric_export_interval_ms,
        )
        meter_provider = MeterProvider(resource=resource, metric_readers=[reader])
        metrics.set_meter_provider(meter_provider)
        self._meter = metrics.get_meter("k8s-startup-watcher")

        # Define granular metrics (using generic names, not vllm-specific)
        metric_definitions = [
            ("scheduling", "Time from pod creation to scheduled on node"),
            ("image_pull", "Time to pull container image"),
            ("container_create", "Time to create container after image pulled"),
            ("container_start", "Time to start container after creation"),
            ("init_to_ready", "Time from container start to ready (init containers + setup)"),
            ("total_to_running", "Total time from pod creation to container running"),
        ]

        for name, description in metric_definitions:
            metric_name = f"k8s.startup.{name}.duration"
            self._gauges[name] = self._meter.create_gauge(
                metric_name,
                description=description,
                unit="s",
            )
            self._histograms[name] = self._meter.create_histogram(
                f"{metric_name}_seconds",
                description=description,
                unit="s",
            )

        # Traces
        self._resource = resource
        self._span_processor = BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint))
        trace_provider = TracerProvider(resource=resource)
        trace_provider.add_span_processor(self._span_processor)
        trace.set_tracer_provider(trace_provider)
        self._tracer = trace.get_tracer("k8s-startup-watcher")

        self._initialized = True
        print(f"Startup watcher telemetry initialized, endpoint={endpoint}")

    def should_watch_pod(self, pod_name: str, labels: Optional[dict] = None) -> bool:
        """Check if a pod should be watched based on configuration."""
        return self._watcher_config.should_watch_pod(pod_name, labels)

    def _fetch_pod_labels(self, pod_name: str) -> Optional[dict]:
        """Fetch pod labels from K8s API."""
        if not self._v1:
            return None
        try:
            pod = self._v1.read_namespaced_pod(pod_name, self._watcher_config.namespace)
            return pod.metadata.labels or {}
        except Exception as e:
            print(f"[WARN] Failed to fetch labels for {pod_name}: {e}")
            return None

    def _get_or_create_pod(
        self, pod_uid: str, pod_name: str = "", labels: Optional[dict] = None
    ) -> Optional[dict]:
        """Get or create pod tracking entry.

        Returns None if pod should not be watched based on configuration.
        """
        # If no labels provided, try to fetch them from K8s API
        if labels is None and pod_name:
            labels = self._fetch_pod_labels(pod_name)

        # Check if we should watch this pod
        if not self.should_watch_pod(pod_name, labels):
            return None

        with self._lock:
            if pod_uid not in self._pods:
                labels = labels or {}
                self._pods[pod_uid] = {
                    "name": pod_name,
                    "endpoint_name": labels.get("endpoint_name", labels.get("app", "unknown")),
                    "model_name": labels.get("model_name", "unknown"),
                    "timestamps": {},
                    "emitted": set(),  # Track which metrics we've already emitted
                }
            # Update labels if we now have them and they were missing before
            elif labels and self._pods[pod_uid].get("endpoint_name") == "unknown":
                self._pods[pod_uid]["endpoint_name"] = labels.get(
                    "endpoint_name", labels.get("app", "unknown")
                )
                self._pods[pod_uid]["model_name"] = labels.get(
                    "model_name", self._pods[pod_uid]["model_name"]
                )
            return self._pods[pod_uid]

    def on_pod_event(self, event_type: str, pod: client.V1Pod):
        """Handle pod watch event."""
        pod_name = pod.metadata.name
        pod_uid = pod.metadata.uid
        labels = pod.metadata.labels or {}

        pod_data = self._get_or_create_pod(pod_uid, pod_name, labels)
        if pod_data is None:
            return  # Pod doesn't match filter criteria
        timestamps = pod_data["timestamps"]

        # Record creation time
        if "created" not in timestamps and pod.metadata.creation_timestamp:
            timestamps["created"] = pod.metadata.creation_timestamp
            print(f"[POD] {pod_name} created at {timestamps['created']}")
            # Try to emit scheduling metric - handles race where K8s Event arrives first
            self._try_emit_metric(pod_uid, "scheduling", "created", "scheduled")

        # Check pod conditions for PodScheduled
        if pod.status and pod.status.conditions:
            for condition in pod.status.conditions:
                if condition.type == "PodScheduled" and condition.status == "True":
                    if "scheduled" not in timestamps and condition.last_transition_time:
                        timestamps["scheduled"] = condition.last_transition_time
                        print(f"[POD] {pod_name} scheduled at {timestamps['scheduled']}")
                        self._try_emit_metric(pod_uid, "scheduling", "created", "scheduled")

                elif condition.type == "Initialized" and condition.status == "True":
                    if "initialized" not in timestamps and condition.last_transition_time:
                        timestamps["initialized"] = condition.last_transition_time

                elif condition.type == "ContainersReady" and condition.status == "True":
                    if "containers_ready" not in timestamps and condition.last_transition_time:
                        timestamps["containers_ready"] = condition.last_transition_time

                elif condition.type == "Ready" and condition.status == "True":
                    if "ready" not in timestamps and condition.last_transition_time:
                        timestamps["ready"] = condition.last_transition_time
                        print(f"[POD] {pod_name} ready at {timestamps['ready']}")

        # Check container statuses for running time
        if pod.status and pod.status.container_statuses:
            for container_status in pod.status.container_statuses:
                if container_status.name == "main" and container_status.state:
                    if container_status.state.running:
                        running_since = container_status.state.running.started_at
                        if "container_running" not in timestamps and running_since:
                            timestamps["container_running"] = running_since
                            print(
                                f"[POD] {pod_name} container running at {timestamps['container_running']}"
                            )
                            self._try_emit_metric(
                                pod_uid, "total_to_running", "created", "container_running"
                            )
                            self._try_emit_metric(
                                pod_uid, "init_to_ready", "container_started", "container_running"
                            )

    def on_k8s_event(self, event: client.CoreV1Event):
        """Handle K8s Event object (Pulling, Pulled, Created, Started, etc.)."""
        involved_object = event.involved_object
        if involved_object.kind != "Pod":
            return

        pod_uid = involved_object.uid
        pod_name = involved_object.name
        reason = event.reason
        event_time = event.last_timestamp or event.first_timestamp or event.event_time

        if not event_time:
            return

        # Only track events for pods we know about or new pods
        pod_data = self._get_or_create_pod(pod_uid, pod_name)
        if pod_data is None:
            return  # Pod doesn't match filter criteria
        timestamps = pod_data["timestamps"]

        # Map event reasons to timestamp keys
        event_mapping = {
            "Scheduled": "scheduled",
            "Pulling": "image_pull_started",
            "Pulled": "image_pulled",
            "Created": "container_created",
            "Started": "container_started",
        }

        if reason in event_mapping:
            ts_key = event_mapping[reason]
            if ts_key not in timestamps:
                timestamps[ts_key] = event_time
                print(f"[EVENT] {pod_name} {reason} at {event_time}")

                # Emit metrics based on available timestamps
                if reason == "Scheduled":
                    self._try_emit_metric(pod_uid, "scheduling", "created", "scheduled")

                elif reason == "Pulled":
                    self._try_emit_metric(
                        pod_uid, "image_pull", "image_pull_started", "image_pulled"
                    )

                elif reason == "Created":
                    self._try_emit_metric(
                        pod_uid, "container_create", "image_pulled", "container_created"
                    )

                elif reason == "Started":
                    self._try_emit_metric(
                        pod_uid, "container_start", "container_created", "container_started"
                    )

    def _try_emit_metric(self, pod_uid: str, metric_name: str, start_key: str, end_key: str):
        """Try to emit a metric if both timestamps are available."""
        if not self._initialized:
            return

        with self._lock:
            if pod_uid not in self._pods:
                return

            pod_data = self._pods[pod_uid]
            timestamps = pod_data["timestamps"]
            emitted = pod_data["emitted"]

            # Check if already emitted
            if metric_name in emitted:
                return

            # Check if both timestamps available
            start_time = timestamps.get(start_key)
            end_time = timestamps.get(end_key)

            if not start_time or not end_time:
                return

            # Skip pods created before this watcher started to prevent duplicates on restart
            # A previous watcher instance would have already emitted metrics for these pods
            if start_time < self._startup_time:
                print(
                    f"[SKIP] {pod_data['name']} {metric_name} - pod created before watcher startup (ignoring historical pod)"
                )
                emitted.add(metric_name)  # Mark as emitted to avoid repeated logs
                return

            # Calculate duration
            duration = (end_time - start_time).total_seconds()

            # Skip negative durations (clock skew)
            if duration < 0:
                print(
                    f"[WARN] {pod_data['name']} {metric_name} negative duration: {duration:.2f}s (skipped)"
                )
                return

            # Mark as emitted
            emitted.add(metric_name)

        # Emit outside the lock
        attrs = {
            "endpoint_name": pod_data.get("endpoint_name", "unknown"),
            "model_name": pod_data.get("model_name", "unknown"),
            "pod_name": pod_data.get("name", "unknown"),
        }

        # Record gauge and histogram
        self._gauges[metric_name].set(duration, attrs)
        self._histograms[metric_name].record(duration, attrs)

        print(
            f"[METRIC] {pod_data['name']} {metric_name}={duration:.2f}s endpoint_name={attrs['endpoint_name']}"
        )

        # Create span with actual timestamps, using deterministic trace context
        if self._tracer and self._span_processor:
            start_ns = int(start_time.timestamp() * 1_000_000_000)
            end_ns = int(end_time.timestamp() * 1_000_000_000)

            span_name = f"k8s_{metric_name}"
            span_attrs = {
                "pod_name": pod_data["name"],
                "pod_uid": pod_uid,
                f"{metric_name}_seconds": duration,
                **attrs,
            }

            if metric_name == "total_to_running":
                # Parent span - use DeterministicSpan with deterministic span_id
                # This is the root span for K8s-side startup tracking
                span = DeterministicSpan(
                    name=span_name,
                    unique_id=pod_uid,
                    span_suffix=SPAN_POD_STARTUP,  # "pod_startup" - root for K8s events
                    parent_suffix="root",  # Parent is the synthetic root span
                    start_time_ns=start_ns,
                    end_time_ns=end_ns,
                    attributes=span_attrs,
                    resource=self._resource,
                )
                # Export directly via span processor
                self._span_processor.on_end(span)
            else:
                # Child spans - use create_parent_context which references
                # pod_startup's deterministic span_id as parent
                parent_ctx = create_parent_context(pod_uid, SPAN_POD_STARTUP)

                span = self._tracer.start_span(
                    span_name,
                    kind=SpanKind.INTERNAL,
                    context=parent_ctx,
                    start_time=start_ns,
                )

                for k, v in span_attrs.items():
                    span.set_attribute(k, v)
                span.set_status(Status(StatusCode.OK))
                span.end(end_time=end_ns)

            print(f"[SPAN] {pod_data['name']} {span_name} trace_id={format_trace_id(pod_uid)}")

    def cleanup_old_pods(self):
        """Remove old pod entries to prevent memory growth."""
        now = datetime.now(timezone.utc)
        with self._lock:
            to_remove = []
            for pod_uid, data in self._pods.items():
                created = data["timestamps"].get("created")
                if created and (now - created).total_seconds() > 3600:  # 1 hour
                    to_remove.append(pod_uid)
            for uid in to_remove:
                del self._pods[uid]
            if to_remove:
                print(f"[CLEANUP] Removed {len(to_remove)} old pod entries")


def watch_pods(v1: client.CoreV1Api, tracker: PodStartupTracker, watcher_config: WatcherConfig):
    """Watch pod events in a thread.

    Watches all pods in namespace and filters in Python via should_watch_pod().
    This is intentional - label_selector filtering at API server level is inefficient
    as it still processes all objects server-side.

    Uses resource_version to resume watches efficiently and avoid re-listing all pods.
    """
    w = watch.Watch()
    resource_version = None

    while True:
        try:
            print(f"[THREAD] Watching pods in namespace {watcher_config.namespace}...")

            # Watch all pods in namespace, filter in Python
            # Note: We intentionally don't use label_selector here because K8s API server
            # label filtering is inefficient (still processes all objects server-side)
            watch_kwargs = {
                "namespace": watcher_config.namespace,
                "timeout_seconds": 300,
            }
            if resource_version:
                watch_kwargs["resource_version"] = resource_version

            for event in w.stream(v1.list_namespaced_pod, **watch_kwargs):
                event_type = event["type"]
                pod = event["object"]

                # Update resource version for resume on reconnect
                if pod.metadata.resource_version:
                    resource_version = pod.metadata.resource_version

                tracker.on_pod_event(event_type, pod)

            tracker.cleanup_old_pods()

        except client.exceptions.ApiException as e:
            if e.status == 410:  # Gone - resource version too old
                print("[THREAD] Resource version expired, resetting watch")
                resource_version = None
            else:
                print(f"[THREAD] K8s API error (pods): {e.reason}")
            time.sleep(5)
        except Exception as e:
            print(f"[THREAD] Error watching pods: {e}")
            time.sleep(5)


def watch_events(v1: client.CoreV1Api, tracker: PodStartupTracker, watcher_config: WatcherConfig):
    """Watch K8s events in a thread.

    Uses field_selector to only watch Pod events, reducing API server load.
    Uses resource_version to resume watches efficiently.
    """
    w = watch.Watch()
    resource_version = None

    while True:
        try:
            print(f"[THREAD] Watching events in namespace {watcher_config.namespace}...")

            # Build watch kwargs - filter to Pod events only
            watch_kwargs = {
                "namespace": watcher_config.namespace,
                "timeout_seconds": 300,
                "field_selector": "involvedObject.kind=Pod",
            }
            if resource_version:
                watch_kwargs["resource_version"] = resource_version

            for event in w.stream(v1.list_namespaced_event, **watch_kwargs):
                k8s_event = event["object"]

                # Update resource version for resume on reconnect
                if k8s_event.metadata.resource_version:
                    resource_version = k8s_event.metadata.resource_version

                tracker.on_k8s_event(k8s_event)

        except client.exceptions.ApiException as e:
            if e.status == 410:  # Gone - resource version too old
                print("[THREAD] Resource version expired, resetting watch")
                resource_version = None
            else:
                print(f"[THREAD] K8s API error (events): {e.reason}")
            time.sleep(5)
        except Exception as e:
            print(f"[THREAD] Error watching events: {e}")
            time.sleep(5)


def main():
    """Main entry point."""
    # Load configuration from environment
    watcher_config = get_watcher_config()
    telemetry_config = get_telemetry_config()

    startup_time = datetime.now(timezone.utc)
    print("Starting K8s Startup Watcher")
    print(f"  Startup time: {startup_time.isoformat()}")
    print("  Historical events: will be skipped (prevents duplicates on restart)")
    print(f"  Namespace: {watcher_config.namespace}")
    print(f"  OTLP endpoint: {telemetry_config.otlp_endpoint}")
    print()
    print("Python-side pod filtering (applied after receiving from API):")
    print(f"  Pod name pattern: {watcher_config.pod_name_pattern or '(any)'}")
    print(f"  Include labels: {watcher_config.include_labels or '(none)'}")
    print(f"  Exclude labels: {watcher_config.exclude_labels or '(none)'}")
    print()
    print("Metrics emitted:")
    print("  - k8s.startup.scheduling.duration      (created -> scheduled)")
    print("  - k8s.startup.image_pull.duration      (pulling -> pulled)")
    print("  - k8s.startup.container_create.duration (pulled -> created)")
    print("  - k8s.startup.container_start.duration  (created -> started)")
    print("  - k8s.startup.total_to_running.duration (created -> running)")
    print()

    # Load K8s config
    try:
        config.load_incluster_config()
        print("Using in-cluster K8s config")
    except config.ConfigException:
        config.load_kube_config()
        print("Using local K8s config")

    v1 = client.CoreV1Api()
    tracker = PodStartupTracker(watcher_config)
    tracker.set_k8s_client(v1)
    tracker.init_telemetry()

    # Start pod watcher thread
    pod_thread = threading.Thread(
        target=watch_pods, args=(v1, tracker, watcher_config), daemon=True
    )
    pod_thread.start()

    # Start event watcher thread
    event_thread = threading.Thread(
        target=watch_events, args=(v1, tracker, watcher_config), daemon=True
    )
    event_thread.start()

    # Keep main thread alive
    while True:
        time.sleep(60)
        tracker.cleanup_old_pods()


if __name__ == "__main__":
    main()
