"""
K8s Event Watcher for vLLM Startup Metrics.

Watches pod events and K8s events to capture granular startup phase timings:
- Scheduling: pod creation → scheduled to node
- Image pull: pulling started → pulling completed
- Container create: image pulled → container created
- Container start: container created → container started
- Total: pod creation → container running

Emits metrics via OTLP to Datadog for correlation with in-container startup metrics.
"""

import os
import threading
import time
from datetime import datetime, timezone
from typing import Optional

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

# Configuration
NAMESPACE = os.environ.get("WATCH_NAMESPACE", "scale-deploy")
LABEL_SELECTOR = os.environ.get("LABEL_SELECTOR", "")  # e.g., "team=ml-infra"
OTLP_ENDPOINT = os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT", "")


class PodStartupTracker:
    """Tracks pod lifecycle events and emits granular startup metrics."""

    def __init__(self):
        # pod_uid -> {timestamps and metadata}
        self._pods: dict = {}
        self._lock = threading.Lock()
        self._meter: Optional[metrics.Meter] = None
        self._tracer: Optional[trace.Tracer] = None
        self._gauges: dict = {}
        self._histograms: dict = {}
        self._initialized = False

    def init_telemetry(self):
        """Initialize OTel SDK."""
        if not OTLP_ENDPOINT:
            print("WARNING: OTEL_EXPORTER_OTLP_ENDPOINT not set, metrics disabled")
            return

        resource = Resource.create(
            {
                "service.name": "vllm-event-watcher",
            }
        )

        # Metrics
        reader = PeriodicExportingMetricReader(
            OTLPMetricExporter(endpoint=OTLP_ENDPOINT),
            export_interval_millis=10000,
        )
        meter_provider = MeterProvider(resource=resource, metric_readers=[reader])
        metrics.set_meter_provider(meter_provider)
        self._meter = metrics.get_meter("vllm-event-watcher")

        # Define granular metrics
        metric_definitions = [
            ("scheduling", "Time from pod creation to scheduled on node"),
            ("image_pull", "Time to pull container image"),
            ("container_create", "Time to create container after image pulled"),
            ("container_start", "Time to start container after creation"),
            ("init_to_ready", "Time from container start to ready (init containers + setup)"),
            ("total_to_running", "Total time from pod creation to container running"),
        ]

        for name, description in metric_definitions:
            metric_name = f"vllm.startup.{name}.duration"
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
        trace_provider = TracerProvider(resource=resource)
        trace_provider.add_span_processor(
            BatchSpanProcessor(OTLPSpanExporter(endpoint=OTLP_ENDPOINT))
        )
        trace.set_tracer_provider(trace_provider)
        self._tracer = trace.get_tracer("vllm-event-watcher")

        self._initialized = True
        print(f"Event watcher telemetry initialized, endpoint={OTLP_ENDPOINT}")

    def _get_or_create_pod(
        self, pod_uid: str, pod_name: str = "", labels: Optional[dict] = None
    ) -> dict:
        """Get or create pod tracking entry."""
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
            return self._pods[pod_uid]

    def on_pod_event(self, event_type: str, pod: client.V1Pod):
        """Handle pod watch event."""
        pod_name = pod.metadata.name
        pod_uid = pod.metadata.uid
        labels = pod.metadata.labels or {}

        pod_data = self._get_or_create_pod(pod_uid, pod_name, labels)
        timestamps = pod_data["timestamps"]

        # Record creation time
        if "created" not in timestamps and pod.metadata.creation_timestamp:
            timestamps["created"] = pod.metadata.creation_timestamp
            print(f"[POD] {pod_name} created at {timestamps['created']}")

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
        }

        # Record gauge and histogram
        self._gauges[metric_name].set(duration, attrs)
        self._histograms[metric_name].record(duration, attrs)

        print(f"[METRIC] {pod_data['name']} {metric_name}={duration:.2f}s")

        # Create span with actual timestamps
        if self._tracer:
            start_ns = int(start_time.timestamp() * 1_000_000_000)
            end_ns = int(end_time.timestamp() * 1_000_000_000)

            span = self._tracer.start_span(
                f"pod_{metric_name}",
                kind=SpanKind.INTERNAL,
                start_time=start_ns,
            )
            span.set_attribute("pod_name", pod_data["name"])
            span.set_attribute("pod_uid", pod_uid)
            span.set_attribute(f"{metric_name}_seconds", duration)
            for k, v in attrs.items():
                span.set_attribute(k, v)
            span.set_status(Status(StatusCode.OK))
            span.end(end_time=end_ns)

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


def watch_pods(v1: client.CoreV1Api, tracker: PodStartupTracker):
    """Watch pod events in a thread."""
    w = watch.Watch()
    while True:
        try:
            print(f"[THREAD] Watching pods in namespace {NAMESPACE}...")
            for event in w.stream(
                v1.list_namespaced_pod,
                namespace=NAMESPACE,
                label_selector=LABEL_SELECTOR,
                timeout_seconds=300,
            ):
                event_type = event["type"]
                pod = event["object"]
                tracker.on_pod_event(event_type, pod)

            tracker.cleanup_old_pods()

        except client.exceptions.ApiException as e:
            print(f"[THREAD] K8s API error (pods): {e.reason}")
            time.sleep(5)
        except Exception as e:
            print(f"[THREAD] Error watching pods: {e}")
            time.sleep(5)


def watch_events(v1: client.CoreV1Api, tracker: PodStartupTracker):
    """Watch K8s events in a thread."""
    w = watch.Watch()
    while True:
        try:
            print(f"[THREAD] Watching events in namespace {NAMESPACE}...")
            for event in w.stream(
                v1.list_namespaced_event,
                namespace=NAMESPACE,
                timeout_seconds=300,
            ):
                k8s_event = event["object"]
                tracker.on_k8s_event(k8s_event)

        except client.exceptions.ApiException as e:
            print(f"[THREAD] K8s API error (events): {e.reason}")
            time.sleep(5)
        except Exception as e:
            print(f"[THREAD] Error watching events: {e}")
            time.sleep(5)


def main():
    """Main entry point."""
    print("Starting K8s Event Watcher (Granular)")
    print(f"  Namespace: {NAMESPACE}")
    print(f"  Label selector: {LABEL_SELECTOR or '(all pods)'}")
    print(f"  OTLP endpoint: {OTLP_ENDPOINT}")
    print()
    print("Metrics emitted:")
    print("  - vllm.startup.scheduling.duration      (created → scheduled)")
    print("  - vllm.startup.image_pull.duration      (pulling → pulled)")
    print("  - vllm.startup.container_create.duration (pulled → created)")
    print("  - vllm.startup.container_start.duration  (created → started)")
    print("  - vllm.startup.total_to_running.duration (created → running)")
    print()

    # Load K8s config
    try:
        config.load_incluster_config()
        print("Using in-cluster K8s config")
    except config.ConfigException:
        config.load_kube_config()
        print("Using local K8s config")

    v1 = client.CoreV1Api()
    tracker = PodStartupTracker()
    tracker.init_telemetry()

    # Start pod watcher thread
    pod_thread = threading.Thread(target=watch_pods, args=(v1, tracker), daemon=True)
    pod_thread.start()

    # Start event watcher thread
    event_thread = threading.Thread(target=watch_events, args=(v1, tracker), daemon=True)
    event_thread.start()

    # Keep main thread alive
    while True:
        time.sleep(60)
        tracker.cleanup_old_pods()


if __name__ == "__main__":
    main()
