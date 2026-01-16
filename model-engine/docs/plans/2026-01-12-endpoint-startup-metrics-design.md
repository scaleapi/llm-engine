# Endpoint Startup Metrics Design

## Overview

Instrument model endpoint startup to capture detailed timing metrics for optimization analysis. Uses OpenTelemetry for vendor-neutral instrumentation with Datadog as the visualization layer.

## Goals

- Understand pod scaling time for GPU endpoints
- Measure: pod initialization, model download (s5cmd), model loading, endpoint ready
- Enable aggregate analysis for optimization (p50/p95 by model, GPU type, region)
- Establish baseline metrics before implementing improvements (e.g., preloading model weights)

## Architecture

### Two-Component System

```
┌─────────────────────────────────────────────────────────────────┐
│                    Kubernetes Cluster                           │
│                                                                 │
│  ┌──────────────────────┐      ┌──────────────────────────────┐│
│  │ K8s Event Watcher    │      │ vLLM Pod                     ││
│  │ (Deployment)         │      │                              ││
│  │                      │      │  ┌────────────────────────┐  ││
│  │ • Watches pod events │      │  │ Startup Instrumenter   │  ││
│  │ • Emits spans:       │      │  │                        │  ││
│  │   - pod_scheduled    │      │  │ • Emits spans:         │  ││
│  │   - image_pulled     │      │  │   - s5cmd_download     │  ││
│  │   - container_started│      │  │   - ray_cluster_init   │  ││
│  │                      │      │  │   - vllm_init          │  ││
│  │ Correlates via       │◄─────┼──│   - server_ready       │  ││
│  │ pod_uid              │      │  │                        │  ││
│  │                      │      │  └───────────┬────────────┘  ││
│  └──────────┬───────────┘      │              │               ││
│             │                  │              │               ││
└─────────────┼──────────────────┼──────────────┼───────────────┘
              │                  │              │
              ▼                  │              ▼
       ┌──────────────────────────────────────────┐
       │         OTel Collector (DaemonSet)       │
       │                                          │
       │  Joins spans by trace_id (derived from   │
       │  pod_uid) into complete startup trace    │
       │                                          │
       │  Exports to Datadog                      │
       └──────────────────────────────────────────┘
```

### Correlation Strategy

- Trace ID = deterministic MD5 hash of `pod_uid`
- Both components emit spans with same trace ID
- Datadog joins them into a single trace view

### Why Two Components

- K8s events (scheduling, image pull) happen before container code runs
- Container instrumentation captures phases we control
- Single trace gives end-to-end visibility

## Trace Structure

```
trace_id: derived from pod_uid (deterministic hash)

K8s Event Watcher spans:
├── pod_pending          [pod created → scheduled]
├── image_pull           [pulling → pulled]
└── container_creating   [creating → started]

In-container spans:
└── startup (parent)
    ├── s5cmd_download   [download start → complete]
    ├── ray_cluster_init [ray start → cluster ready] (batch only)
    ├── vllm_init        [from_vllm_config() call duration]
    └── server_ready     [server listening]
```

## Attributes (Low Cardinality)

All spans include:

| Attribute | Type | Example | Cardinality |
|-----------|------|---------|-------------|
| `endpoint_name` | string | `llama-70b-prod` | Low |
| `model_name` | string | `llama-70b` | Low |
| `gpu_type` | string | `h100`, `a100` | Low |
| `node_name` | string | `gpu-node-pool-abc123` | Medium |
| `namespace` | string | `model-endpoints` | Low |
| `num_gpus` | int | `8` | Low |
| `image_tag` | string | `v1.2.3` | Low |
| `region` | string | `us-east-1` | Low |

## Metrics (Histograms)

For aggregate analysis and optimization:

```
# Download phase
startup.download.duration_seconds    {endpoint_name, model_name, region}
startup.download.throughput_mbps     {endpoint_name, model_name, region}

# vLLM init phase
startup.vllm_init.duration_seconds   {endpoint_name, model_name, gpu_type, num_gpus}

# Total startup
startup.total.duration_seconds       {endpoint_name, model_name, gpu_type}

# K8s phases
startup.pod_pending.duration_seconds {endpoint_name, namespace}
startup.image_pull.duration_seconds  {endpoint_name, image_tag}
```

## Implementation Components

### 1. Helm Chart Changes

Add environment variables for telemetry context:

```yaml
# Add to _helpers.tpl baseServiceTemplateEnv
- name: POD_UID
  valueFrom:
    fieldRef:
      fieldPath: metadata.uid
- name: POD_NAME
  valueFrom:
    fieldRef:
      fieldPath: metadata.name
- name: NODE_NAME
  valueFrom:
    fieldRef:
      fieldPath: spec.nodeName
- name: GPU_TYPE
  value: "${GPU_TYPE}"
- name: AWS_REGION
  value: "${AWS_REGION}"
- name: OTEL_EXPORTER_OTLP_ENDPOINT
  value: "${OTEL_COLLECTOR_ENDPOINT}"
```

Enable vLLM native request tracing:

```yaml
# vLLM server args
args:
  - "--otlp-traces-endpoint=$(OTEL_EXPORTER_OTLP_ENDPOINT)"
```

### 2. Startup Telemetry Module (New File)

Location: `model_engine_server/inference/vllm/startup_telemetry.py`

```python
import hashlib
import os
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Generator

from opentelemetry import trace, metrics
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.trace import SpanKind, Status, StatusCode

_tracer: trace.Tracer | None = None
_meter: metrics.Meter | None = None
_histograms: dict = {}
_context: "StartupContext | None" = None


@dataclass
class StartupContext:
    """Runtime context for startup telemetry."""
    endpoint_name: str
    model_name: str
    gpu_type: str
    num_gpus: int
    region: str


def init_startup_telemetry(ctx: StartupContext) -> None:
    """Initialize OTel SDK for startup instrumentation."""
    global _tracer, _meter, _histograms, _context

    endpoint = os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT")
    if not endpoint:
        return  # Telemetry disabled

    _context = ctx

    resource = Resource.create({
        "service.name": "vllm-startup",
        "k8s.pod.uid": os.environ.get("POD_UID", "unknown"),
        "k8s.pod.name": os.environ.get("POD_NAME", "unknown"),
        "k8s.node.name": os.environ.get("NODE_NAME", "unknown"),
        "endpoint_name": ctx.endpoint_name,
        "model_name": ctx.model_name,
    })

    # Traces
    provider = TracerProvider(resource=resource)
    provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint)))
    trace.set_tracer_provider(provider)
    _tracer = trace.get_tracer("vllm-startup")

    # Metrics
    reader = PeriodicExportingMetricReader(OTLPMetricExporter(endpoint=endpoint))
    meter_provider = MeterProvider(resource=resource, metric_readers=[reader])
    metrics.set_meter_provider(meter_provider)
    _meter = metrics.get_meter("vllm-startup")

    # Create histograms
    _histograms["download_duration"] = _meter.create_histogram(
        "startup.download.duration_seconds",
        description="Model download duration",
    )
    _histograms["vllm_init_duration"] = _meter.create_histogram(
        "startup.vllm_init.duration_seconds",
        description="vLLM initialization duration",
    )
    _histograms["total_duration"] = _meter.create_histogram(
        "startup.total.duration_seconds",
        description="Total startup duration",
    )


def derive_trace_id(pod_uid: str) -> str:
    """Deterministic trace ID from pod UID for correlation."""
    return hashlib.md5(pod_uid.encode()).hexdigest()[:32]


@contextmanager
def startup_span(name: str, attributes: dict | None = None) -> Generator:
    """Context manager for startup phase spans."""
    if not _tracer or not _context:
        yield None
        return

    attrs = {
        "endpoint_name": _context.endpoint_name,
        "model_name": _context.model_name,
        "gpu_type": _context.gpu_type,
        "num_gpus": _context.num_gpus,
        "region": _context.region,
        **(attributes or {}),
    }

    with _tracer.start_as_current_span(name, kind=SpanKind.INTERNAL) as span:
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


def record_metric(name: str, value: float, extra_attrs: dict | None = None) -> None:
    """Record a histogram metric."""
    if not _context or name not in _histograms:
        return

    attrs = {
        "endpoint_name": _context.endpoint_name,
        "model_name": _context.model_name,
        "gpu_type": _context.gpu_type,
        **(extra_attrs or {}),
    }
    _histograms[name].record(value, attrs)
```

### 3. Integration with vllm_batch.py

```python
# Add imports
from model_engine_server.inference.vllm.startup_telemetry import (
    StartupContext,
    init_startup_telemetry,
    startup_span,
    record_metric,
)

# In handle_batch_job()
async def handle_batch_job(request: CreateBatchCompletionsEngineRequest):
    # Initialize telemetry with runtime context
    ctx = StartupContext(
        endpoint_name=os.environ.get("ENDPOINT_NAME", "unknown"),
        model_name=request.model_cfg.model,
        gpu_type=os.environ.get("GPU_TYPE", "unknown"),
        num_gpus=request.model_cfg.num_gpus or 1,
        region=os.environ.get("AWS_REGION", "unknown"),
    )
    init_startup_telemetry(ctx)

    total_start = time.perf_counter()

    with startup_span("startup") as parent_span:
        # Download phase
        with startup_span("s5cmd_download"):
            download_start = time.perf_counter()
            await download_model()
            record_metric("download_duration", time.perf_counter() - download_start)

        # Ray init (multi-node only)
        with startup_span("ray_cluster_init"):
            init_ray()

        # vLLM init
        with startup_span("vllm_init"):
            init_start = time.perf_counter()
            engine = await init_engine(...)
            record_metric("vllm_init_duration", time.perf_counter() - init_start)

        # Server ready
        with startup_span("server_ready"):
            total_duration = time.perf_counter() - total_start
            record_metric("total_duration", total_duration)
            if parent_span:
                parent_span.set_attribute("total_startup_seconds", total_duration)

        # Continue to serve (vLLM native tracing takes over)
        await generate_completions(engine, ...)
```

### 4. K8s Event Watcher (New Deployment)

New service that watches pod lifecycle events.

```python
# k8s_startup_watcher.py
import hashlib
from kubernetes import client, watch
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

LABEL_SELECTOR = "app.kubernetes.io/managed-by=model-engine"


def derive_trace_id(pod_uid: str) -> str:
    return hashlib.md5(pod_uid.encode()).hexdigest()[:32]


def watch_pod_events(namespace: str):
    v1 = client.CoreV1Api()
    w = watch.Watch()
    tracer = trace.get_tracer("k8s-startup-watcher")

    pending_pods = {}  # pod_uid -> timestamps

    for event in w.stream(
        v1.list_namespaced_event,
        namespace=namespace,
        field_selector="involvedObject.kind=Pod"
    ):
        obj = event["object"]
        pod_uid = obj.involved_object.uid
        reason = obj.reason
        timestamp = obj.first_timestamp

        # Filter to managed pods only
        if not is_managed_pod(obj.involved_object.name, namespace):
            continue

        trace_id = derive_trace_id(pod_uid)

        if reason == "Scheduled":
            emit_span(tracer, "pod_scheduled", trace_id, timestamp)
        elif reason == "Pulling":
            pending_pods.setdefault(pod_uid, {})["pull_start"] = timestamp
        elif reason == "Pulled":
            emit_span(tracer, "image_pull", trace_id, timestamp,
                     start=pending_pods.get(pod_uid, {}).get("pull_start"))
        elif reason == "Started":
            emit_span(tracer, "container_started", trace_id, timestamp)
```

Deployment manifest:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: startup-watcher
  namespace: model-engine
spec:
  replicas: 1
  selector:
    matchLabels:
      app: startup-watcher
  template:
    spec:
      serviceAccountName: startup-watcher
      containers:
        - name: watcher
          image: your-registry/startup-watcher:latest
          env:
            - name: OTEL_EXPORTER_OTLP_ENDPOINT
              value: "localhost:4317"
            - name: WATCH_NAMESPACE
              value: "model-endpoints"
          resources:
            requests:
              cpu: 100m
              memory: 128Mi
---
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRole
metadata:
  name: startup-watcher
rules:
  - apiGroups: [""]
    resources: ["pods", "events"]
    verbs: ["get", "list", "watch"]
```

### 5. OTel Collector (DaemonSet)

```yaml
# otel-collector-config.yaml
receivers:
  otlp:
    protocols:
      grpc:
        endpoint: 0.0.0.0:4317

processors:
  batch:
    timeout: 10s

exporters:
  datadog:
    api:
      key: ${DD_API_KEY}
      site: datadoghq.com
    traces:
      span_name_as_resource_name: true
    metrics:
      histograms:
        mode: distributions

service:
  pipelines:
    traces:
      receivers: [otlp]
      processors: [batch]
      exporters: [datadog]
    metrics:
      receivers: [otlp]
      processors: [batch]
      exporters: [datadog]
```

## vLLM Native Tracing

vLLM has built-in OTel support for request tracing (not startup). Enable with:

```bash
vllm serve model --otlp-traces-endpoint="localhost:4317"
```

This traces each inference request automatically once the server is ready.

## Limitations

### vLLM Init is a Black Box

Without forking vLLM, we cannot break down `vllm_init` into sub-phases:
- Weight loading
- KV cache allocation
- CUDA graph compilation

We can only measure the total `AsyncLLM.from_vllm_config()` duration.

### Future Granularity Options

1. **Accept the black box** - sufficient for initial benchmarking
2. **Parse vLLM logs** - fragile but no fork required
3. **Upstream contribution** - vLLM has open issues for startup tracing
4. **Lightweight fork** - add ~15 lines to `LLMEngine.__init__()`

## HA Path for K8s Watcher

Start with single replica. Path to HA:

```
Single Replica (v1) → Leader Election (v2)
       │                      │
       │                      ├── Add Lease-based leader election
       │                      ├── Bump replicas to 2-3
       │                      └── No changes to span emission logic
```

## Implementation Order

1. **Helm chart changes** - Add env vars (POD_UID, GPU_TYPE, etc.)
2. **startup_telemetry.py** - New module with OTel helpers
3. **vllm_batch.py integration** - Wrap startup phases
4. **OTel Collector DaemonSet** - Deploy to cluster
5. **K8s Event Watcher** - Deploy as separate service
6. **Datadog dashboards** - Build startup metrics views

## Example Datadog Queries

```
# P95 total startup by model
p95:startup.total.duration_seconds{*} by {model_name}

# Download throughput by region
avg:startup.download.throughput_mbps{*} by {region}

# vLLM init time: H100 vs A100
avg:startup.vllm_init.duration_seconds{*} by {gpu_type}

# Slowest endpoints
top10:startup.total.duration_seconds{*} by {endpoint_name}
```
