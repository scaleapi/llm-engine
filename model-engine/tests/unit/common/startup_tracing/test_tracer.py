"""Tests for startup_tracing tracer module."""

import os
from unittest.mock import patch

from model_engine_server.common.startup_tracing.tracer import StartupContext, StartupTracer


class TestStartupContext:
    """Tests for StartupContext dataclass."""

    def test_default_values(self):
        """Test default values are set correctly."""
        ctx = StartupContext(pod_uid="test-uid")
        assert ctx.pod_uid == "test-uid"
        assert ctx.pod_name == "unknown"
        assert ctx.node_name == "unknown"
        assert ctx.endpoint_name == "unknown"
        assert ctx.model_name == "unknown"
        assert ctx.gpu_type == "unknown"
        assert ctx.num_gpus == 1
        assert ctx.region == "unknown"

    def test_custom_values(self):
        """Test custom values are set correctly."""
        ctx = StartupContext(
            pod_uid="uid-123",
            pod_name="my-pod",
            node_name="node-1",
            endpoint_name="my-endpoint",
            model_name="llama-7b",
            gpu_type="a100",
            num_gpus=4,
            region="us-west-2",
        )
        assert ctx.pod_uid == "uid-123"
        assert ctx.pod_name == "my-pod"
        assert ctx.node_name == "node-1"
        assert ctx.endpoint_name == "my-endpoint"
        assert ctx.model_name == "llama-7b"
        assert ctx.gpu_type == "a100"
        assert ctx.num_gpus == 4
        assert ctx.region == "us-west-2"

    def test_from_env(self):
        """Test creating context from environment variables."""
        env_vars = {
            "POD_UID": "env-pod-uid",
            "POD_NAME": "env-pod-name",
            "NODE_NAME": "env-node-name",
            "ENDPOINT_NAME": "env-endpoint",
            "MODEL_NAME": "env-model",
            "GPU_TYPE": "h100",
            "NUM_GPUS": "8",
            "AWS_REGION": "us-east-1",
        }
        with patch.dict(os.environ, env_vars, clear=False):
            ctx = StartupContext.from_env()
            assert ctx.pod_uid == "env-pod-uid"
            assert ctx.pod_name == "env-pod-name"
            assert ctx.node_name == "env-node-name"
            assert ctx.endpoint_name == "env-endpoint"
            assert ctx.model_name == "env-model"
            assert ctx.gpu_type == "h100"
            assert ctx.num_gpus == 8
            assert ctx.region == "us-east-1"

    def test_from_env_defaults(self):
        """Test from_env uses defaults when env vars not set."""
        with patch.dict(os.environ, {}, clear=True):
            ctx = StartupContext.from_env()
            assert ctx.pod_uid == "unknown"
            assert ctx.pod_name == "unknown"
            assert ctx.region == "unknown"


class TestStartupTracer:
    """Tests for StartupTracer class."""

    def test_init_without_otel(self):
        """Test initialization when OTel is not available."""
        ctx = StartupContext(pod_uid="test-uid")
        tracer = StartupTracer(ctx, service_name="test-service")
        assert tracer._context == ctx
        assert tracer._service_name == "test-service"
        assert not tracer._initialized

    def test_trace_id_property(self):
        """Test trace_id property returns formatted trace ID."""
        ctx = StartupContext(pod_uid="test-uid-12345")
        tracer = StartupTracer(ctx)
        trace_id = tracer.trace_id
        assert isinstance(trace_id, str)
        assert len(trace_id) == 32

    def test_common_attributes(self):
        """Test _common_attributes returns expected attributes."""
        ctx = StartupContext(
            pod_uid="uid",
            pod_name="my-pod",
            node_name="node-1",
            endpoint_name="endpoint",
            model_name="model",
            gpu_type="a100",
            num_gpus=2,
            region="us-west-2",
        )
        tracer = StartupTracer(ctx)
        attrs = tracer._common_attributes()

        assert attrs["endpoint_name"] == "endpoint"
        assert attrs["model_name"] == "model"
        assert attrs["gpu_type"] == "a100"
        assert attrs["num_gpus"] == 2
        assert attrs["region"] == "us-west-2"
        assert attrs["node_name"] == "node-1"
        assert attrs["pod_name"] == "my-pod"

    def test_create_span_not_initialized(self):
        """Test create_span does nothing when not initialized."""
        ctx = StartupContext(pod_uid="test-uid")
        tracer = StartupTracer(ctx)
        # Should not raise
        tracer.create_span("test", 0, 1000)

    def test_record_metric_not_initialized(self):
        """Test record_metric does nothing when not initialized."""
        ctx = StartupContext(pod_uid="test-uid")
        tracer = StartupTracer(ctx)
        # Should not raise
        tracer.record_metric("test_metric", 1.5)

    def test_complete_not_initialized(self):
        """Test complete returns duration even when not initialized."""
        ctx = StartupContext(pod_uid="test-uid")
        tracer = StartupTracer(ctx)
        duration = tracer.complete()
        assert isinstance(duration, float)
        assert duration >= 0

    def test_flush_not_initialized(self):
        """Test flush does nothing when not initialized."""
        ctx = StartupContext(pod_uid="test-uid")
        tracer = StartupTracer(ctx)
        # Should not raise
        tracer.flush()

    def test_span_context_manager_not_initialized(self):
        """Test span context manager works when not initialized."""
        ctx = StartupContext(pod_uid="test-uid")
        tracer = StartupTracer(ctx)
        with tracer.span("test") as span:
            assert span is None
