"""Tests for startup_tracing correlation module."""

from model_engine_server.common.startup_tracing.correlation import (
    derive_span_id,
    derive_trace_id,
    format_trace_id,
)


class TestDeriveTraceId:
    """Tests for derive_trace_id function."""

    def test_derive_trace_id_deterministic(self):
        """Same input should always produce same output."""
        unique_id = "test-pod-uid-12345"
        result1 = derive_trace_id(unique_id)
        result2 = derive_trace_id(unique_id)
        assert result1 == result2

    def test_derive_trace_id_different_inputs(self):
        """Different inputs should produce different outputs."""
        result1 = derive_trace_id("pod-uid-1")
        result2 = derive_trace_id("pod-uid-2")
        assert result1 != result2

    def test_derive_trace_id_returns_int(self):
        """Should return an integer."""
        result = derive_trace_id("test-pod-uid")
        assert isinstance(result, int)

    def test_derive_trace_id_positive(self):
        """Should return a positive integer."""
        result = derive_trace_id("test-pod-uid")
        assert result > 0


class TestDeriveSpanId:
    """Tests for derive_span_id function."""

    def test_derive_span_id_deterministic(self):
        """Same inputs should always produce same output."""
        unique_id = "test-pod-uid"
        suffix = "download"
        result1 = derive_span_id(unique_id, suffix)
        result2 = derive_span_id(unique_id, suffix)
        assert result1 == result2

    def test_derive_span_id_different_suffixes(self):
        """Different suffixes should produce different span IDs."""
        unique_id = "test-pod-uid"
        result1 = derive_span_id(unique_id, "download")
        result2 = derive_span_id(unique_id, "vllm_init")
        assert result1 != result2

    def test_derive_span_id_different_unique_ids(self):
        """Different unique IDs should produce different span IDs."""
        suffix = "download"
        result1 = derive_span_id("pod-1", suffix)
        result2 = derive_span_id("pod-2", suffix)
        assert result1 != result2

    def test_derive_span_id_returns_int(self):
        """Should return an integer."""
        result = derive_span_id("test-pod-uid", "suffix")
        assert isinstance(result, int)


class TestFormatTraceId:
    """Tests for format_trace_id function."""

    def test_format_trace_id_returns_hex_string(self):
        """Should return a hex string."""
        result = format_trace_id("test-pod-uid")
        assert isinstance(result, str)
        # Should be valid hex
        int(result, 16)

    def test_format_trace_id_length(self):
        """Should return 32 character hex string."""
        result = format_trace_id("test-pod-uid")
        assert len(result) == 32

    def test_format_trace_id_deterministic(self):
        """Same input should produce same output."""
        result1 = format_trace_id("test-pod-uid")
        result2 = format_trace_id("test-pod-uid")
        assert result1 == result2
