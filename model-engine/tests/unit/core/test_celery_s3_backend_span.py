from unittest import mock

from model_engine_server.core.celery import s3 as s3_backend_module


def test_backend_get_span_nullcontext_without_tracer(monkeypatch):
    monkeypatch.setattr(s3_backend_module, "_tracer", None)
    with s3_backend_module._backend_get_span():
        pass  # must be a usable no-op context manager


def test_backend_get_span_uses_tracer_when_available(monkeypatch):
    tracer = mock.MagicMock()
    monkeypatch.setattr(s3_backend_module, "_tracer", tracer)
    s3_backend_module._backend_get_span()
    tracer.trace.assert_called_once_with("celery.s3_backend.get", span_type="storage")
