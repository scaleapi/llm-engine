from contextlib import contextmanager
from typing import Any, Dict, Generator

from model_engine_server.core.tracing.span import Span
from model_engine_server.core.tracing.tracing_gateway import TracingGateway


class LiveTracingGateway(TracingGateway):
    """
    A default tracing gateway that does not perform any tracing, essentially a no-op.
    """

    def encode_trace_headers(self) -> Dict[str, Any]:
        return {}

    def encode_trace_kwargs(self) -> Dict[str, Any]:
        return {}

    @contextmanager
    def create_span(self, name: str) -> Generator[Span, None, None]:
        yield Span(name=name)
