from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, Dict, Generator
from model_engine_server.core.tracing.span import Span
from model_engine_server.core.tracing.tracing_gateway import TracingGateway

class FakeTracingGateway(TracingGateway):

    def encode_trace_headers(self) -> Dict[str, Any]:
        return {}

    def encode_trace_kwargs(self) -> Dict[str, Any]:
        return {}

    @contextmanager
    def create_span(self, name: str) -> Generator[Span, None, None]:
        yield Span(name=name)