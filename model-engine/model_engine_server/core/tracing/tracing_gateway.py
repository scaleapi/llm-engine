from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Dict, Generator
from model_engine_server.core.tracing.span import Span

if TYPE_CHECKING:
    from fastapi import Request

class TracingGateway(ABC):

    def extract_tracing_headers(self, request: "Request") -> None:
        """
        Extracts tracing headers from the request and sets them in the current context when present.
        """
        pass

    @abstractmethod
    def encode_trace_headers(self) -> Dict[str, Any]:
        pass

    @abstractmethod
    def encode_trace_kwargs(self) -> Dict[str, Any]:
        pass

    @abstractmethod
    def create_span(self, name: str) -> Generator[Span, None, None]:
        pass
