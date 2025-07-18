from abc import ABC, abstractmethod
from contextlib import AbstractContextManager
from typing import TYPE_CHECKING, Any, Dict, Optional, Union

from model_engine_server.core.tracing.span import Span

if TYPE_CHECKING:
    from fastapi import Request


class TracingGateway(ABC):

    def extract_tracing_headers(
        self, request: Union["Request", str, dict], service: Optional[str] = None
    ) -> Optional[str]:
        """
        Extracts tracing headers from the request and sets them in the current context when present.
        Accepts either
        - a FastAPI Request object
        - A kwargs dictionary containing the appropriate keyword argument with the encoded trace config.
          the name of the keyword argument is defined in:
          https://github.com/scaleapi/scaleapi/blob/c09aac749496478c1643c680dc8d49c62ede8a6c/packages/egp-api-backend/s2s_tracing/s2s_tracing/constants.py#L7
        - the string value of the base64-encoded JSON tracing configuration (as sent in the HTTP header)
        Returns the serialized trace config if it was found, otherwise None.
        """
        return None

    def encode_trace_config(self) -> Optional[str]:
        """
        Encodes the current trace configuration into a base64-encoded JSON string.
        Returns None if no trace configuration is set.
        """
        headers = self.encode_trace_headers()
        # return first header value if available, otherwise None
        return None if len(headers) == 0 else list(headers.values())[0]

    @abstractmethod
    def encode_trace_headers(self) -> Dict[str, Any]:
        pass

    @abstractmethod
    def encode_trace_kwargs(self) -> Dict[str, Any]:
        pass

    @abstractmethod
    def create_span(self, name: str) -> AbstractContextManager[Span]:
        pass
