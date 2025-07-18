from functools import lru_cache
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from model_engine_server.core.tracing.tracing_gateway import TracingGateway


@lru_cache(maxsize=1)
def get_tracing_gateway() -> "TracingGateway":
    """
    Returns the configured tracing gateway. Cached for performance.
    """
    try:
        from plugins.tracing import get_tracing_gateway as get_custom_tracing_gateway

        return get_custom_tracing_gateway()
    except ModuleNotFoundError:
        pass
    from model_engine_server.core.tracing.live_tracing_gateway import LiveTracingGateway

    return LiveTracingGateway()
