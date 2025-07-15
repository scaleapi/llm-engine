from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from model_engine_server.core.tracing.tracing_gateway import TracingGateway

def get_tracing_gateway() -> "TracingGateway":
    """
    Returns the configured tracing gateway.
    """
    try:
        from plugins.tracing import (
            get_tracing_gateway as get_custom_tracing_gateway
        )
        return get_custom_tracing_gateway()
    except ModuleNotFoundError:
        pass
    from model_engine_server.core.tracing.fake_tracing_gateway import FakeTracingGateway
    return FakeTracingGateway()