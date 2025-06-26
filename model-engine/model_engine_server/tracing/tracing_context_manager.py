from contextlib import contextmanager
from typing import Generator, Optional
from scale_gp_beta.lib.tracing.span import Span
from scale_gp_beta.lib.tracing.scope import Scope
from scale_gp_beta.lib.tracing import create_span
from model_engine_server.tracing.trace_ctx_var import ctx_var_sgp_trace_queue_manager

@contextmanager
def span(name: str)->Generator[Optional[Span], None, None]:
    """
    If there is a trace context, create a span with the given name.
    Otherwise, yield None.
    Automatically flushes the queue manager at the end of the span.
    """
    if Scope.get_current_trace() is None:
        yield None
    else:
        queue_manager = ctx_var_sgp_trace_queue_manager.get()
        with create_span(name, queue_manager=queue_manager) as new_span:
            try:
                yield new_span
            finally:
                queue_manager.flush_queue()
