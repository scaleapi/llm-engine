from contextlib import contextmanager
import datetime
from typing import Generator, Optional
from scale_gp_beta.lib.tracing.span import Span
from scale_gp_beta.lib.tracing.types import SpanStatusLiterals
from scale_gp_beta.lib.tracing.scope import Scope
from scale_gp_beta.lib.tracing import create_span
from model_engine_server.tracing.trace_ctx_var import ctx_var_sgp_trace_queue_manager, ctx_var_sgp_trace_config

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
        exception_raised = None
        queue_manager = ctx_var_sgp_trace_queue_manager.get()
        parent_span_id = Scope.get_current_span().span_id if Scope.get_current_span() else ctx_var_sgp_trace_config.get().parent_span_id
        with create_span(name, parent_id=parent_span_id, queue_manager=queue_manager) as new_span:
            try:
                yield new_span
            except Exception as e:
                exception_raised = e
            finally:
                if new_span is not None:
                    new_span.end_time = datetime.datetime.now(datetime.timezone.utc)
                    if exception_raised:
                        new_span.metadata["exception"] = str(exception_raised)
                        new_span.status = SpanStatusLiterals.ERROR
                    queue_manager.flush_queue()
