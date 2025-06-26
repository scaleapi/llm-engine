import contextvars
from typing import Optional, Any

SGP_TRACE_CONFIG_HEADER = "x-sgp-trace-config"

ctx_var_sgp_trace_config:contextvars.ContextVar[Optional[Any]] = contextvars.ContextVar("ctx_var_sgp_trace_config", default=None)
# Type should be import scale_gp_beta.lib.tracing.trace_queue_manager.TraceQueueManager, but to avoid circular imports we use Any.
ctx_var_sgp_trace_queue_manager:contextvars.ContextVar[Optional[Any]] = contextvars.ContextVar("ctx_var_sgp_trace_queue_manager", default=None)