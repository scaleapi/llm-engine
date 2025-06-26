import asyncio
import base64
import json
from typing import Any, Optional

from mergedeep import Strategy, merge

from model_engine_server.tracing.trace_context_schema import (
    TraceConfig,
)
from model_engine_server.tracing.trace_ctx_var import ctx_var_sgp_trace_queue_manager, ctx_var_sgp_trace_config
from model_engine_server.core.loggers import (
    logger_name,
    make_logger,
)

import scale_gp_beta.lib.tracing.trace_queue_manager as trace_queue_manager
from scale_gp_beta import SGPClient
from scale_gp_beta.lib.tracing.scope import Scope
from scale_gp_beta.lib.tracing.trace import Trace
import os

SGP_TRACE_CONFIG_HEADER = "x-sgp-trace-config"
TRACER_SGP_CLIENT_BASE_URL = os.environ.get('TRACER_SGP_CLIENT_BASE_URL', "http://egp-api-backend.egp.svc.cluster.local:80/private/")

logger = make_logger(logger_name())

def get_parent_span_id() -> Optional[str]:
    current_span = Scope.get_current_span()
    if current_span is not None:
        return current_span.span_id
    caller_trace_config = ctx_var_sgp_trace_config.get()
    return caller_trace_config.parent_span_id if caller_trace_config else None

def get_trace_config_headers(
    extra_metadata: Optional[dict[str, Any]] = None,
) -> dict[str, str]:
    queue_manager = ctx_var_sgp_trace_queue_manager.get()
    if queue_manager is None:
        return None
    caller_trace_config = ctx_var_sgp_trace_config.get()
    trace_config = TraceConfig(
        trace_id=Scope.get_current_trace().trace_id,
        group_id=Scope.get_current_trace().group_id,
        # Note that metadata added to the parent span will NOT automatically be added to the trace config.
        default_metadata=merge_metadata(
            caller_trace_config.default_metadata if caller_trace_config else None,
            extra_metadata
        ),
        parent_span_id=get_parent_span_id(),
    )
    return {SGP_TRACE_CONFIG_HEADER: encode_trace_config(trace_config)}


def encode_trace_config(trace_config: TraceConfig) -> str:
    """
    Encodes the trace context into a string format.
    """
    return base64.b64encode(
        json.dumps(trace_config.model_dump(), sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).decode("utf-8")


def decode_trace_config(serialized_trace_config: str) -> TraceConfig:
    """
    Decodes a string back into a TraceConfig object.
    """
    decoded_str = base64.b64decode(serialized_trace_config.encode("utf-8")).decode("utf-8")
    return TraceConfig.model_validate(json.loads(decoded_str))

def init_trace_queue_manager(trace_config:TraceConfig)->trace_queue_manager.TraceQueueManager:
    tracer_client = SGPClient(
        api_key="ignored-by-endpoint",
        account_id=trace_config.account_id,
        base_url=TRACER_SGP_CLIENT_BASE_URL,
        default_headers={
            "x-sgp-user-id": trace_config.user_id
        }
    )
    queue_manager = trace_queue_manager.TraceQueueManager(
        client=tracer_client,
        worker_enabled=False)
    trace = Trace(
        name="distributed trace", # can this be None? The trace has already been created by SGP, we just need the id.
        trace_id=trace_config.trace_id,
        group_id=trace_config.group_id,
        span_id=trace_config.parent_span_id,
        queue_manager=queue_manager
    )
    # Span is set to the trace's root span automatically, no need to call Scope.set_current_span.
    Scope.set_current_trace(trace)
    return queue_manager



def init_tracing(serialized_trace_config: Optional[str]) -> Optional[trace_queue_manager.TraceQueueManager]:
    """
    Adds a trace context to the stack based on the SGP trace config header value.
    """
    if serialized_trace_config is None:
        return None
    trace_config = None
    try:
        trace_config = decode_trace_config(serialized_trace_config)
        ctx_var_sgp_trace_config.set(trace_config)
        trace_queue_manager_instance = init_trace_queue_manager(trace_config)
        ctx_var_sgp_trace_queue_manager.set(trace_queue_manager_instance)
        return trace_queue_manager_instance
    except Exception as e:
        logger.error(f"Failed to decode trace config: {e} - {serialized_trace_config}")
    return None


def merge_metadata(
    default_metadata: Optional[dict[str, Any]] = None,
    extra_metadata: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """
    Merges default metadata with extra metadata.
    """
    destination = {**(default_metadata or {})}
    return merge(destination, extra_metadata or {}, strategy=Strategy.TYPESAFE_ADDITIVE)
