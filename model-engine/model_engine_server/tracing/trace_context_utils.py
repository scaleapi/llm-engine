import asyncio
import base64
import json
from typing import Any, Optional

from mergedeep import Strategy, merge

from model_engine_server.tracing.trace_context_schema import (
    TraceConfig,
)

from scale_gp_beta.lib.tracing import BaseSpan
from scale_gp_beta.lib.tracing.scope import Scope


SGP_TRACE_CONFIG_HEADER = "x-sgp-trace-config"


def get_trace_config_headers(
    extra_metadata: Optional[dict[str, Any]] = None,
) -> dict[str, str]:
    headers = {}
    trace_state = ctx_var_sgp_trace_state.get()
    if trace_state is not None:
        trace_config = TraceConfig(
            trace_id=trace_state.trace_config.trace_id,
            group_id=trace_state.trace_config.group_id,
            # Note that metadata added to the parent span will NOT automatically be added to the trace config.
            default_metadata=merge_metadata(
                trace_state.trace_config.default_metadata, extra_metadata
            ),
            parent_span_id=get_parent_span_id(trace_state),
        )
        headers[SGP_TRACE_CONFIG_HEADER] = encode_trace_config(trace_config)
    return headers


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


def init_trace_state(serialized_trace_config: Optional[str]) -> Optional[SGPTraceState]:
    """
    Adds a trace context to the stack based on the SGP trace config header value.
    """
    if serialized_trace_config is None:
        return None
    trace_config = None
    try:
        trace_config = decode_trace_config(serialized_trace_config)
        # Create a new SGPTraceState with the trace config
        trace_state = SGPTraceState(
            trace_config=trace_config,
        )
        ctx_var_sgp_trace_state.set(trace_state)
        return trace_state
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
