from typing import Optional

from ddtrace import tracer


def add_trace_request_id(request_id: Optional[str]):
    """Adds a custom tag to a given dd trace corresponding to the request id
    so that we can filter in Datadog easier
    """
    if not request_id:
        return

    current_span = tracer.current_span()
    if current_span:
        current_span.set_tag("launch.request_id", request_id)


def add_trace_model_name(model_name: Optional[str]):
    """Adds a custom tag to a given dd trace corresponding to the model name
    so that we can filter in Datadog easier

    Only use this when the number of model names is small, otherwise it will
    blow up the cardinality in Datadog
    """
    if not model_name:
        return

    current_span = tracer.current_span()
    if current_span:
        current_span.set_tag("launch.model_name", model_name)
        current_span.set_tag("_dd.p.launch_model_name", model_name)
