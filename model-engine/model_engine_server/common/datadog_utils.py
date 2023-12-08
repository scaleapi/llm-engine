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
