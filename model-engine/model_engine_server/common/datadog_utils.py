from contextvars import ContextVar

from ddtrace import tracer

request_id: ContextVar[str] = ContextVar("request_id", default=None)  # type: ignore


def add_trace_resource_name(tag: str):
    """Adds a custom tag to a given dd trace corresponding to the route
    (e.g. get_model_bundles for GET /model-bundles, etc.) so that we can filter in Datadog easier
    """
    current_span = tracer.current_span()
    if current_span:
        current_span.set_tag("launch.resource_name", tag)


def add_trace_request_id(request_id: str):
    """Adds a custom tag to a given dd trace corresponding to the request id
    so that we can filter in Datadog easier
    """
    current_span = tracer.current_span()
    if current_span:
        current_span.set_tag("launch.request_id", request_id)


def get_request_id():
    """Gets the request id for an api call (in our case, dd trace id) so that we can filter in Datadog easier"""
    current_span = tracer.current_span()
    return current_span.trace_id if current_span else None


def set_request_id_context():
    """Sets the request id context var for an api call (in our case, dd trace id) so that we can filter in Datadog easier"""
    request_id.set(get_request_id())
