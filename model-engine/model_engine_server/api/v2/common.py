from fastapi import Depends, Request
from model_engine_server.api.dependencies import (
    ExternalInterfaces,
    get_external_interfaces_read_only,
    verify_authentication,
)
from model_engine_server.core.auth.authentication_repository import User
from model_engine_server.domain.gateways.monitoring_metrics_gateway import MetricMetadata


def format_request_route(request: Request) -> str:
    url_path = request.url.path
    for path_param in request.path_params:
        url_path = url_path.replace(request.path_params[path_param], f":{path_param}")
    return f"{request.method}_{url_path}".lower()


async def get_metric_metadata(
    request: Request,
    auth: User = Depends(verify_authentication),
) -> MetricMetadata:
    # note that this is ok because request will cache the body
    model_name = None
    try:
        body = await request.json()
        model_name = body.get("model", None)
        if not model_name:
            # get model name from batch completion request
            model_name = body.get("model_config", {}).get("model", None)
    except Exception:
        # request has no body
        pass

    return MetricMetadata(user=auth, model_name=model_name)


async def record_route_call(
    external_interfaces: ExternalInterfaces = Depends(get_external_interfaces_read_only),
    route: str = Depends(format_request_route),
    metric_metadata: MetricMetadata = Depends(get_metric_metadata),
):
    external_interfaces.monitoring_metrics_gateway.emit_route_call_metric(route, metric_metadata)
