from fastapi import APIRouter, Depends, Request
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
    model_name = request.query_params.get("model", None)
    return MetricMetadata(user=auth, model_name=model_name)


async def record_route_call(
    external_interfaces: ExternalInterfaces = Depends(get_external_interfaces_read_only),
    route: str = Depends(format_request_route),
    metric_metadata: MetricMetadata = Depends(get_metric_metadata),
):
    external_interfaces.monitoring_metrics_gateway.emit_route_call_metric(route, metric_metadata)


llm_router_v2 = APIRouter(prefix="/v2", dependencies=[Depends(record_route_call)])
