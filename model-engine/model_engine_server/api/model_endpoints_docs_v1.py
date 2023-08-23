from fastapi import APIRouter, Depends
from fastapi.encoders import jsonable_encoder
from fastapi.openapi.docs import get_redoc_html
from fastapi.responses import JSONResponse
from model_engine_server.api.dependencies import (
    ExternalInterfaces,
    get_external_interfaces_read_only,
    verify_authentication,
)
from model_engine_server.core.auth.authentication_repository import User
from model_engine_server.core.loggers import filename_wo_ext, make_logger
from model_engine_server.domain.use_cases.model_endpoints_schema_use_cases import (
    GetModelEndpointsSchemaV1UseCase,
)
from starlette.responses import HTMLResponse

model_endpoints_docs_router_v1 = APIRouter(prefix="/v1")
logger = make_logger(filename_wo_ext(__name__))


@model_endpoints_docs_router_v1.get("/model-endpoints-schema.json")
async def get_model_endpoints_schema(
    auth: User = Depends(verify_authentication),
    external_interfaces: ExternalInterfaces = Depends(get_external_interfaces_read_only),
) -> JSONResponse:
    """
    Lists the schemas of the Model Endpoints owned by the current owner.
    """
    logger.info(f"GET /model-endpoints-schema.json for {auth}")
    use_case = GetModelEndpointsSchemaV1UseCase(
        model_endpoint_service=external_interfaces.model_endpoint_service
    )
    response = await use_case.execute(auth)
    return jsonable_encoder(response.model_endpoints_schema, by_alias=True, exclude_none=True)


@model_endpoints_docs_router_v1.get("/model-endpoints-api")
async def get_model_endpoints_api(
    auth: User = Depends(verify_authentication),
) -> HTMLResponse:
    """
    Shows the API of the Model Endpoints owned by the current owner.
    """
    logger.info(f"GET /model-endpoints-api for {auth}")
    return get_redoc_html(
        openapi_url="/v1/model-endpoints-schema.json",
        title="Model Endpoints Documentation",
    )
