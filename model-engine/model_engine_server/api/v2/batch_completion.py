from fastapi import APIRouter, Depends, HTTPException
from model_engine_server.api.dependencies import (
    ExternalInterfaces,
    get_external_interfaces,
    get_external_interfaces_read_only,
    verify_authentication,
)
from model_engine_server.common.dtos.llms.batch_completion import (
    CancelBatchCompletionsV2Response,
    CreateBatchCompletionsV2Request,
    CreateBatchCompletionsV2Response,
    GetBatchCompletionV2Response,
    UpdateBatchCompletionsV2Request,
    UpdateBatchCompletionsV2Response,
)
from model_engine_server.core.auth.authentication_repository import User
from model_engine_server.core.loggers import logger_name, make_logger
from model_engine_server.domain.exceptions import (
    ObjectNotAuthorizedException,
    ObjectNotFoundException,
)
from model_engine_server.domain.gateways.monitoring_metrics_gateway import MetricMetadata
from model_engine_server.domain.use_cases.llm_model_endpoint_use_cases import (
    CancelBatchCompletionV2UseCase,
    CreateBatchCompletionsV2UseCase,
    GetBatchCompletionV2UseCase,
    UpdateBatchCompletionV2UseCase,
)

from .common import get_metric_metadata, record_route_call

logger = make_logger(logger_name())


batch_completions_router_v2 = APIRouter(
    prefix="/batch-completions", dependencies=[Depends(record_route_call)]
)


@batch_completions_router_v2.post("/", response_model=CreateBatchCompletionsV2Response)
async def batch_completions(
    request: CreateBatchCompletionsV2Request,
    auth: User = Depends(verify_authentication),
    external_interfaces: ExternalInterfaces = Depends(get_external_interfaces_read_only),
) -> CreateBatchCompletionsV2Response:
    logger.info(f"POST /v2/batch-completions {request} for {auth}")
    try:
        use_case = CreateBatchCompletionsV2UseCase(
            llm_batch_completions_service=external_interfaces.llm_batch_completions_service,
            llm_artifact_gateway=external_interfaces.llm_artifact_gateway,
        )

        return await use_case.execute(request, user=auth)
    except ObjectNotFoundException as exc:
        raise HTTPException(
            status_code=404,
            detail=str(exc),
        ) from exc

    except Exception as exc:
        logger.exception(f"Error processing request {request} for {auth}")
        raise HTTPException(
            status_code=500,
            detail="Internal server error",
        ) from exc


@batch_completions_router_v2.get(
    "/{batch_completion_id}",
    response_model=GetBatchCompletionV2Response,
)
async def get_batch_completion(
    batch_completion_id: str,
    auth: User = Depends(verify_authentication),
    external_interfaces: ExternalInterfaces = Depends(get_external_interfaces_read_only),
    metric_metadata: MetricMetadata = Depends(get_metric_metadata),
) -> GetBatchCompletionV2Response:
    logger.info(f"GET /v2/batch-completions/{batch_completion_id} for {auth}")
    try:
        use_case = GetBatchCompletionV2UseCase(
            llm_batch_completions_service=external_interfaces.llm_batch_completions_service,
        )
        return await use_case.execute(batch_completion_id=batch_completion_id, user=auth)
    except (ObjectNotFoundException, ObjectNotAuthorizedException) as exc:
        raise HTTPException(
            status_code=404,
            detail=str(exc),
        ) from exc


@batch_completions_router_v2.post(
    "/{batch_completion_id}",
    response_model=UpdateBatchCompletionsV2Response,
)
async def update_batch_completion(
    batch_completion_id: str,
    request: UpdateBatchCompletionsV2Request,
    auth: User = Depends(verify_authentication),
    external_interfaces: ExternalInterfaces = Depends(get_external_interfaces),
) -> UpdateBatchCompletionsV2Response:
    logger.info(f"POST /v2/batch-completions/{batch_completion_id} {request} for {auth}")
    try:
        use_case = UpdateBatchCompletionV2UseCase(
            llm_batch_completions_service=external_interfaces.llm_batch_completions_service,
        )
        return await use_case.execute(
            batch_completion_id=batch_completion_id, request=request, user=auth
        )
    except (ObjectNotFoundException, ObjectNotAuthorizedException) as exc:
        raise HTTPException(
            status_code=404,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        logger.exception(f"Error processing request {request} for {auth}", exc_info=exc)
        raise HTTPException(
            status_code=500,
            detail="Internal server error",
        ) from exc


@batch_completions_router_v2.post(
    "/{batch_completion_id}/actions/cancel",
    response_model=CancelBatchCompletionsV2Response,
)
async def cancel_batch_completion(
    batch_completion_id: str,
    auth: User = Depends(verify_authentication),
    external_interfaces: ExternalInterfaces = Depends(get_external_interfaces),
) -> CancelBatchCompletionsV2Response:
    logger.info(f"POST /v2/batch-completions/{batch_completion_id}/actions/cancel for {auth}")
    try:
        use_case = CancelBatchCompletionV2UseCase(
            llm_batch_completions_service=external_interfaces.llm_batch_completions_service,
        )
        return await use_case.execute(batch_completion_id=batch_completion_id, user=auth)
    except (ObjectNotFoundException, ObjectNotAuthorizedException) as exc:
        raise HTTPException(
            status_code=404,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        logger.exception(
            f"Error canceling batch completions {batch_completion_id} for {auth}",
            exc_info=exc,
        )
        raise HTTPException(
            status_code=500,
            detail="Internal server error",
        ) from exc
