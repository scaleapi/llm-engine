"""LLM Model Endpoint routes for the hosted model inference service."""

import traceback
from datetime import datetime
from typing import Optional

import pytz
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query, Request
from model_engine_server.api.dependencies import (
    ExternalInterfaces,
    get_external_interfaces,
    get_external_interfaces_read_only,
    verify_authentication,
)
from model_engine_server.common.config import hmi_config
from model_engine_server.common.dtos.llms import (
    CancelFineTuneResponse,
    CompletionStreamV1Request,
    CompletionStreamV1Response,
    CompletionSyncV1Request,
    CompletionSyncV1Response,
    CreateBatchCompletionsV1Request,
    CreateBatchCompletionsV1Response,
    CreateFineTuneRequest,
    CreateFineTuneResponse,
    CreateLLMModelEndpointV1Request,
    CreateLLMModelEndpointV1Response,
    DeleteLLMEndpointResponse,
    GetFineTuneEventsResponse,
    GetFineTuneResponse,
    GetLLMModelEndpointV1Response,
    ListFineTunesResponse,
    ListLLMModelEndpointsV1Response,
    ModelDownloadRequest,
    ModelDownloadResponse,
    StreamError,
    StreamErrorContent,
    TokenUsage,
    UpdateLLMModelEndpointV1Request,
    UpdateLLMModelEndpointV1Response,
)
from model_engine_server.common.dtos.model_endpoints import ModelEndpointOrderBy
from model_engine_server.core.auth.authentication_repository import User
from model_engine_server.core.loggers import (
    LoggerTagKey,
    LoggerTagManager,
    logger_name,
    make_logger,
)
from model_engine_server.core.utils.timer import timer
from model_engine_server.domain.exceptions import (
    DockerImageNotFoundException,
    EndpointDeleteFailedException,
    EndpointLabelsException,
    EndpointResourceInvalidRequestException,
    EndpointUnsupportedInferenceTypeException,
    ExistingEndpointOperationInProgressException,
    FailToInferHardwareException,
    InvalidRequestException,
    LLMFineTuningMethodNotImplementedException,
    LLMFineTuningQuotaReached,
    ObjectAlreadyExistsException,
    ObjectHasInvalidValueException,
    ObjectNotAuthorizedException,
    ObjectNotFoundException,
    UpstreamServiceError,
)
from model_engine_server.domain.gateways.monitoring_metrics_gateway import MetricMetadata
from model_engine_server.domain.use_cases.llm_fine_tuning_use_cases import (
    CancelFineTuneV1UseCase,
    CreateFineTuneV1UseCase,
    GetFineTuneEventsV1UseCase,
    GetFineTuneV1UseCase,
    ListFineTunesV1UseCase,
)
from model_engine_server.domain.use_cases.llm_model_endpoint_use_cases import (
    CompletionStreamV1UseCase,
    CompletionSyncV1UseCase,
    CreateBatchCompletionsUseCase,
    CreateLLMModelBundleV1UseCase,
    CreateLLMModelEndpointV1UseCase,
    DeleteLLMEndpointByNameUseCase,
    GetLLMModelEndpointByNameV1UseCase,
    ListLLMModelEndpointsV1UseCase,
    ModelDownloadV1UseCase,
    UpdateLLMModelEndpointV1UseCase,
)
from model_engine_server.domain.use_cases.model_bundle_use_cases import CreateModelBundleV2UseCase
from pydantic import RootModel
from sse_starlette.sse import EventSourceResponse


def format_request_route(request: Request) -> str:
    url_path = request.url.path
    for path_param in request.path_params:
        url_path = url_path.replace(request.path_params[path_param], f":{path_param}")
    return f"{request.method}_{url_path}".lower()


async def get_metric_metadata(
    request: Request,
    auth: User = Depends(verify_authentication),
) -> MetricMetadata:
    model_name = request.query_params.get("model_endpoint_name", None)
    return MetricMetadata(user=auth, model_name=model_name)


async def record_route_call(
    external_interfaces: ExternalInterfaces = Depends(get_external_interfaces_read_only),
    route: str = Depends(format_request_route),
    metric_metadata: MetricMetadata = Depends(get_metric_metadata),
):
    external_interfaces.monitoring_metrics_gateway.emit_route_call_metric(route, metric_metadata)


llm_router_v1 = APIRouter(prefix="/v1/llm", dependencies=[Depends(record_route_call)])
logger = make_logger(logger_name())


def handle_streaming_exception(
    e: Exception,
    code: int,
    message: str,
):
    tb_str = traceback.format_exception(e)
    request_id = LoggerTagManager.get(LoggerTagKey.REQUEST_ID)
    timestamp = datetime.now(pytz.timezone("US/Pacific")).strftime("%Y-%m-%d %H:%M:%S %Z")
    structured_log = {
        "error": message,
        "request_id": str(request_id),
        "traceback": "".join(tb_str),
    }
    logger.error("Exception: %s", structured_log)
    return {
        "data": CompletionStreamV1Response(
            request_id=str(request_id),
            error=StreamError(
                status_code=code,
                content=StreamErrorContent(
                    error=message,
                    timestamp=timestamp,
                ),
            ),
        ).json()
    }


@llm_router_v1.post("/model-endpoints", response_model=CreateLLMModelEndpointV1Response)
async def create_model_endpoint(
    wrapped_request: RootModel[CreateLLMModelEndpointV1Request],
    auth: User = Depends(verify_authentication),
    external_interfaces: ExternalInterfaces = Depends(get_external_interfaces),
) -> CreateLLMModelEndpointV1Response:
    request = wrapped_request.root
    """
    Creates an LLM endpoint for the current user.
    """
    logger.info(f"POST /llm/model-endpoints with {request} for {auth}")
    try:
        create_model_bundle_use_case = CreateModelBundleV2UseCase(
            model_bundle_repository=external_interfaces.model_bundle_repository,
            docker_repository=external_interfaces.docker_repository,
            model_primitive_gateway=external_interfaces.model_primitive_gateway,
        )
        create_llm_model_bundle_use_case = CreateLLMModelBundleV1UseCase(
            create_model_bundle_use_case=create_model_bundle_use_case,
            model_bundle_repository=external_interfaces.model_bundle_repository,
            llm_artifact_gateway=external_interfaces.llm_artifact_gateway,
            docker_repository=external_interfaces.docker_repository,
        )
        use_case = CreateLLMModelEndpointV1UseCase(
            create_llm_model_bundle_use_case=create_llm_model_bundle_use_case,
            model_endpoint_service=external_interfaces.model_endpoint_service,
            docker_repository=external_interfaces.docker_repository,
            llm_artifact_gateway=external_interfaces.llm_artifact_gateway,
        )
        return await use_case.execute(user=auth, request=request)
    except ObjectAlreadyExistsException as exc:
        raise HTTPException(
            status_code=400,
            detail="The specified model endpoint already exists.",
        ) from exc
    except EndpointLabelsException as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc
    except ObjectHasInvalidValueException as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except EndpointResourceInvalidRequestException as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc
    except (ObjectNotFoundException, ObjectNotAuthorizedException) as exc:
        raise HTTPException(
            status_code=404,
            detail="The specified model bundle could not be found.",
        ) from exc
    except DockerImageNotFoundException as exc:
        raise HTTPException(
            status_code=404,
            detail="The specified docker image could not be found.",
        ) from exc
    except FailToInferHardwareException as exc:
        raise HTTPException(
            status_code=500,
            detail="Failed to infer hardware exception.",
        ) from exc


@llm_router_v1.get("/model-endpoints", response_model=ListLLMModelEndpointsV1Response)
async def list_model_endpoints(
    name: Optional[str] = Query(default=None),
    order_by: Optional[ModelEndpointOrderBy] = Query(default=None),
    auth: User = Depends(verify_authentication),
    external_interfaces: ExternalInterfaces = Depends(get_external_interfaces_read_only),
) -> ListLLMModelEndpointsV1Response:
    """
    Lists the LLM model endpoints owned by the current owner, plus all public_inference LLMs.
    """
    logger.info(f"GET /llm/model-endpoints?name={name}&order_by={order_by} for {auth}")
    use_case = ListLLMModelEndpointsV1UseCase(
        llm_model_endpoint_service=external_interfaces.llm_model_endpoint_service,
    )
    return await use_case.execute(user=auth, name=name, order_by=order_by)


@llm_router_v1.get(
    "/model-endpoints/{model_endpoint_name}",
    response_model=GetLLMModelEndpointV1Response,
)
async def get_model_endpoint(
    model_endpoint_name: str,
    auth: User = Depends(verify_authentication),
    external_interfaces: ExternalInterfaces = Depends(get_external_interfaces_read_only),
) -> GetLLMModelEndpointV1Response:
    """
    Describe the LLM Model endpoint with given name.
    """
    logger.info(f"GET /llm/model-endpoints/{model_endpoint_name} for {auth}")
    try:
        use_case = GetLLMModelEndpointByNameV1UseCase(
            llm_model_endpoint_service=external_interfaces.llm_model_endpoint_service
        )
        return await use_case.execute(user=auth, model_endpoint_name=model_endpoint_name)
    except (ObjectNotFoundException, ObjectNotAuthorizedException) as exc:
        if isinstance(exc, ObjectNotAuthorizedException):  # pragma: no cover
            logger.info(
                f"GET /llm/model-endpoints/{model_endpoint_name} for {auth} failed with authz error {exc.args}"
            )

        raise HTTPException(
            status_code=404,
            detail=f"Model Endpoint {model_endpoint_name}  was not found.",
        ) from exc


@llm_router_v1.put(
    "/model-endpoints/{model_endpoint_name}",
    response_model=UpdateLLMModelEndpointV1Response,
)
async def update_model_endpoint(
    model_endpoint_name: str,
    wrapped_request: RootModel[UpdateLLMModelEndpointV1Request],
    auth: User = Depends(verify_authentication),
    external_interfaces: ExternalInterfaces = Depends(get_external_interfaces),
) -> UpdateLLMModelEndpointV1Response:
    """
    Updates an LLM endpoint for the current user.
    """
    request = wrapped_request.root
    logger.info(f"PUT /llm/model-endpoints/{model_endpoint_name} with {request} for {auth}")
    try:
        create_model_bundle_use_case = CreateModelBundleV2UseCase(
            model_bundle_repository=external_interfaces.model_bundle_repository,
            docker_repository=external_interfaces.docker_repository,
            model_primitive_gateway=external_interfaces.model_primitive_gateway,
        )
        create_llm_model_bundle_use_case = CreateLLMModelBundleV1UseCase(
            create_model_bundle_use_case=create_model_bundle_use_case,
            model_bundle_repository=external_interfaces.model_bundle_repository,
            llm_artifact_gateway=external_interfaces.llm_artifact_gateway,
            docker_repository=external_interfaces.docker_repository,
        )
        use_case = UpdateLLMModelEndpointV1UseCase(
            create_llm_model_bundle_use_case=create_llm_model_bundle_use_case,
            model_endpoint_service=external_interfaces.model_endpoint_service,
            llm_model_endpoint_service=external_interfaces.llm_model_endpoint_service,
            docker_repository=external_interfaces.docker_repository,
        )
        return await use_case.execute(
            user=auth, model_endpoint_name=model_endpoint_name, request=request
        )
    except EndpointLabelsException as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc
    except ObjectHasInvalidValueException as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except EndpointResourceInvalidRequestException as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc
    except (ObjectNotFoundException, ObjectNotAuthorizedException) as exc:
        raise HTTPException(
            status_code=404,
            detail="The specified LLM endpoint could not be found.",
        ) from exc
    except DockerImageNotFoundException as exc:
        raise HTTPException(
            status_code=404,
            detail="The specified docker image could not be found.",
        ) from exc


@llm_router_v1.post("/completions-sync", response_model=CompletionSyncV1Response)
async def create_completion_sync_task(
    model_endpoint_name: str,
    request: CompletionSyncV1Request,
    background_tasks: BackgroundTasks,
    auth: User = Depends(verify_authentication),
    external_interfaces: ExternalInterfaces = Depends(get_external_interfaces_read_only),
    metric_metadata: MetricMetadata = Depends(get_metric_metadata),
) -> CompletionSyncV1Response:
    """
    Runs a sync prompt completion on an LLM.
    """
    if hmi_config.sensitive_log_mode:  # pragma: no cover
        logger.info(f"POST /completions-sync to endpoint {model_endpoint_name} for {auth}")
    else:
        logger.info(
            f"POST /completions-sync with {request} to endpoint {model_endpoint_name} for {auth}"
        )
    try:
        use_case = CompletionSyncV1UseCase(
            model_endpoint_service=external_interfaces.model_endpoint_service,
            llm_model_endpoint_service=external_interfaces.llm_model_endpoint_service,
            tokenizer_repository=external_interfaces.tokenizer_repository,
        )
        with timer() as use_case_timer:
            response = await use_case.execute(
                user=auth, model_endpoint_name=model_endpoint_name, request=request
            )
        background_tasks.add_task(
            external_interfaces.monitoring_metrics_gateway.emit_token_count_metrics,
            TokenUsage(
                num_prompt_tokens=(response.output.num_prompt_tokens if response.output else None),
                num_completion_tokens=(
                    response.output.num_completion_tokens if response.output else None
                ),
                total_duration=use_case_timer.duration,
            ),
            metric_metadata,
        )
        return response
    except UpstreamServiceError as exc:
        request_id = LoggerTagManager.get(LoggerTagKey.REQUEST_ID)
        logger.exception(
            f"Upstream service error for request {request_id}. Error detail: {str(exc.content)}"
        )
        raise HTTPException(
            status_code=500,
            detail=f"Upstream service error for request_id {request_id}",
        )
    except (ObjectNotFoundException, ObjectNotAuthorizedException) as exc:
        if isinstance(exc, ObjectNotAuthorizedException):  # pragma: no cover
            logger.info(
                f"POST /completions-sync to endpoint {model_endpoint_name} for {auth} failed with authz error {exc.args}"
            )

        raise HTTPException(
            status_code=404,
            detail="The specified endpoint could not be found.",
        ) from exc
    except ObjectHasInvalidValueException as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except InvalidRequestException as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except EndpointUnsupportedInferenceTypeException as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported inference type: {str(exc)}",
        ) from exc


@llm_router_v1.post("/completions-stream", response_model=CompletionStreamV1Response)
async def create_completion_stream_task(
    model_endpoint_name: str,
    request: CompletionStreamV1Request,
    background_tasks: BackgroundTasks,
    auth: User = Depends(verify_authentication),
    external_interfaces: ExternalInterfaces = Depends(get_external_interfaces_read_only),
    metric_metadata: MetricMetadata = Depends(get_metric_metadata),
) -> EventSourceResponse:
    """
    Runs a stream prompt completion on an LLM.
    """
    if hmi_config.sensitive_log_mode:  # pragma: no cover
        logger.info(f"POST /completions-stream to endpoint {model_endpoint_name} for {auth}")
    else:
        logger.info(
            f"POST /completions-stream with {request} to endpoint {model_endpoint_name} for {auth}"
        )
    use_case = CompletionStreamV1UseCase(
        model_endpoint_service=external_interfaces.model_endpoint_service,
        llm_model_endpoint_service=external_interfaces.llm_model_endpoint_service,
        tokenizer_repository=external_interfaces.tokenizer_repository,
    )

    try:
        # Call execute() with await, since it needs to handle exceptions before we begin streaming the response below.
        # execute() will create a response chunk generator and return a reference to it.
        response = await use_case.execute(
            user=auth, model_endpoint_name=model_endpoint_name, request=request
        )
    except (ObjectNotFoundException, ObjectNotAuthorizedException) as exc:
        raise HTTPException(
            status_code=404,
            detail=str(exc),
        ) from exc
    except EndpointUnsupportedInferenceTypeException as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc
    except ObjectHasInvalidValueException as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="Internal error occurred. Our team has been notified.",
        ) from exc

    async def event_generator():
        try:
            time_to_first_token = None
            with timer() as use_case_timer:
                async for message in response:
                    if time_to_first_token is None and message.output is not None:
                        time_to_first_token = use_case_timer.lap()
                    yield {"data": message.json()}
            background_tasks.add_task(
                external_interfaces.monitoring_metrics_gateway.emit_token_count_metrics,
                TokenUsage(
                    num_prompt_tokens=(
                        message.output.num_prompt_tokens if message.output else None
                    ),
                    num_completion_tokens=(
                        message.output.num_completion_tokens if message.output else None
                    ),
                    total_duration=use_case_timer.duration,
                    time_to_first_token=time_to_first_token,
                ),
                metric_metadata,
            )
        # The following two exceptions are only raised after streaming begins, so we wrap the exception within a Response object
        except InvalidRequestException as exc:
            yield handle_streaming_exception(exc, 400, str(exc))
        except UpstreamServiceError as exc:
            request_id = LoggerTagManager.get(LoggerTagKey.REQUEST_ID)
            logger.exception(
                f"Upstream service error for request {request_id}. Error detail: {str(exc.content)}"
            )
            yield handle_streaming_exception(
                exc,
                500,
                f"Upstream service error for request_id {request_id}",
            )
        except Exception as exc:
            yield handle_streaming_exception(
                exc, 500, "Internal error occurred. Our team has been notified."
            )

    return EventSourceResponse(event_generator())


@llm_router_v1.post("/fine-tunes", response_model=CreateFineTuneResponse)
async def create_fine_tune(
    request: CreateFineTuneRequest,
    auth: User = Depends(verify_authentication),
    external_interfaces: ExternalInterfaces = Depends(get_external_interfaces),
) -> CreateFineTuneResponse:
    logger.info(f"POST /fine-tunes with {request} for {auth}")
    try:
        use_case = CreateFineTuneV1UseCase(
            llm_fine_tuning_service=external_interfaces.llm_fine_tuning_service,
            model_endpoint_service=external_interfaces.model_endpoint_service,
            llm_fine_tune_events_repository=external_interfaces.llm_fine_tune_events_repository,
            file_storage_gateway=external_interfaces.file_storage_gateway,
        )
        return await use_case.execute(user=auth, request=request)
    except (ObjectNotFoundException, ObjectNotAuthorizedException) as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except (
        LLMFineTuningMethodNotImplementedException,
        LLMFineTuningQuotaReached,
        InvalidRequestException,
    ) as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc


@llm_router_v1.get("/fine-tunes/{fine_tune_id}", response_model=GetFineTuneResponse)
async def get_fine_tune(
    fine_tune_id: str,
    auth: User = Depends(verify_authentication),
    external_interfaces: ExternalInterfaces = Depends(get_external_interfaces_read_only),
) -> GetFineTuneResponse:
    logger.info(f"GET /fine-tunes/{fine_tune_id} for {auth}")
    try:
        use_case = GetFineTuneV1UseCase(
            llm_fine_tuning_service=external_interfaces.llm_fine_tuning_service,
        )
        return await use_case.execute(user=auth, fine_tune_id=fine_tune_id)
    except (ObjectNotFoundException, ObjectNotAuthorizedException) as exc:
        raise HTTPException(
            status_code=404,
            detail="The specified fine-tune job could not be found.",
        ) from exc


@llm_router_v1.get("/fine-tunes", response_model=ListFineTunesResponse)
async def list_fine_tunes(
    auth: User = Depends(verify_authentication),
    external_interfaces: ExternalInterfaces = Depends(get_external_interfaces_read_only),
) -> ListFineTunesResponse:
    logger.info(f"GET /fine-tunes for {auth}")
    use_case = ListFineTunesV1UseCase(
        llm_fine_tuning_service=external_interfaces.llm_fine_tuning_service,
    )
    return await use_case.execute(user=auth)


@llm_router_v1.put("/fine-tunes/{fine_tune_id}/cancel", response_model=CancelFineTuneResponse)
async def cancel_fine_tune(
    fine_tune_id: str,
    auth: User = Depends(verify_authentication),
    external_interfaces: ExternalInterfaces = Depends(get_external_interfaces),
) -> CancelFineTuneResponse:
    logger.info(f"PUT /fine-tunes/{fine_tune_id}/cancel for {auth}")
    try:
        use_case = CancelFineTuneV1UseCase(
            llm_fine_tuning_service=external_interfaces.llm_fine_tuning_service,
        )
        return await use_case.execute(user=auth, fine_tune_id=fine_tune_id)
    except (ObjectNotFoundException, ObjectNotAuthorizedException) as exc:
        raise HTTPException(
            status_code=404,
            detail="The specified fine-tune job could not be found.",
        ) from exc


@llm_router_v1.get("/fine-tunes/{fine_tune_id}/events", response_model=GetFineTuneEventsResponse)
async def get_fine_tune_events(
    fine_tune_id: str,
    auth: User = Depends(verify_authentication),
    external_interfaces: ExternalInterfaces = Depends(get_external_interfaces_read_only),
) -> GetFineTuneEventsResponse:
    logger.info(f"GET /fine-tunes/{fine_tune_id}/events for {auth}")
    try:
        use_case = GetFineTuneEventsV1UseCase(
            llm_fine_tune_events_repository=external_interfaces.llm_fine_tune_events_repository,
            llm_fine_tuning_service=external_interfaces.llm_fine_tuning_service,
        )
        return await use_case.execute(user=auth, fine_tune_id=fine_tune_id)
    except (ObjectNotFoundException, ObjectNotAuthorizedException) as exc:
        raise HTTPException(
            status_code=404,
            detail="The specified fine-tune job's events could not be found.",
        ) from exc


@llm_router_v1.post("/model-endpoints/download", response_model=ModelDownloadResponse)
async def download_model_endpoint(
    request: ModelDownloadRequest,
    auth: User = Depends(verify_authentication),
    external_interfaces: ExternalInterfaces = Depends(get_external_interfaces),
) -> ModelDownloadResponse:
    logger.info(f"POST /model-endpoints/download with {request} for {auth}")
    try:
        use_case = ModelDownloadV1UseCase(
            filesystem_gateway=external_interfaces.filesystem_gateway,
            model_endpoint_service=external_interfaces.model_endpoint_service,
            llm_artifact_gateway=external_interfaces.llm_artifact_gateway,
        )
        return await use_case.execute(user=auth, request=request)
    except (ObjectNotFoundException, ObjectHasInvalidValueException) as exc:
        raise HTTPException(
            status_code=404,
            detail="The requested fine-tuned model could not be found.",
        ) from exc


@llm_router_v1.delete(
    "/model-endpoints/{model_endpoint_name}", response_model=DeleteLLMEndpointResponse
)
async def delete_llm_model_endpoint(
    model_endpoint_name: str,
    auth: User = Depends(verify_authentication),
    external_interfaces: ExternalInterfaces = Depends(get_external_interfaces),
) -> DeleteLLMEndpointResponse:
    logger.info(f"DELETE /model-endpoints/{model_endpoint_name} for {auth}")
    try:
        use_case = DeleteLLMEndpointByNameUseCase(
            llm_model_endpoint_service=external_interfaces.llm_model_endpoint_service,
            model_endpoint_service=external_interfaces.model_endpoint_service,
        )
        return await use_case.execute(user=auth, model_endpoint_name=model_endpoint_name)
    except ObjectNotFoundException as exc:
        raise HTTPException(
            status_code=404,
            detail="The requested model endpoint could not be found.",
        ) from exc
    except ObjectNotAuthorizedException as exc:
        raise HTTPException(
            status_code=403,
            detail="You don't have permission to delete the requested model endpoint.",
        ) from exc
    except ExistingEndpointOperationInProgressException as exc:
        raise HTTPException(
            status_code=409,
            detail="Existing operation on endpoint in progress, try again later.",
        ) from exc
    except EndpointDeleteFailedException as exc:  # pragma: no cover
        raise HTTPException(
            status_code=500,
            detail="deletion of endpoint failed.",
        ) from exc


@llm_router_v1.post("/batch-completions", response_model=CreateBatchCompletionsV1Response)
async def create_batch_completions(
    request: CreateBatchCompletionsV1Request,
    auth: User = Depends(verify_authentication),
    external_interfaces: ExternalInterfaces = Depends(get_external_interfaces),
) -> CreateBatchCompletionsV1Response:
    logger.info(f"POST /batch-completions with {request} for {auth}")
    try:
        use_case = CreateBatchCompletionsUseCase(
            docker_image_batch_job_gateway=external_interfaces.docker_image_batch_job_gateway,
            docker_repository=external_interfaces.docker_repository,
            docker_image_batch_job_bundle_repo=external_interfaces.docker_image_batch_job_bundle_repository,
            llm_artifact_gateway=external_interfaces.llm_artifact_gateway,
        )
        return await use_case.execute(user=auth, request=request)
    except (ObjectNotFoundException, ObjectNotAuthorizedException) as exc:
        raise HTTPException(
            status_code=404,
            detail="The specified endpoint could not be found.",
        ) from exc
    except (InvalidRequestException, ObjectHasInvalidValueException) as exc:
        raise HTTPException(status_code=400, detail=str(exc))
