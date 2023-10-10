"""LLM Model Endpoint routes for the hosted model inference service.
"""
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from model_engine_server.api.dependencies import (
    ExternalInterfaces,
    get_external_interfaces,
    get_external_interfaces_read_only,
    verify_authentication,
)
from model_engine_server.common.datadog_utils import add_trace_resource_name
from model_engine_server.common.dtos.llms import (
    CancelFineTuneResponse,
    CompletionStreamV1Request,
    CompletionStreamV1Response,
    CompletionSyncV1Request,
    CompletionSyncV1Response,
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
)
from model_engine_server.common.dtos.model_endpoints import ModelEndpointOrderBy
from model_engine_server.core.auth.authentication_repository import User
from model_engine_server.core.loggers import filename_wo_ext, get_request_id, make_logger
from model_engine_server.domain.exceptions import (
    EndpointDeleteFailedException,
    EndpointLabelsException,
    EndpointResourceInvalidRequestException,
    EndpointUnsupportedInferenceTypeException,
    ExistingEndpointOperationInProgressException,
    InvalidRequestException,
    LLMFineTuningMethodNotImplementedException,
    LLMFineTuningQuotaReached,
    ObjectAlreadyExistsException,
    ObjectHasInvalidValueException,
    ObjectNotApprovedException,
    ObjectNotAuthorizedException,
    ObjectNotFoundException,
    UpstreamServiceError,
)
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
    CreateLLMModelEndpointV1UseCase,
    DeleteLLMEndpointByNameUseCase,
    GetLLMModelEndpointByNameV1UseCase,
    ListLLMModelEndpointsV1UseCase,
    ModelDownloadV1UseCase,
)
from model_engine_server.domain.use_cases.model_bundle_use_cases import CreateModelBundleV2UseCase
from sse_starlette.sse import EventSourceResponse

llm_router_v1 = APIRouter(prefix="/v1/llm")
logger = make_logger(filename_wo_ext(__name__))


@llm_router_v1.post("/model-endpoints", response_model=CreateLLMModelEndpointV1Response)
async def create_model_endpoint(
    request: CreateLLMModelEndpointV1Request,
    auth: User = Depends(verify_authentication),
    external_interfaces: ExternalInterfaces = Depends(get_external_interfaces),
) -> CreateLLMModelEndpointV1Response:
    """
    Creates an LLM endpoint for the current user.
    """
    add_trace_resource_name("llm_model_endpoints_post")
    logger.info(f"POST /llm/model-endpoints with {request} for {auth}")
    try:
        create_model_bundle_use_case = CreateModelBundleV2UseCase(
            model_bundle_repository=external_interfaces.model_bundle_repository,
            docker_repository=external_interfaces.docker_repository,
            model_primitive_gateway=external_interfaces.model_primitive_gateway,
        )
        use_case = CreateLLMModelEndpointV1UseCase(
            create_model_bundle_use_case=create_model_bundle_use_case,
            model_bundle_repository=external_interfaces.model_bundle_repository,
            model_endpoint_service=external_interfaces.model_endpoint_service,
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
    except ObjectNotApprovedException as exc:
        raise HTTPException(
            status_code=403,
            detail="The specified model bundle was not approved yet.",
        ) from exc
    except (ObjectNotFoundException, ObjectNotAuthorizedException) as exc:
        raise HTTPException(
            status_code=404,
            detail="The specified model bundle could not be found.",
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
    add_trace_resource_name("llm_model_endpoints_get")
    logger.info(f"GET /llm/model-endpoints?name={name}&order_by={order_by} for {auth}")
    use_case = ListLLMModelEndpointsV1UseCase(
        llm_model_endpoint_service=external_interfaces.llm_model_endpoint_service,
    )
    return await use_case.execute(user=auth, name=name, order_by=order_by)


@llm_router_v1.get(
    "/model-endpoints/{model_endpoint_name}", response_model=GetLLMModelEndpointV1Response
)
async def get_model_endpoint(
    model_endpoint_name: str,
    auth: User = Depends(verify_authentication),
    external_interfaces: ExternalInterfaces = Depends(get_external_interfaces_read_only),
) -> GetLLMModelEndpointV1Response:
    """
    Describe the LLM Model endpoint with given name.
    """
    add_trace_resource_name("llm_model_endpoints_name_get")
    logger.info(f"GET /llm/model-endpoints/{model_endpoint_name} for {auth}")
    try:
        use_case = GetLLMModelEndpointByNameV1UseCase(
            llm_model_endpoint_service=external_interfaces.llm_model_endpoint_service
        )
        return await use_case.execute(user=auth, model_endpoint_name=model_endpoint_name)
    except (ObjectNotFoundException, ObjectNotAuthorizedException) as exc:
        raise HTTPException(
            status_code=404,
            detail=f"Model Endpoint {model_endpoint_name}  was not found.",
        ) from exc


@llm_router_v1.post("/completions-sync", response_model=CompletionSyncV1Response)
async def create_completion_sync_task(
    model_endpoint_name: str,
    request: CompletionSyncV1Request,
    auth: User = Depends(verify_authentication),
    external_interfaces: ExternalInterfaces = Depends(get_external_interfaces_read_only),
) -> CompletionSyncV1Response:
    """
    Runs a sync prompt completion on an LLM.
    """
    add_trace_resource_name("llm_completion_sync_post")
    logger.info(
        f"POST /completion_sync with {request} to endpoint {model_endpoint_name} for {auth}"
    )
    try:
        use_case = CompletionSyncV1UseCase(
            model_endpoint_service=external_interfaces.model_endpoint_service,
            llm_model_endpoint_service=external_interfaces.llm_model_endpoint_service,
        )
        return await use_case.execute(
            user=auth, model_endpoint_name=model_endpoint_name, request=request
        )
    except UpstreamServiceError:
        request_id = get_request_id()
        logger.exception(f"Upstream service error for request {request_id}")
        raise HTTPException(
            status_code=500,
            detail=f"Upstream service error for request_id {request_id}.",
        )
    except (ObjectNotFoundException, ObjectNotAuthorizedException) as exc:
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
    auth: User = Depends(verify_authentication),
    external_interfaces: ExternalInterfaces = Depends(get_external_interfaces_read_only),
) -> EventSourceResponse:
    """
    Runs a stream prompt completion on an LLM.
    """
    add_trace_resource_name("llm_completion_stream_post")
    logger.info(
        f"POST /completion_stream with {request} to endpoint {model_endpoint_name} for {auth}"
    )
    use_case = CompletionStreamV1UseCase(
        model_endpoint_service=external_interfaces.model_endpoint_service,
        llm_model_endpoint_service=external_interfaces.llm_model_endpoint_service,
    )
    response = use_case.execute(user=auth, model_endpoint_name=model_endpoint_name, request=request)

    async def event_generator():
        try:
            async for message in response:
                yield {"data": message.json()}
        except InvalidRequestException as exc:
            yield {"data": {"error": {"status_code": 400, "detail": str(exc)}}}
        except UpstreamServiceError as exc:
            request_id = get_request_id()
            logger.exception(f"Upstream service error for request {request_id}")
            yield {"data": {"error": {"status_code": 500, "detail": str(exc)}}}
        except (ObjectNotFoundException, ObjectNotAuthorizedException) as exc:
            yield {"data": {"error": {"status_code": 404, "detail": str(exc)}}}
        except ObjectHasInvalidValueException as exc:
            yield {"data": {"error": {"status_code": 400, "detail": str(exc)}}}
        except EndpointUnsupportedInferenceTypeException as exc:
            yield {
                "data": {
                    "error": {
                        "status_code": 400,
                        "detail": f"Unsupported inference type: {str(exc)}",
                    }
                }
            }

    return EventSourceResponse(event_generator())


@llm_router_v1.post("/fine-tunes", response_model=CreateFineTuneResponse)
async def create_fine_tune(
    request: CreateFineTuneRequest,
    auth: User = Depends(verify_authentication),
    external_interfaces: ExternalInterfaces = Depends(get_external_interfaces),
) -> CreateFineTuneResponse:
    add_trace_resource_name("fine_tunes_create")
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
    add_trace_resource_name("fine_tunes_get")
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
    add_trace_resource_name("fine_tunes_list")
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
    add_trace_resource_name("fine_tunes_cancel")
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
    add_trace_resource_name("fine_tunes_events_get")
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
    add_trace_resource_name("model_endpoints_download")
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
    add_trace_resource_name("llm_model_endpoints_delete")
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
