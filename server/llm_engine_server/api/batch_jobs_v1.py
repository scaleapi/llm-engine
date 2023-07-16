from fastapi import APIRouter, Depends, HTTPException

from llm_engine_server.api.dependencies import (
    ExternalInterfaces,
    get_external_interfaces,
    get_external_interfaces_read_only,
    verify_authentication,
)
from llm_engine_server.common.datadog_utils import add_trace_resource_name
from llm_engine_server.common.dtos.batch_jobs import (
    CreateBatchJobV1Request,
    CreateBatchJobV1Response,
    CreateDockerImageBatchJobV1Request,
    CreateDockerImageBatchJobV1Response,
    GetBatchJobV1Response,
    GetDockerImageBatchJobV1Response,
    UpdateBatchJobV1Request,
    UpdateBatchJobV1Response,
    UpdateDockerImageBatchJobV1Request,
    UpdateDockerImageBatchJobV1Response,
)
from llm_engine_server.core.auth.authentication_repository import User
from llm_engine_server.core.domain_exceptions import (
    DockerImageNotFoundException,
    ObjectHasInvalidValueException,
    ObjectNotAuthorizedException,
    ObjectNotFoundException,
)
from llm_engine_server.core.loggers import filename_wo_ext, make_logger
from llm_engine_server.domain.exceptions import (
    EndpointLabelsException,
    EndpointResourceInvalidRequestException,
)
from llm_engine_server.domain.use_cases.batch_job_use_cases import (
    CreateBatchJobV1UseCase,
    CreateDockerImageBatchJobV1UseCase,
    GetBatchJobV1UseCase,
    GetDockerImageBatchJobV1UseCase,
    UpdateBatchJobV1UseCase,
    UpdateDockerImageBatchJobV1UseCase,
)

batch_job_router_v1 = APIRouter(prefix="/v1")

logger = make_logger(filename_wo_ext(__name__))


@batch_job_router_v1.post("/batch-jobs", response_model=CreateBatchJobV1Response)
async def create_batch_job(
    request: CreateBatchJobV1Request,
    auth: User = Depends(verify_authentication),
    external_interfaces: ExternalInterfaces = Depends(get_external_interfaces),
) -> CreateBatchJobV1Response:
    """
    Runs a batch job.
    """
    add_trace_resource_name("batch_jobs_post")
    logger.info(f"POST /batch-jobs with {request} for {auth}")
    try:
        use_case = CreateBatchJobV1UseCase(
            model_bundle_repository=external_interfaces.model_bundle_repository,
            model_endpoint_service=external_interfaces.model_endpoint_service,
            batch_job_service=external_interfaces.batch_job_service,
        )
        return await use_case.execute(user=auth, request=request)
    except EndpointLabelsException as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc
    except (ObjectNotFoundException, ObjectNotAuthorizedException) as exc:
        raise HTTPException(
            status_code=404,
            detail="The specified model bundle could not be found.",
        ) from exc


@batch_job_router_v1.get("/batch-jobs/{batch_job_id}", response_model=GetBatchJobV1Response)
async def get_batch_job(
    batch_job_id: str,
    auth: User = Depends(verify_authentication),
    external_interfaces: ExternalInterfaces = Depends(get_external_interfaces_read_only),
) -> GetBatchJobV1Response:
    """
    Gets a batch job.
    """
    add_trace_resource_name("batch_jobs_get")
    logger.info(f"GET /batch-jobs/{batch_job_id} for {auth}")
    try:
        use_case = GetBatchJobV1UseCase(batch_job_service=external_interfaces.batch_job_service)
        return await use_case.execute(user=auth, batch_job_id=batch_job_id)
    except (ObjectNotFoundException, ObjectNotAuthorizedException) as exc:
        raise HTTPException(
            status_code=404,
            detail="The specified batch job could not be found.",
        ) from exc


@batch_job_router_v1.put("/batch-jobs/{batch_job_id}", response_model=UpdateBatchJobV1Response)
async def update_batch_job(
    batch_job_id: str,
    request: UpdateBatchJobV1Request,
    auth: User = Depends(verify_authentication),
    external_interfaces: ExternalInterfaces = Depends(get_external_interfaces),
) -> UpdateBatchJobV1Response:
    """
    Updates a batch job.
    """
    add_trace_resource_name("batch_jobs_put")
    logger.info(f"PUT /batch-jobs/{batch_job_id} for {auth}")
    try:
        use_case = UpdateBatchJobV1UseCase(batch_job_service=external_interfaces.batch_job_service)
        return await use_case.execute(user=auth, batch_job_id=batch_job_id, request=request)
    except (ObjectNotFoundException, ObjectNotAuthorizedException) as exc:
        raise HTTPException(
            status_code=404,
            detail="The specified batch job could not be found.",
        ) from exc


@batch_job_router_v1.post(
    "/docker-image-batch-jobs", response_model=CreateDockerImageBatchJobV1Response
)
async def create_docker_image_batch_job(
    request: CreateDockerImageBatchJobV1Request,
    auth: User = Depends(verify_authentication),
    external_interfaces: ExternalInterfaces = Depends(get_external_interfaces),
) -> CreateDockerImageBatchJobV1Response:

    add_trace_resource_name("batch_jobs_di_create")
    logger.info(f"POST /docker-image-batch-jobs with {request} for {auth}")
    try:
        use_case = CreateDockerImageBatchJobV1UseCase(
            docker_image_batch_job_gateway=external_interfaces.docker_image_batch_job_gateway,
            docker_image_batch_job_bundle_repository=external_interfaces.docker_image_batch_job_bundle_repository,
            docker_repository=external_interfaces.docker_repository,
        )
        return await use_case.execute(user=auth, request=request)
    except (ObjectNotFoundException, ObjectNotAuthorizedException) as exc:
        raise HTTPException(
            status_code=404, detail="The specified batch job bundle could not be found"
        ) from exc
    except DockerImageNotFoundException as exc:
        raise HTTPException(
            status_code=404,
            detail=f"The specified docker image {exc.repository}:{exc.tag} was not found",
        )
    except ObjectHasInvalidValueException as exc:
        raise HTTPException(
            status_code=400,
            detail=f"The user specified an invalid value: {exc}",
        ) from exc
    except EndpointResourceInvalidRequestException as exc:
        raise HTTPException(
            status_code=400, detail=f"Final endpoint resources requested is invalid: {exc}"
        ) from exc
    except EndpointLabelsException as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@batch_job_router_v1.get(
    "/docker-image-batch-jobs/{batch_job_id}", response_model=GetDockerImageBatchJobV1Response
)
async def get_docker_image_batch_job(
    batch_job_id: str,
    auth: User = Depends(verify_authentication),
    external_interfaces: ExternalInterfaces = Depends(get_external_interfaces_read_only),
) -> GetDockerImageBatchJobV1Response:
    add_trace_resource_name("batch_jobs_di_get")
    logger.info(f"GET /docker-image-batch-jobs/{batch_job_id} for {auth}")
    try:
        use_case = GetDockerImageBatchJobV1UseCase(
            docker_image_batch_job_gateway=external_interfaces.docker_image_batch_job_gateway
        )
        return await use_case.execute(user=auth, batch_job_id=batch_job_id)
    except (ObjectNotFoundException, ObjectNotAuthorizedException) as exc:
        raise HTTPException(
            status_code=404, detail="The specified batch job could not be found"
        ) from exc


@batch_job_router_v1.put(
    "/docker-image-batch-jobs/{batch_job_id}", response_model=UpdateDockerImageBatchJobV1Response
)
async def update_docker_image_batch_job(
    batch_job_id: str,
    request: UpdateDockerImageBatchJobV1Request,
    auth: User = Depends(verify_authentication),
    external_interfaces: ExternalInterfaces = Depends(get_external_interfaces),
) -> UpdateDockerImageBatchJobV1Response:
    add_trace_resource_name("batch_jobs_di_put")
    logger.info(f"PUT /docker-image-batch-jobs/{batch_job_id} with {request} for {auth}")
    try:
        use_case = UpdateDockerImageBatchJobV1UseCase(
            docker_image_batch_job_gateway=external_interfaces.docker_image_batch_job_gateway
        )
        return await use_case.execute(user=auth, batch_job_id=batch_job_id, request=request)
    except (ObjectNotFoundException, ObjectNotAuthorizedException) as exc:
        raise HTTPException(
            status_code=404, detail="The specified batch job could not be found"
        ) from exc
