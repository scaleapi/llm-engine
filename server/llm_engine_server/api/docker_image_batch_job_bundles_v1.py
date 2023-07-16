from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query

from llm_engine_server.api.dependencies import (
    ExternalInterfaces,
    get_external_interfaces,
    verify_authentication,
)
from llm_engine_server.common.datadog_utils import add_trace_resource_name
from llm_engine_server.common.dtos.batch_jobs import (
    CreateDockerImageBatchJobBundleV1Request,
    CreateDockerImageBatchJobBundleV1Response,
    DockerImageBatchJobBundleV1Response,
    ListDockerImageBatchJobBundleV1Response,
)
from llm_engine_server.common.dtos.model_bundles import ModelBundleOrderBy
from llm_engine_server.core.auth.authentication_repository import User
from llm_engine_server.core.domain_exceptions import (
    ObjectNotAuthorizedException,
    ObjectNotFoundException,
)
from llm_engine_server.core.loggers import filename_wo_ext, make_logger
from llm_engine_server.domain.exceptions import EndpointResourceInvalidRequestException
from llm_engine_server.domain.use_cases.docker_image_batch_job_bundle_use_cases import (
    CreateDockerImageBatchJobBundleV1UseCase,
    GetDockerImageBatchJobBundleByIdV1UseCase,
    GetLatestDockerImageBatchJobBundleByNameV1UseCase,
    ListDockerImageBatchJobBundleV1UseCase,
)

docker_image_batch_job_bundle_router_v1 = APIRouter(prefix="/v1")

logger = make_logger(filename_wo_ext(__name__))


@docker_image_batch_job_bundle_router_v1.post(
    "/docker-image-batch-job-bundles", response_model=CreateDockerImageBatchJobBundleV1Response
)
async def create_docker_image_batch_job_bundle(
    request: CreateDockerImageBatchJobBundleV1Request,
    auth: User = Depends(verify_authentication),
    external_interfaces: ExternalInterfaces = Depends(get_external_interfaces),
):
    """
    Creates a docker iamge batch job bundle
    """
    add_trace_resource_name("docker_image_batch_job_bundle_post")
    logger.info(f"POST /docker-image-batch-job-bundles with {request} for {auth}")
    try:
        use_case = CreateDockerImageBatchJobBundleV1UseCase(
            docker_image_batch_job_bundle_repo=external_interfaces.docker_image_batch_job_bundle_repository
        )
        return await use_case.execute(user=auth, request=request)
    except EndpointResourceInvalidRequestException as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Default batch job bundle resource request is invalid: {exc}",
        )


@docker_image_batch_job_bundle_router_v1.get(
    "/docker-image-batch-job-bundles", response_model=ListDockerImageBatchJobBundleV1Response
)
async def list_docker_image_batch_job_model_bundles(
    bundle_name: Optional[str] = Query(default=None),
    order_by: Optional[ModelBundleOrderBy] = Query(default=None),
    auth: User = Depends(verify_authentication),
    external_interfaces: ExternalInterfaces = Depends(get_external_interfaces),
) -> ListDockerImageBatchJobBundleV1Response:
    """
    Lists docker image batch job bundles owned by current owner

    """
    add_trace_resource_name("docker_image_batch_job_bundle_get")
    logger.info(
        f"GET /docker-image-batch-job-bundles?bundle_name={bundle_name}&order_by={order_by} for auth"
    )
    use_case = ListDockerImageBatchJobBundleV1UseCase(
        docker_image_batch_job_bundle_repo=external_interfaces.docker_image_batch_job_bundle_repository
    )
    return await use_case.execute(user=auth, bundle_name=bundle_name, order_by=order_by)


@docker_image_batch_job_bundle_router_v1.get(
    "/docker-image-batch-job-bundles/latest", response_model=DockerImageBatchJobBundleV1Response
)
async def get_latest_docker_image_batch_job_bundle(
    bundle_name: str,
    auth: User = Depends(verify_authentication),
    external_interfaces: ExternalInterfaces = Depends(get_external_interfaces),
) -> DockerImageBatchJobBundleV1Response:
    """Gets latest Docker Image Batch Job Bundle with given name owned by the current owner"""
    add_trace_resource_name("docker_image_batch_job_bundle_latest_get")
    logger.info(f"GET /docker-image-batch-job-bundles/latest?bundle_name={bundle_name} for {auth}")
    try:
        use_case = GetLatestDockerImageBatchJobBundleByNameV1UseCase(
            docker_image_batch_job_bundle_repo=external_interfaces.docker_image_batch_job_bundle_repository
        )
        return await use_case.execute(user=auth, bundle_name=bundle_name)
    except ObjectNotFoundException as exc:
        raise HTTPException(
            status_code=404,
            detail=f"Docker Image Batch Job Bundle with name {bundle_name} was not found.",
        ) from exc


@docker_image_batch_job_bundle_router_v1.get(
    "/docker-image-batch-job-bundles/{docker_image_batch_job_bundle_id}",
    response_model=DockerImageBatchJobBundleV1Response,
)
async def get_docker_image_batch_job_model_bundle(
    docker_image_batch_job_bundle_id: str,
    auth: User = Depends(verify_authentication),
    external_interfaces: ExternalInterfaces = Depends(get_external_interfaces),
) -> DockerImageBatchJobBundleV1Response:
    """Get details for a given DockerImageBatchJobBundle owned by the current owner"""
    add_trace_resource_name("docker_image_batch_job_bundle_id_get")
    logger.info(
        f"GET /docker-image-batch-job-bundles/{docker_image_batch_job_bundle_id} for {auth}"
    )
    try:
        use_case = GetDockerImageBatchJobBundleByIdV1UseCase(
            docker_image_batch_job_bundle_repo=external_interfaces.docker_image_batch_job_bundle_repository
        )
        return await use_case.execute(
            user=auth, docker_image_batch_job_bundle_id=docker_image_batch_job_bundle_id
        )
    except (ObjectNotFoundException, ObjectNotAuthorizedException) as exc:
        raise HTTPException(
            status_code=404,
            detail=f"Docker Image Batch Job Bundle {docker_image_batch_job_bundle_id} was not found.",
        ) from exc
