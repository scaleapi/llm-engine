import hashlib
import json
import os
import tempfile
from contextlib import AsyncExitStack
from logging import LoggerAdapter
from typing import List, Optional, Sequence

from datadog import statsd
from llm_engine_server.common.constants import (
    FEATURE_FLAG_USE_MULTI_CONTAINER_ARCHITECTURE_FOR_ARTIFACTLIKE_BUNDLE,
)
from llm_engine_server.common.dtos.docker_repository import BuildImageRequest, BuildImageResponse
from llm_engine_server.common.dtos.endpoint_builder import (
    BuildEndpointRequest,
    BuildEndpointResponse,
    BuildEndpointStatus,
)
from llm_engine_server.common.dtos.resource_manager import CreateOrUpdateResourcesRequest
from llm_engine_server.common.env_vars import LOCAL
from llm_engine_server.common.io import open_wrapper
from llm_engine_server.common.serialization_utils import bool_to_str
from llm_engine_server.core.config import ml_infra_config
from llm_engine_server.core.domain_exceptions import DockerBuildFailedException
from llm_engine_server.core.loggers import make_logger
from llm_engine_server.core.notification_gateway import NotificationApp, NotificationGateway
from llm_engine_server.core.utils.env import environment
from llm_engine_server.domain.entities import (
    ArtifactLike,
    CloudpickleArtifactFlavor,
    CustomFramework,
    ModelBundleFlavorType,
    ModelEndpointConfig,
    ModelEndpointDeploymentState,
    ModelEndpointInfraState,
    ModelEndpointResourceState,
    ModelEndpointStatus,
    ModelEndpointUserConfigState,
    PytorchFramework,
    RunnableImageFlavor,
    RunnableImageLike,
    TensorflowFramework,
    ZipArtifactFlavor,
)
from llm_engine_server.domain.exceptions import EndpointResourceInfraException
from llm_engine_server.domain.gateways import MonitoringMetricsGateway
from llm_engine_server.domain.repositories import DockerRepository
from llm_engine_server.domain.services import EndpointBuilderService
from llm_engine_server.domain.use_cases.model_endpoint_use_cases import (
    CONVERTED_FROM_ARTIFACT_LIKE_KEY,
)
from llm_engine_server.infra.gateways import FilesystemGateway
from llm_engine_server.infra.gateways.resources.endpoint_resource_gateway import (
    EndpointResourceGateway,
)
from llm_engine_server.infra.infra_utils import make_exception_log
from llm_engine_server.infra.repositories import FeatureFlagRepository, ModelEndpointCacheRepository
from llm_engine_server.infra.repositories.model_endpoint_record_repository import (
    ModelEndpointRecordRepository,
)

if LOCAL:
    with environment(KUBERNETES_SERVICE_HOST=None):
        logger = make_logger("llm_engine_server.service_builder")
else:
    logger = make_logger("llm_engine_server.service_builder")

__all__: Sequence[str] = (
    "INITIAL_K8S_CACHE_TTL_SECONDS",
    "LiveEndpointBuilderService",
)

# We check that the env vars aren't None right after
ECR_AWS_PROFILE: str = os.getenv("ECR_READ_AWS_PROFILE", "default")  # type: ignore
GIT_TAG: str = os.getenv("GIT_TAG")  # type: ignore
ENV: str = os.getenv("DD_ENV")  # type: ignore

INITIAL_K8S_CACHE_TTL_SECONDS: int = 60
MAX_IMAGE_TAG_LEN = 128


class LiveEndpointBuilderService(EndpointBuilderService):
    def __init__(
        self,
        docker_repository: DockerRepository,
        resource_gateway: EndpointResourceGateway,
        monitoring_metrics_gateway: MonitoringMetricsGateway,
        model_endpoint_record_repository: ModelEndpointRecordRepository,
        model_endpoint_cache_repository: ModelEndpointCacheRepository,
        filesystem_gateway: FilesystemGateway,
        notification_gateway: NotificationGateway,
        feature_flag_repo: FeatureFlagRepository,
    ) -> None:
        self.docker_repository = docker_repository
        self.resource_gateway = resource_gateway
        self.monitoring_metrics_gateway = monitoring_metrics_gateway
        self.model_endpoint_record_repository = model_endpoint_record_repository
        self.model_endpoint_cache_repository = model_endpoint_cache_repository
        self.filesystem_gateway = filesystem_gateway
        self.notification_gateway = notification_gateway
        self.feature_flag_repo = feature_flag_repo

    async def build_endpoint(
        self, build_endpoint_request: BuildEndpointRequest
    ) -> BuildEndpointResponse:
        self.monitoring_metrics_gateway.emit_attempted_build_metric()

        logger_extra = build_endpoint_request.dict()
        logger_adapter = LoggerAdapter(logger, extra=logger_extra)
        log_error = make_exception_log(logger_adapter)

        model_endpoint_record = build_endpoint_request.model_endpoint_record
        endpoint_id = model_endpoint_record.id
        model_bundle = model_endpoint_record.current_model_bundle
        logger_adapter.info(
            f"Building service for endpoint_name: {model_endpoint_record.name}",
        )

        self._validate_build_endpoint_request(build_endpoint_request)

        use_multi_container_architecture_for_artifactlike_bundle = (
            await self.feature_flag_repo.read_feature_flag_bool(
                FEATURE_FLAG_USE_MULTI_CONTAINER_ARCHITECTURE_FOR_ARTIFACTLIKE_BUNDLE
            )
        )

        async with AsyncExitStack() as stack:
            lock_ctx = self.model_endpoint_record_repository.get_lock_context(model_endpoint_record)
            lock = await stack.enter_async_context(lock_ctx)
            # If this can't acquire the lock by the timeout it'll happily keep on going and create
            # the requisite resources. Not sure this makes complete sense?
            logger_adapter.info(f"Acquiring lock on endpoint {endpoint_id}")
            if not lock.lock_acquired():  # pragma: no cover
                logger_adapter.warning(
                    "Lock not acquired, still making k8s changes. This may lead to orphaned "
                    "resources."
                )

            try:
                # First, build the image if the model bundle does not have a docker image
                if not model_bundle.is_runnable():
                    if use_multi_container_architecture_for_artifactlike_bundle:
                        assert isinstance(
                            model_bundle.flavor, CloudpickleArtifactFlavor
                        ) or isinstance(model_bundle.flavor, ZipArtifactFlavor)
                        logger_adapter.info(
                            f"Create a new runnable image model bundle for artifact flavor model bundle {model_bundle.id=} ..."
                        )
                    logger_adapter.info("Building base & user image...")
                    # Build service image in two steps for better caching.
                    # First we build a base image, which is expected to be shared between
                    # many different bundles.
                    try:
                        base_image_params = self._get_base_image_params(
                            build_endpoint_request, logger_adapter
                        )
                        base_image = await self._build_image(
                            base_image_params, build_endpoint_request, logger_adapter
                        )
                        user_image_params = self._get_user_image_params(
                            base_image, build_endpoint_request, logger_adapter
                        )
                        image = await self._build_image(
                            user_image_params, build_endpoint_request, logger_adapter
                        )

                        image_repo = user_image_params.repo
                        image_tag = user_image_params.image_tag

                        # Add third stage to inject bundle into image if necessary (only for high priority endpoints at the moment)
                        if (
                            isinstance(
                                build_endpoint_request.model_endpoint_record.current_model_bundle.flavor,
                                ZipArtifactFlavor,
                            )
                            and build_endpoint_request.high_priority
                        ):
                            inject_bundle_image_params = self._get_inject_bundle_image_params(
                                image,
                                user_image_params,
                                build_endpoint_request,
                                logger_adapter,
                            )

                            image_repo = inject_bundle_image_params.repo
                            image_tag = inject_bundle_image_params.image_tag

                            image = await self._build_image(
                                inject_bundle_image_params,
                                build_endpoint_request,
                                logger_adapter,
                            )

                            # Now that it's no longer needed, clean up serialized bundle file to save storage
                            model_bundle_path = inject_bundle_image_params.substitution_args[  # type: ignore
                                "LOCAL_BUNDLE_PATH"
                            ]
                            if os.path.exists(model_bundle_path):
                                os.remove(model_bundle_path)
                            else:
                                logger.error(f"No bundle object found at {model_bundle_path}!")

                    except DockerBuildFailedException:
                        log_error("Failed to build base and user docker images")
                        self.monitoring_metrics_gateway.emit_docker_failed_build_metric()
                        raise

                    if use_multi_container_architecture_for_artifactlike_bundle:
                        self.convert_artifact_like_bundle_to_runnable_image(
                            build_endpoint_request, image_repo, image_tag
                        )

                        # CONVERTED_FROM_ARTIFACT_LIKE_KEY will be checked by `get_endpoint_resource_arguments_from_request()` in k8s_resource_types.py
                        if not model_endpoint_record.metadata:
                            model_endpoint_record.metadata = {}
                        model_endpoint_record.metadata.update(
                            {CONVERTED_FROM_ARTIFACT_LIKE_KEY: True}
                        )
                        await self.model_endpoint_record_repository.update_model_endpoint_record(
                            model_endpoint_id=endpoint_id,
                            metadata=model_endpoint_record.metadata,
                        )

                else:
                    flavor = model_bundle.flavor
                    assert isinstance(flavor, RunnableImageLike)
                    repository = (
                        f"{ml_infra_config().docker_repo_prefix}/{flavor.repository}"
                        if self.docker_repository.is_repo_name(flavor.repository)
                        else flavor.repository
                    )
                    image = f"{repository}:{flavor.tag}"

                # Because this update is not the final update in the lock, the 'update_in_progress'
                # value isn't really necessary for correctness in not having races, but it's still
                # informative at the least
                await self.model_endpoint_record_repository.update_model_endpoint_record(
                    model_endpoint_id=endpoint_id,
                    status=ModelEndpointStatus.UPDATE_IN_PROGRESS,
                )

                try:
                    params = CreateOrUpdateResourcesRequest(
                        build_endpoint_request=build_endpoint_request,
                        image=image,
                    )
                    create_or_update_response = (
                        await self.resource_gateway.create_or_update_resources(params)
                    )

                except EndpointResourceInfraException:
                    log_error("K8s resource update failed")
                    raise

                endpoint_info = ModelEndpointInfraState(
                    deployment_name=build_endpoint_request.deployment_name,
                    aws_role=build_endpoint_request.aws_role,
                    results_s3_bucket=build_endpoint_request.results_s3_bucket,
                    child_fn_info=build_endpoint_request.child_fn_info,
                    labels=build_endpoint_request.labels,
                    deployment_state=ModelEndpointDeploymentState(
                        min_workers=build_endpoint_request.min_workers,
                        max_workers=build_endpoint_request.max_workers,
                        per_worker=build_endpoint_request.per_worker,
                    ),
                    resource_state=ModelEndpointResourceState(
                        cpus=build_endpoint_request.cpus,
                        gpus=build_endpoint_request.gpus,
                        memory=build_endpoint_request.memory,
                        gpu_type=build_endpoint_request.gpu_type,
                        storage=build_endpoint_request.storage,
                        optimize_costs=build_endpoint_request.optimize_costs,
                    ),
                    user_config_state=ModelEndpointUserConfigState(
                        app_config=build_endpoint_request.model_endpoint_record.current_model_bundle.app_config,
                        endpoint_config=ModelEndpointConfig(
                            endpoint_name=build_endpoint_request.model_endpoint_record.name,
                            bundle_name=build_endpoint_request.model_endpoint_record.current_model_bundle.name,
                            post_inference_hooks=build_endpoint_request.post_inference_hooks,
                            default_callback_url=build_endpoint_request.default_callback_url,
                            default_callback_auth=build_endpoint_request.default_callback_auth,
                        ),
                    ),
                    prewarm=build_endpoint_request.prewarm,
                    high_priority=build_endpoint_request.high_priority,
                    num_queued_items=None,
                    image=image,
                )

                endpoint_config = endpoint_info.user_config_state.endpoint_config
                updated_endpoint_name: Optional[str] = (
                    endpoint_config.endpoint_name if endpoint_config is not None else None
                )
                logger_adapter.info(
                    f"Created {endpoint_id=}: "
                    f"deployment_name={build_endpoint_request.deployment_name} "
                    f"endpoint_name={updated_endpoint_name}"
                )

                # Write to cache immediately to make sure an entry actually exists
                await self.model_endpoint_cache_repository.write_endpoint_info(
                    endpoint_id=endpoint_id,
                    endpoint_info=endpoint_info,
                    ttl_seconds=INITIAL_K8S_CACHE_TTL_SECONDS,
                )

                await self.model_endpoint_record_repository.update_model_endpoint_record(
                    model_endpoint_id=endpoint_id,
                    destination=create_or_update_response.destination,
                    status=ModelEndpointStatus.READY,
                )

            except Exception as error:  # noqa
                log_error("Failed endpoint build process!")
                # Update status as failed endpoint creation on unhandled error
                try:
                    await self.model_endpoint_record_repository.update_model_endpoint_record(
                        model_endpoint_id=endpoint_id,
                        status=ModelEndpointStatus.UPDATE_FAILED,
                    )
                except Exception as error_update:
                    log_error("Failed to update endpoint build status to FAILED")
                    raise error_update from error
                raise error

            finally:
                logger_adapter.info(
                    f"Releasing lock on endpoint {model_endpoint_record.name}, user "
                    f"{model_endpoint_record.created_by}"
                )

        try:
            self.monitoring_metrics_gateway.emit_successful_build_metric()
        except Exception:  # noqa
            log_error(f"[Continuing] Failed to emit successful build metric for {endpoint_id=}")

        return BuildEndpointResponse(status=BuildEndpointStatus.OK)

    def convert_artifact_like_bundle_to_runnable_image(
        self,
        build_endpoint_request: BuildEndpointRequest,
        image_repo: str,
        image_tag: str,
    ) -> None:
        """
        With LLMEngine Inference Re-Architecture, we want to deploy endpoints with ArtifactLike bundle using
        multi-container architecture, which RunnableImageFlavor has already adopted.

        This function mutates the build_endpoint_request by converting the ArtifactLike bundle flavor into
        a RunnableImageFlavor on the fly so that K8SEndpointResourceDelegate will use the deployment template
        for RunnableImageFlavor to deploy the endpoint.

        Note that the converted model bundle flavor is not persisted in the database. So users will not be aware of
        this conversion.
        """
        model_bundle = build_endpoint_request.model_endpoint_record.current_model_bundle
        assert isinstance(model_bundle.flavor, ArtifactLike)
        new_model_bundle = model_bundle.copy()

        if ml_infra_config().env == "circleci":
            ml_infra_service_config_file = "config.yaml"
        else:
            ml_infra_service_config_file = ml_infra_config().env + ".yaml"

        new_flavor = RunnableImageFlavor(
            flavor=ModelBundleFlavorType.RUNNABLE_IMAGE,
            repository=image_repo,
            tag=image_tag,
            command=[
                "dumb-init",
                "--",
                "ddtrace-run",
                "python",
                "-m",
                "llm_engine_server.inference.sync_inference.start_fastapi_server",
            ],
            env={
                "OMP_NUM_THREADS": '"1"',
                "BASE_PATH": "/app",
                "BUNDLE_URL": model_bundle.flavor.location,
                "AWS_PROFILE": build_endpoint_request.aws_role,
                "RESULTS_S3_BUCKET": ml_infra_config().s3_bucket,
                "CHILD_FN_INFO": json.dumps(
                    build_endpoint_request.child_fn_info
                    if build_endpoint_request.child_fn_info
                    else {}
                ),
                "PREWARM": bool_to_str(build_endpoint_request.prewarm) or "false",
                "PORT": "5005",
                "ML_INFRA_SERVICES_CONFIG_PATH": f"/app/ml_infra_core/llm_engine_server.core/llm_engine_server.core/configs/{ml_infra_service_config_file}",
            },
            protocol="http",
        )

        if isinstance(model_bundle.flavor, ZipArtifactFlavor):
            if new_flavor.env is None:
                new_flavor.env = {}
            new_flavor.env[
                "LOAD_PREDICT_FN_MODULE_PATH"
            ] = model_bundle.flavor.load_predict_fn_module_path
            new_flavor.env[
                "LOAD_MODEL_FN_MODULE_PATH"
            ] = model_bundle.flavor.load_model_fn_module_path

        new_model_bundle.flavor = new_flavor
        new_model_bundle.model_artifact_ids = []

        build_endpoint_request.model_endpoint_record.current_model_bundle = new_model_bundle

    def _get_base_image_params(
        self,
        build_endpoint_request: BuildEndpointRequest,
        logger_adapter: LoggerAdapter,
    ) -> BuildImageRequest:
        model_endpoint_record = build_endpoint_request.model_endpoint_record
        model_bundle = model_endpoint_record.current_model_bundle

        assert isinstance(model_bundle.flavor, ArtifactLike)
        env_params = model_bundle.flavor.framework

        # Determine dockerfile/ecr repo to use
        if isinstance(env_params, PytorchFramework):
            image_tag = env_params.pytorch_image_tag
            if image_tag is None:  # pragma: no cover
                raise ValueError("Pytorch image tag must be specified if the framework is Pytorch.")
            logger_adapter.info(f"Using pytorch image tag: {image_tag}")
            dockerfile = "pytorch_or_tf.base.Dockerfile"
            base_image = f"pytorch/pytorch:{image_tag}"
            resulting_image_tag = f"pytorch-{image_tag}-{GIT_TAG}"
        elif isinstance(env_params, TensorflowFramework):
            if build_endpoint_request.gpus > 0:
                raise NotImplementedError("Tensorflow GPU image not supported yet")
            # NOTE: The base image is not a Tensorflow image, but a miniconda one. This is
            # because the base Tensorflow image has some weirdness with its python installation.
            # We may change this for Tensorflow GPU mages.
            tensorflow_version = env_params.tensorflow_version
            if tensorflow_version is None:  # pragma: no cover
                raise ValueError("Tensorflow version must be specified if the framework is TF.")
            logger_adapter.info(f"Using tensorflow version: {tensorflow_version}")
            dockerfile = "pytorch_or_tf.base.Dockerfile"
            base_image = "continuumio/miniconda3:4.9.2"
            resulting_image_tag = f"tensorflow-{GIT_TAG}"
        elif isinstance(env_params, CustomFramework):
            if env_params.image_tag is None or env_params.image_repository is None:
                raise ValueError("Base image tag and ECR repo must be specified for custom images.")
            base_image_tag = env_params.image_tag
            ecr_repo = env_params.image_repository
            logger_adapter.info(f"Using ECR base image tag: {base_image_tag} in repo: {ecr_repo}")
            dockerfile = "base.Dockerfile"
            base_image = self.docker_repository.get_image_url(base_image_tag, ecr_repo)
            resulting_image_tag = "-".join([ecr_repo, base_image_tag, GIT_TAG]).replace("/", "-")
        else:  # pragma: no cover
            raise ValueError(f"Unsupported framework_type: {env_params.framework_type}")

        # The context should be whatever WORKDIR is in the container running the build app itself.
        inference_folder = "llm_engine/llm_engine/inference"
        base_path: str = os.getenv("WORKSPACE")  # type: ignore

        return BuildImageRequest(
            repo="llm-engine",
            image_tag=resulting_image_tag[:MAX_IMAGE_TAG_LEN],
            aws_profile=ECR_AWS_PROFILE,  # type: ignore
            base_path=base_path,
            dockerfile=f"{inference_folder}/{dockerfile}",
            base_image=base_image,
            requirements_folder=None,
            substitution_args=None,
        )

    def _get_user_image_params(
        self,
        base_image: str,
        build_endpoint_request: BuildEndpointRequest,
        logger_adapter: LoggerAdapter,
    ) -> BuildImageRequest:
        model_endpoint_record = build_endpoint_request.model_endpoint_record
        model_bundle = model_endpoint_record.current_model_bundle

        assert isinstance(model_bundle.flavor, ArtifactLike)
        env_params = model_bundle.flavor.framework
        requirements_hash = self._get_requirements_hash(model_bundle.requirements or [])

        # Determine dockerfile/ecr repo to use
        if isinstance(env_params, PytorchFramework):
            base_image_tag = env_params.pytorch_image_tag
            if base_image_tag is None:  # pragma: no cover
                raise ValueError("Pytorch image tag must be specified if the framework is Pytorch.")

            dockerfile = "pytorch_or_tf.user.Dockerfile"
            service_image_tag = self._get_image_tag(base_image_tag, GIT_TAG, requirements_hash)
            ecr_repo = "hosted-model-inference/async-pytorch"
        elif isinstance(env_params, TensorflowFramework):
            if build_endpoint_request.gpus > 0:
                raise NotImplementedError("Tensorflow GPU image not supported yet")
            # NOTE: The base image is not a Tensorflow image, but a miniconda one. This is
            # because the base Tensorflow image has some weirdness with its python installation.
            # We may change this for Tensorflow GPU mages.
            tensorflow_version = env_params.tensorflow_version
            if tensorflow_version is None:  # pragma: no cover
                raise ValueError("Tensorflow version must be specified if the framework is TF.")
            dockerfile = "pytorch_or_tf.user.Dockerfile"
            service_image_tag = self._get_image_tag(tensorflow_version, GIT_TAG, requirements_hash)
            ecr_repo = "hosted-model-inference/async-tensorflow-cpu"
        elif isinstance(env_params, CustomFramework):
            if (
                env_params.image_tag is None or env_params.image_repository is None
            ):  # pragma: no cover
                raise ValueError("Base image tag and ECR repo must be specified for custom images.")
            base_image_tag = env_params.image_tag
            dockerfile = "user.Dockerfile"
            service_image_tag = self._get_image_tag(base_image_tag, GIT_TAG, requirements_hash)
            ecr_repo = env_params.image_repository
        else:  # pragma: no cover
            raise ValueError(f"Unsupported framework_type: {env_params.framework_type}")

        # The context should be whatever WORKDIR is in the container running the build app itself.
        inference_folder = "llm_engine/llm_engine/inference"
        base_path: str = os.getenv("WORKSPACE")  # type: ignore

        requirements_folder = os.path.join(base_path, f"requirements_{requirements_hash}")
        try:
            os.mkdir(requirements_folder)
        except FileExistsError:
            pass

        requirements_file = os.path.join(requirements_folder, "requirements.txt")
        with open(requirements_file, "w") as f:
            requirements_contents = "\n".join(model_bundle.requirements or [])
            logger_adapter.info(f"Will pip install these requirements: {requirements_contents}")
            f.write(requirements_contents)

        substitution_args = {"REQUIREMENTS_FILE": requirements_file}

        return BuildImageRequest(
            repo=ecr_repo,
            image_tag=service_image_tag[:MAX_IMAGE_TAG_LEN],
            aws_profile=ECR_AWS_PROFILE,
            base_path=base_path,
            dockerfile=f"{inference_folder}/{dockerfile}",
            base_image=base_image,
            requirements_folder=requirements_folder,
            substitution_args=substitution_args,
        )

    def _get_inject_bundle_image_params(
        self,
        base_image: str,
        base_image_params: BuildImageRequest,
        build_endpoint_request: BuildEndpointRequest,
        logger_adapter: LoggerAdapter,
    ) -> BuildImageRequest:
        model_endpoint_record = build_endpoint_request.model_endpoint_record
        model_bundle = model_endpoint_record.current_model_bundle

        bundle_id = model_bundle.id
        service_image_str = "-".join([base_image_params.image_tag, GIT_TAG, bundle_id])
        service_image_hash = hashlib.md5(str(service_image_str).encode("utf-8")).hexdigest()
        service_image_tag = f"inject-bundle-image-{service_image_hash}"
        ecr_repo = base_image_params.repo

        logger_adapter.info(
            f"Running step to inject bundle with repo {ecr_repo} and tag {service_image_tag}"
        )

        # The context should be whatever WORKDIR is in the container running the build app itself.
        dockerfile = "inject_bundle.Dockerfile"
        inference_folder = "llm_engine/llm_engine/inference"
        base_path: str = os.getenv("WORKSPACE")  # type: ignore

        bundle_folder = os.path.join(base_path, f"bundle_{service_image_hash}")
        try:
            os.mkdir(bundle_folder)
        except FileExistsError:
            pass
        _, model_bundle_path = tempfile.mkstemp(dir=bundle_folder, suffix=".zip")
        bundle_url = model_bundle.location
        logger.info(
            f"Downloading bundle from serialized object at location {bundle_url} to local path {model_bundle_path}"
        )
        with open_wrapper(bundle_url, "rb") as bundle_data:  # type: ignore
            with open(model_bundle_path, "wb") as local_bundle_file:
                local_bundle_file.write(bundle_data.read())

        substitution_args = {
            "LOCAL_BUNDLE_PATH": model_bundle_path,
            "LOAD_MODEL_MODULE_PATH": model_bundle.flavor.load_model_fn_module_path,  # type: ignore
            "LOAD_PREDICT_MODULE_PATH": model_bundle.flavor.load_predict_fn_module_path,  # type: ignore
        }

        return BuildImageRequest(
            repo=ecr_repo,
            image_tag=service_image_tag[:MAX_IMAGE_TAG_LEN],
            aws_profile=ECR_AWS_PROFILE,
            base_path=base_path,
            dockerfile=f"{inference_folder}/{dockerfile}",
            base_image=base_image,
            requirements_folder=bundle_folder,
            substitution_args=substitution_args,
        )

    async def _build_image(
        self,
        image_params: BuildImageRequest,
        build_endpoint_request: BuildEndpointRequest,
        logger_adapter: LoggerAdapter,
    ) -> str:
        """
        Builds the service image and updates the endpoint status if the image building fails.

        Returns:
            Image url: if image build is successful or if it already exists.
        Raises:
            DockerBuildFailedException: If image build is required and building it fails.
        """
        model_endpoint_record = build_endpoint_request.model_endpoint_record
        model_endpoint_name = model_endpoint_record.name
        user_id = model_endpoint_record.created_by
        endpoint_id = build_endpoint_request.model_endpoint_record.id

        log_error = make_exception_log(logger_adapter)

        # image_exists hardcodes the ML ECR account, which needs to change for external self-hosted
        if not self.docker_repository.image_exists(
            repository_name=image_params.repo,
            image_tag=image_params.image_tag,
            aws_profile=ECR_AWS_PROFILE,
        ):
            tags = [
                f"kube_deployment:{build_endpoint_request.deployment_name}",
                f"user_id:{user_id}",
            ]
            with statsd.timed("kaniko.build_time", tags=tags):
                try:
                    build_result: BuildImageResponse = self.docker_repository.build_image(
                        image_params,
                    )
                    build_result_status = build_result.status
                    build_result_logs: str = build_result.logs
                except Exception:  # noqa
                    build_result_status = False
                    s3_logs_location: Optional[str] = None
                    log_error(
                        "Unknown error encountered on image build"
                        f"No logs to write for {model_endpoint_name}, since docker build threw exception"
                    )
                else:
                    # Write builder logs into a remote file
                    s3_logs_location = self._get_service_builder_logs_location(
                        user_id=user_id,
                        endpoint_name=model_endpoint_name,
                    )
                    try:
                        with self.filesystem_gateway.open(
                            s3_logs_location,
                            "w",
                            aws_profile=ml_infra_config().profile_ml_worker,
                        ) as file_out:
                            file_out.write(build_result_logs)
                    except Exception:  # noqa
                        log_error(
                            f"Unable to publish service builder logs for {model_endpoint_name}"
                        )

                # Check that the build didn't succeed and the image doesn't exist.
                # There's a race condition where if another simultaneous build is started,
                # then one of the builds will not succeed, but the docker image ends up built.
                if not build_result_status and not self.docker_repository.image_exists(
                    repository_name=image_params.repo,
                    image_tag=image_params.image_tag,
                    aws_profile=ECR_AWS_PROFILE,
                ):
                    log_error(
                        f"Image build failed for endpoint {model_endpoint_name}, user {user_id}"
                    )

                    await self.model_endpoint_record_repository.update_model_endpoint_record(
                        model_endpoint_id=build_endpoint_request.model_endpoint_record.id,
                        status=ModelEndpointStatus.UPDATE_FAILED,
                    )

                    if s3_logs_location is not None:
                        help_url = self.filesystem_gateway.generate_signed_url(
                            s3_logs_location,
                            expiration=43200,  # 12 hours
                            aws_profile=ml_infra_config().profile_ml_worker,
                        )
                    else:
                        help_url = (
                            "https://app.datadoghq.com/logs?query=service%3Allm-engine-"
                            f"endpoint-builder%20env%3A{ENV}&cols=host%2Cservice&"
                            "index=%2A&messageDisplay=inline&stream_sort=time%2C"
                            "desc&viz=stream&live=true"
                        )

                    user_id = build_endpoint_request.model_endpoint_record.created_by

                    endpoint_name = build_endpoint_request.model_endpoint_record.name
                    bundle_id = build_endpoint_request.model_endpoint_record.current_model_bundle.id
                    message = (
                        f"Your endpoint '{endpoint_name}' failed to build! "
                        f"Endpoint ID: {endpoint_id}. Bundle ID: {bundle_id}."
                    )

                    self.notification_gateway.send_notification(
                        title="LLMEngine Endpoint Build Failed",
                        description=message,
                        help_url=help_url,
                        notification_apps=[
                            NotificationApp.SLACK,
                            NotificationApp.EMAIL,
                        ],
                        users=[user_id],
                    )

                    raise DockerBuildFailedException(f"Image build failed ({endpoint_id=})")

        else:
            logger_adapter.info(
                f"Image {image_params.repo}:{image_params.image_tag} already exists, "
                f"skipping build for {endpoint_id=}"
            )

        return self.docker_repository.get_image_url(image_params.image_tag, image_params.repo)

    @staticmethod
    def _validate_build_endpoint_request(
        build_endpoint_request: BuildEndpointRequest,
    ) -> None:
        """Raises ValueError if the request's AWS role isn't allowed."""
        allowed_aws_roles = {
            ml_infra_config().profile_ml_worker,
            ml_infra_config().profile_ml_inference_worker,
        }

        if build_endpoint_request.aws_role not in allowed_aws_roles:
            raise ValueError(
                f"AWS role {build_endpoint_request.aws_role} not in allowed roles "
                f"{allowed_aws_roles}."
            )

    @staticmethod
    def _get_requirements_hash(requirements: List[str]) -> str:
        """Identifying hash for endpoint's Python requirements."""
        return hashlib.md5("\n".join(sorted(requirements)).encode("utf-8")).hexdigest()[:6]

    @staticmethod
    def _get_image_tag(base_image_tag: str, git_tag: str, requirements_hash: str) -> str:
        """An identifier from an endpoint's base Docker image & git tag, plus the identify of its
        pip-installable requirements.
        """
        return "-".join([base_image_tag, git_tag, requirements_hash])

    @staticmethod
    def _get_service_builder_logs_location(user_id: str, endpoint_name: str) -> str:
        """Creates an S3 url for the user's endpoint docker build logs.

        This function uses creates a key from the endpoint's name and owning user's ID.
        It uses an S3 bucket that is accessible by the Gateway & Service Builder.
        """
        return f"s3://{ml_infra_config().s3_bucket}/service_builder_logs/{user_id}_{endpoint_name}"
