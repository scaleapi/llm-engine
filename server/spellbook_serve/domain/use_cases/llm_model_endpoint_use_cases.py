"""
TODO figure out how to do: (or if we want to do it)
List model endpoint history: GET model-endpoints/<endpoint id>/history
Read model endpoint creation logs: GET model-endpoints/<endpoint id>/creation-logs
"""

from dataclasses import asdict
from typing import Any, Dict, Optional

from spellbook_serve.common.dtos.llms import (
    CompletionOutput,
    CompletionSyncV1Request,
    CompletionSyncV1Response,
    CreateLLMModelEndpointV1Request,
    CreateLLMModelEndpointV1Response,
    GetLLMModelEndpointV1Response,
    ListLLMModelEndpointsV1Response,
)
from spellbook_serve.common.dtos.model_bundles import CreateModelBundleV2Request
from spellbook_serve.common.dtos.model_endpoints import ModelEndpointOrderBy
from spellbook_serve.common.dtos.tasks import EndpointPredictV1Request, TaskStatus
from spellbook_serve.common.resource_limits import validate_resource_requests
from spellbook_serve.core.auth.authentication_repository import User
from spellbook_serve.core.domain_exceptions import (
    ObjectHasInvalidValueException,
    ObjectNotAuthorizedException,
    ObjectNotFoundException,
)
from spellbook_serve.core.loggers import filename_wo_ext, make_logger
from spellbook_serve.domain.authorization.scale_authorization_module import ScaleAuthorizationModule
from spellbook_serve.domain.entities import (
    LLMInferenceFramework,
    LLMMetadata,
    LLMSource,
    ModelBundle,
    ModelBundleFlavorType,
    ModelEndpoint,
    ModelEndpointType,
    RunnableImageFlavor,
    StreamingEnhancedRunnableImageFlavor,
)
from spellbook_serve.domain.exceptions import (
    EndpointLabelsException,
    EndpointUnsupportedInferenceTypeException,
)
from spellbook_serve.domain.repositories import ModelBundleRepository
from spellbook_serve.domain.services import LLMModelEndpointService, ModelEndpointService

from .model_bundle_use_cases import CreateModelBundleV2UseCase
from .model_endpoint_use_cases import (
    _handle_post_inference_hooks,
    model_endpoint_entity_to_get_model_endpoint_response,
    validate_deployment_resources,
    validate_labels,
    validate_post_inference_hooks,
)

logger = make_logger(filename_wo_ext(__name__))

_SUPPORTED_MODEL_NAMES = {
    LLMInferenceFramework.DEEPSPEED: [
        "mpt-7b",
        "mpt-7b-instruct",
        "gpt-j-6b",
        "gpt-j-6b-zh-en",
        "gpt4all-j",
        "dolly-v2-12b",
        "stablelm-tuned-7b",
        "flan-t5-xxl",
        "llama-7b",
        "vicuna-13b",
    ]
}


def _model_endpoint_entity_to_get_llm_model_endpoint_response(
    model_endpoint: ModelEndpoint,
) -> GetLLMModelEndpointV1Response:
    if model_endpoint.record.metadata is None or "_llm" not in model_endpoint.record.metadata:
        raise ObjectHasInvalidValueException(
            f"Can't translate model entity to response, endpoint {model_endpoint.record.id} does not have LLM metadata."
        )
    llm_metadata = model_endpoint.record.metadata.get("_llm", {})
    response = GetLLMModelEndpointV1Response(
        id=model_endpoint.record.id,
        name=model_endpoint.record.name,
        model_name=llm_metadata["model_name"],
        source=llm_metadata["source"],
        inference_framework=llm_metadata["inference_framework"],
        inference_framework_image_tag=llm_metadata["inference_framework_image_tag"],
        num_shards=llm_metadata["num_shards"],
        spec=model_endpoint_entity_to_get_model_endpoint_response(model_endpoint),
    )
    return response


def validate_model_name(model_name: str, inference_framework: LLMInferenceFramework) -> None:
    if model_name not in _SUPPORTED_MODEL_NAMES[inference_framework]:
        raise ObjectHasInvalidValueException(
            f"Model name {model_name} is not supported for inference framework {inference_framework}."
        )


def validate_num_shards(
    num_shards: int, inference_framework: LLMInferenceFramework, gpus: int
) -> None:
    if inference_framework == LLMInferenceFramework.DEEPSPEED:
        if num_shards <= 1:
            raise ObjectHasInvalidValueException("DeepSpeed requires more than 1 GPU.")
        if num_shards != gpus:
            raise ObjectHasInvalidValueException(
                f"DeepSpeed requires num shard {num_shards} to be the same as number of GPUs {gpus}."
            )


class CreateLLMModelEndpointV1UseCase:
    def __init__(
        self,
        create_model_bundle_use_case: CreateModelBundleV2UseCase,
        model_bundle_repository: ModelBundleRepository,
        model_endpoint_service: ModelEndpointService,
    ):
        self.authz_module = ScaleAuthorizationModule()
        self.create_model_bundle_use_case = create_model_bundle_use_case
        self.model_bundle_repository = model_bundle_repository
        self.model_endpoint_service = model_endpoint_service

    async def create_model_bundle(
        self,
        user: User,
        endpoint_name: str,
        model_name: str,
        source: LLMSource,
        framework: LLMInferenceFramework,
        framework_image_tag: str,
        endpoint_type: ModelEndpointType,
    ) -> ModelBundle:
        endpoint_unique_name = f"llm-{endpoint_name}-{str(framework)}-{str(endpoint_type)}"
        if source == LLMSource.HUGGING_FACE:
            if framework == LLMInferenceFramework.DEEPSPEED:
                if endpoint_type == ModelEndpointType.STREAMING:
                    bundle_id = (
                        await self.create_model_bundle_use_case.execute(
                            user,
                            CreateModelBundleV2Request(
                                name=endpoint_unique_name,
                                schema_location="TBA",
                                flavor=StreamingEnhancedRunnableImageFlavor(
                                    flavor=ModelBundleFlavorType.STREAMING_ENHANCED_RUNNABLE_IMAGE,
                                    repository="instant-llm",
                                    tag=framework_image_tag,
                                    command=[
                                        "dumb-init",
                                        "--",
                                        "ddtrace-run",
                                        "run-streamer",
                                        "--http",
                                        "production_threads",
                                        "--concurrency",
                                        "1",
                                        "--config",
                                        "/install/spellbook/inference/service--spellbook_streaming_inference.yaml",
                                    ],
                                    streaming_command=[
                                        "dumb-init",
                                        "--",
                                        "ddtrace-run",
                                        "run-streamer",
                                        "--http",
                                        "production_threads",
                                        "--concurrency",
                                        "1",
                                        "--config",
                                        "/install/spellbook/inference/service--spellbook_streaming_inference.yaml",
                                    ],
                                    env={
                                        "MODEL_NAME": model_name,
                                    },
                                    protocol="http",
                                    readiness_initial_delay_seconds=1800,
                                ),
                                metadata={},
                            ),
                        )
                    ).model_bundle_id
                else:
                    bundle_id = (
                        await self.create_model_bundle_use_case.execute(
                            user,
                            CreateModelBundleV2Request(
                                name=endpoint_unique_name,
                                schema_location="TBA",
                                flavor=RunnableImageFlavor(
                                    flavor=ModelBundleFlavorType.RUNNABLE_IMAGE,
                                    repository="instant-llm",
                                    tag=framework_image_tag,
                                    command=[
                                        "dumb-init",
                                        "--",
                                        "ddtrace-run",
                                        "run-service",
                                        "--http",
                                        "production_threads",
                                        "--concurrency",
                                        "1",
                                        "--config",
                                        "/install/spellbook/inference/service--spellbook_inference.yaml",
                                    ],
                                    env={
                                        "MODEL_NAME": model_name,
                                    },
                                    protocol="http",
                                    readiness_initial_delay_seconds=1800,
                                ),
                                metadata={},
                            ),
                        )
                    ).model_bundle_id

                model_bundle = await self.model_bundle_repository.get_model_bundle(bundle_id)
                if model_bundle is None:
                    raise ObjectNotFoundException(
                        f"Model bundle {bundle_id} was not found after creation."
                    )
                return model_bundle
            else:
                raise ObjectHasInvalidValueException(
                    f"Framework {framework} is not supported for source {source}."
                )
        else:
            raise ObjectHasInvalidValueException(f"Source {source} is not supported.")

    async def execute(
        self, user: User, request: CreateLLMModelEndpointV1Request
    ) -> CreateLLMModelEndpointV1Response:
        validate_resource_requests(
            cpus=request.cpus,
            memory=request.memory,
            storage=request.storage,
            gpus=request.gpus,
            gpu_type=request.gpu_type,
        )
        validate_deployment_resources(
            min_workers=request.min_workers,
            max_workers=request.max_workers,
            endpoint_type=request.endpoint_type,
        )
        if request.labels is None:
            raise EndpointLabelsException("Endpoint labels cannot be None!")
        validate_labels(request.labels)
        validate_post_inference_hooks(user, request.post_inference_hooks)
        validate_model_name(request.model_name, request.inference_framework)
        validate_num_shards(request.num_shards, request.inference_framework, request.gpus)

        bundle = await self.create_model_bundle(
            user,
            endpoint_name=request.name,
            model_name=request.model_name,
            source=request.source,
            framework=request.inference_framework,
            framework_image_tag=request.inference_framework_image_tag,
            endpoint_type=request.endpoint_type,
        )

        prewarm = request.prewarm
        if prewarm is None:
            prewarm = True

        high_priority = request.high_priority
        if high_priority is None:
            high_priority = False

        aws_role = self.authz_module.get_aws_role_for_user(user)
        results_s3_bucket = self.authz_module.get_s3_bucket_for_user(user)

        request.metadata["_llm"] = asdict(
            LLMMetadata(
                model_name=request.model_name,
                source=request.source,
                inference_framework=request.inference_framework,
                inference_framework_image_tag=request.inference_framework_image_tag,
                num_shards=request.num_shards,
            )
        )

        model_endpoint_record = await self.model_endpoint_service.create_model_endpoint(
            name=request.name,
            created_by=user.user_id,
            model_bundle_id=bundle.id,
            endpoint_type=request.endpoint_type,
            metadata=request.metadata,
            post_inference_hooks=request.post_inference_hooks,
            child_fn_info=None,
            cpus=request.cpus,
            gpus=request.gpus,
            memory=request.memory,
            gpu_type=request.gpu_type,
            storage=request.storage,
            optimize_costs=bool(request.optimize_costs),
            min_workers=request.min_workers,
            max_workers=request.max_workers,
            per_worker=request.per_worker,
            labels=request.labels,
            aws_role=aws_role,
            results_s3_bucket=results_s3_bucket,
            prewarm=prewarm,
            high_priority=high_priority,
            owner=user.team_id,
            default_callback_url=request.default_callback_url,
            default_callback_auth=request.default_callback_auth,
            public_inference=request.public_inference,
        )
        _handle_post_inference_hooks(
            created_by=user.user_id,
            name=request.name,
            post_inference_hooks=request.post_inference_hooks,
        )

        return CreateLLMModelEndpointV1Response(
            endpoint_creation_task_id=model_endpoint_record.creation_task_id  # type: ignore
        )


class ListLLMModelEndpointsV1UseCase:
    """
    Use case for listing all LLM Model Endpoint of a given user and model endpoint name.
    Also include public_inference LLM endpoints.
    """

    def __init__(self, llm_model_endpoint_service: LLMModelEndpointService):
        self.llm_model_endpoint_service = llm_model_endpoint_service

    async def execute(
        self, user: User, name: Optional[str], order_by: Optional[ModelEndpointOrderBy]
    ) -> ListLLMModelEndpointsV1Response:
        """
        Runs the use case to list all Model Endpoints owned by the user with the given name.

        Args:
            user: The owner of the model endpoint(s).
            name: The name of the Model Endpoint(s).
            order_by: An optional argument to specify the output ordering of the model endpoints.

        Returns:
            A response object that contains the model endpoints.
        """
        model_endpoints = await self.llm_model_endpoint_service.list_llm_model_endpoints(
            owner=user.team_id, name=name, order_by=order_by
        )
        return ListLLMModelEndpointsV1Response(
            model_endpoints=[
                _model_endpoint_entity_to_get_llm_model_endpoint_response(m)
                for m in model_endpoints
            ]
        )


class GetLLMModelEndpointByNameV1UseCase:
    """
    Use case for getting an LLM Model Endpoint of a given user by name.
    """

    def __init__(self, llm_model_endpoint_service: LLMModelEndpointService):
        self.llm_model_endpoint_service = llm_model_endpoint_service
        self.authz_module = ScaleAuthorizationModule()

    async def execute(self, user: User, model_endpoint_name: str) -> GetLLMModelEndpointV1Response:
        """
        Runs the use case to get the LLM endpoint with the given name.

        Args:
            user: The owner of the model endpoint.
            model_endpoint_name: The name of the model endpoint.

        Returns:
            A response object that contains the model endpoint.

        Raises:
            ObjectNotFoundException: If a model endpoint with the given name could not be found.
            ObjectNotAuthorizedException: If the owner does not own the model endpoint.
        """
        model_endpoint = await self.llm_model_endpoint_service.get_llm_model_endpoint(
            model_endpoint_name
        )
        if not model_endpoint:
            raise ObjectNotFoundException
        if not self.authz_module.check_access_read_owned_entity(
            user, model_endpoint.record
        ) and not self.authz_module.check_endpoint_public_inference_for_user(
            user, model_endpoint.record
        ):
            raise ObjectNotAuthorizedException
        return _model_endpoint_entity_to_get_llm_model_endpoint_response(model_endpoint)


class DeleteLLMModelEndpointByIdV1UseCase:
    pass


class CompletionSyncV1UseCase:
    """
    Use case for running a prompt completion on an LLM endpoint.
    """

    def __init__(
        self,
        model_endpoint_service: ModelEndpointService,
        llm_model_endpoint_service: LLMModelEndpointService,
    ):
        self.model_endpoint_service = model_endpoint_service
        self.llm_model_endpoint_service = llm_model_endpoint_service
        self.authz_module = ScaleAuthorizationModule()

    def model_output_to_completion_output(
        self,
        model_output: Dict[str, Any],
        model_endpoint: ModelEndpoint,
    ) -> CompletionOutput:
        model_content = _model_endpoint_entity_to_get_llm_model_endpoint_response(model_endpoint)

        if model_content.inference_framework == LLMInferenceFramework.DEEPSPEED:
            completion_token_count = len(model_output["token_probs"]["tokens"])
            total_token_count = model_output["tokens_consumed"]
            return CompletionOutput(
                text=model_output["text"],
                num_prompt_tokens=total_token_count - completion_token_count,
                num_completion_tokens=completion_token_count,
            )
        else:
            raise EndpointUnsupportedInferenceTypeException(
                f"Unsupported inference framework {model_content.inference_framework}"
            )

    async def execute(
        self, user: User, model_endpoint_name: str, request: CompletionSyncV1Request
    ) -> CompletionSyncV1Response:
        """
        Runs the use case to create a sync inference task.

        Args:
            user: The user who is creating the sync inference task.
            model_endpoint_name: The name of the model endpoint for the task.
            request: The body of the request to forward to the endpoint.

        Returns:
            A response object that contains the status and result of the task.

        Raises:
            ObjectNotFoundException: If a model endpoint with the given name could not be found.
            ObjectNotAuthorizedException: If the owner does not own the model endpoint.
        """

        model_endpoints = await self.llm_model_endpoint_service.list_llm_model_endpoints(
            owner=user.team_id, name=model_endpoint_name, order_by=None
        )

        if not len(model_endpoints) == 1:
            raise ObjectHasInvalidValueException(
                f"Expected 1 LLM model endpoint for model name {model_endpoint_name}, got {len(model_endpoints)}"
            )

        model_endpoint = model_endpoints[0]

        if not self.authz_module.check_access_read_owned_entity(
            user, model_endpoint.record
        ) and not self.authz_module.check_endpoint_public_inference_for_user(
            user, model_endpoint.record
        ):
            raise ObjectNotAuthorizedException

        if model_endpoint.record.endpoint_type != ModelEndpointType.SYNC:
            raise EndpointUnsupportedInferenceTypeException(
                f"Endpoint {model_endpoint_name} is not a sync endpoint."
            )

        inference_gateway = self.model_endpoint_service.get_sync_model_endpoint_inference_gateway()
        args: Any = {
            "prompts": request.prompts,
            "token_probs": True,
            "generate_kwargs": {
                "do_sample": True,
                "temperature": request.temperature,
                "max_new_tokens": request.max_new_tokens,
            },
            "serialize_results_as_string": False,
        }
        inference_request = EndpointPredictV1Request(args=args)
        predict_result = await inference_gateway.predict(
            topic=model_endpoint.record.destination, predict_request=inference_request
        )

        if predict_result.status == TaskStatus.SUCCESS and predict_result.result is not None:
            return CompletionSyncV1Response(
                status=predict_result.status,
                outputs=[
                    self.model_output_to_completion_output(result, model_endpoint)
                    for result in predict_result.result["result"]
                ],
            )
        else:
            return CompletionSyncV1Response(
                status=predict_result.status,
                outputs=[],
                traceback=predict_result.traceback,
            )
