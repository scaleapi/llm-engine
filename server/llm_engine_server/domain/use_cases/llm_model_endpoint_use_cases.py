import json
from dataclasses import asdict
from typing import Any, AsyncIterable, Dict, Optional

from llm_engine_server.common.dtos.llms import (
    CompletionOutput,
    CompletionStreamOutput,
    CompletionStreamV1Request,
    CompletionStreamV1Response,
    CompletionSyncV1Request,
    CompletionSyncV1Response,
    CreateLLMModelEndpointV1Request,
    CreateLLMModelEndpointV1Response,
    GetLLMModelEndpointV1Response,
    ListLLMModelEndpointsV1Response,
)
from llm_engine_server.common.dtos.model_bundles import CreateModelBundleV2Request
from llm_engine_server.common.dtos.model_endpoints import ModelEndpointOrderBy
from llm_engine_server.common.dtos.tasks import EndpointPredictV1Request, TaskStatus
from llm_engine_server.common.resource_limits import validate_resource_requests
from llm_engine_server.core.auth.authentication_repository import User
from llm_engine_server.core.domain_exceptions import (
    ObjectHasInvalidValueException,
    ObjectNotAuthorizedException,
    ObjectNotFoundException,
)
from llm_engine_server.core.loggers import filename_wo_ext, make_logger
from llm_engine_server.domain.authorization.scale_authorization_module import (
    ScaleAuthorizationModule,
)
from llm_engine_server.domain.entities import (
    LLMInferenceFramework,
    LLMMetadata,
    LLMSource,
    ModelBundle,
    ModelBundleFlavorType,
    ModelEndpoint,
    ModelEndpointType,
    Quantization,
    RunnableImageFlavor,
    StreamingEnhancedRunnableImageFlavor,
)
from llm_engine_server.domain.exceptions import (
    EndpointLabelsException,
    EndpointUnsupportedInferenceTypeException,
)
from llm_engine_server.domain.repositories import ModelBundleRepository
from llm_engine_server.domain.services import LLMModelEndpointService, ModelEndpointService

from .model_bundle_use_cases import CreateModelBundleV2UseCase
from .model_endpoint_use_cases import (
    _handle_post_inference_hooks,
    model_endpoint_entity_to_get_model_endpoint_response,
    validate_deployment_resources,
    validate_post_inference_hooks,
)

logger = make_logger(filename_wo_ext(__name__))

_SUPPORTED_MODEL_NAMES = {
    LLMInferenceFramework.DEEPSPEED: {
        "mpt-7b": "mosaicml/mpt-7b",
        "mpt-7b-instruct": "mosaicml/mpt-7b-instruct",
        "gpt-j-6b": "EleutherAI/gpt-j-6b",
        "gpt-j-6b-zh-en": "EleutherAI/gpt-j-6b",
        "gpt4all-j": "nomic-ai/gpt4all-j",
        "dolly-v2-12b": "databricks/dolly-v2-12b",
        "stablelm-tuned-7b": "StabilityAI/stablelm-tuned-alpha-7b",
        "flan-t5-xxl": "google/flan-t5-xxl",
        "llama-7b": "decapoda-research/llama-7b-hf",
        "vicuna-13b": "eachadea/vicuna-13b-1.1",
    },
    LLMInferenceFramework.TEXT_GENERATION_INFERENCE: {
        "mpt-7b": "mosaicml/mpt-7b",
        "mpt-7b-instruct": "mosaicml/mpt-7b-instruct",
        "flan-t5-xxl": "google/flan-t5-xxl",
        "llama-7b": "decapoda-research/llama-7b-hf",
        "falcon-7b": "tiiuae/falcon-7b",
        "falcon-7b-instruct": "tiiuae/falcon-7b-instruct",
        "falcon-40b": "tiiuae/falcon-40b",
        "falcon-40b-instruct": "tiiuae/falcon-40b-instruct",
    },
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
        quantize=llm_metadata.get("quantize"),
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
        num_shards: int,
        quantize: Optional[Quantization],
        checkpoint_path: Optional[str],
    ) -> ModelBundle:
        if source == LLMSource.HUGGING_FACE:
            if framework == LLMInferenceFramework.DEEPSPEED:
                bundle_id = await self.create_deepspeed_bundle(
                    user,
                    model_name,
                    framework_image_tag,
                    endpoint_type,
                    endpoint_name,
                )
            elif framework == LLMInferenceFramework.TEXT_GENERATION_INFERENCE:
                bundle_id = await self.create_text_generation_inference_bundle(
                    user,
                    model_name,
                    framework_image_tag,
                    endpoint_name,
                    num_shards,
                    quantize,
                    checkpoint_path,
                )
            else:
                raise ObjectHasInvalidValueException(
                    f"Framework {framework} is not supported for source {source}."
                )
        else:
            raise ObjectHasInvalidValueException(f"Source {source} is not supported.")

        model_bundle = await self.model_bundle_repository.get_model_bundle(bundle_id)
        if model_bundle is None:
            raise ObjectNotFoundException(f"Model bundle {bundle_id} was not found after creation.")
        return model_bundle

    async def create_text_generation_inference_bundle(
        self,
        user: User,
        model_name: str,
        framework_image_tag: str,
        endpoint_unique_name: str,
        num_shards: int,
        quantize: Optional[Quantization],
        checkpoint_path: Optional[str],
    ):
        command = []
        if checkpoint_path is not None:
            if checkpoint_path.startswith("s3://"):
                command = ["bash", "launch_s3_model.sh", checkpoint_path, str(num_shards)]
                if quantize:
                    command = command + [f"'--quantize {str(quantize)}'"]
            else:
                raise ObjectHasInvalidValueException(
                    f"Not able to load checkpoint path {checkpoint_path}."
                )
        else:
            hf_model_name = _SUPPORTED_MODEL_NAMES[LLMInferenceFramework.TEXT_GENERATION_INFERENCE][
                model_name
            ]

            command = [
                "text-generation-launcher",
                "--model-id",
                hf_model_name,
                "--num-shard",
                str(num_shards),
                "--port",
                "5005",
                "--hostname",
                "::",
            ]
            if quantize:
                command = command + ["--quantize", str(quantize)]

        return (
            await self.create_model_bundle_use_case.execute(
                user,
                CreateModelBundleV2Request(
                    name=endpoint_unique_name,
                    schema_location="TBA",
                    flavor=StreamingEnhancedRunnableImageFlavor(
                        flavor=ModelBundleFlavorType.STREAMING_ENHANCED_RUNNABLE_IMAGE,
                        repository="text-generation-inference",  # TODO: let user choose repo
                        tag=framework_image_tag,
                        command=command,
                        streaming_command=command,
                        protocol="http",
                        readiness_initial_delay_seconds=60,
                        healthcheck_route="/health",
                        predict_route="/generate",
                        streaming_predict_route="/generate_stream",
                        env={},
                    ),
                    metadata={},
                ),
            )
        ).model_bundle_id

    async def create_deepspeed_bundle(
        self,
        user: User,
        model_name: str,
        framework_image_tag: str,
        endpoint_type: ModelEndpointType,
        endpoint_unique_name: str,
    ):
        if endpoint_type == ModelEndpointType.STREAMING:
            command = [
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
            ]
            return (
                await self.create_model_bundle_use_case.execute(
                    user,
                    CreateModelBundleV2Request(
                        name=endpoint_unique_name,
                        schema_location="TBA",
                        flavor=StreamingEnhancedRunnableImageFlavor(
                            flavor=ModelBundleFlavorType.STREAMING_ENHANCED_RUNNABLE_IMAGE,
                            repository="instant-llm",  # TODO: let user choose repo
                            tag=framework_image_tag,
                            command=command,
                            streaming_command=command,
                            env={
                                "MODEL_NAME": model_name,
                            },
                            protocol="http",
                            readiness_initial_delay_seconds=60,
                        ),
                        metadata={},
                    ),
                )
            ).model_bundle_id
        else:
            return (
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

    async def execute(
        self, user: User, request: CreateLLMModelEndpointV1Request
    ) -> CreateLLMModelEndpointV1Response:
        validate_deployment_resources(
            min_workers=request.min_workers,
            max_workers=request.max_workers,
            endpoint_type=request.endpoint_type,
        )
        if request.labels is None:
            raise EndpointLabelsException("Endpoint labels cannot be None!")
        validate_post_inference_hooks(user, request.post_inference_hooks)
        validate_model_name(request.model_name, request.inference_framework)
        validate_num_shards(request.num_shards, request.inference_framework, request.gpus)

        if request.inference_framework == LLMInferenceFramework.TEXT_GENERATION_INFERENCE:
            if request.endpoint_type != ModelEndpointType.STREAMING:
                raise ObjectHasInvalidValueException(
                    f"Creating endpoint type {str(request.endpoint_type)} is not allowed. Can only create streaming endpoints for text-generation-inference."
                )

        bundle = await self.create_model_bundle(
            user,
            endpoint_name=request.name,
            model_name=request.model_name,
            source=request.source,
            framework=request.inference_framework,
            framework_image_tag=request.inference_framework_image_tag,
            endpoint_type=request.endpoint_type,
            num_shards=request.num_shards,
            quantize=request.quantize,
            checkpoint_path=request.checkpoint_path,
        )
        validate_resource_requests(
            bundle=bundle,
            cpus=request.cpus,
            memory=request.memory,
            storage=request.storage,
            gpus=request.gpus,
            gpu_type=request.gpu_type,
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
                quantize=request.quantize,
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
        elif model_content.inference_framework == LLMInferenceFramework.TEXT_GENERATION_INFERENCE:
            return CompletionOutput(
                text=model_output["generated_text"],
                num_prompt_tokens=None,
                # len(model_output["details"]["prefill"]) does not return the correct value reliably
                num_completion_tokens=model_output["details"]["generated_tokens"],
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

        if len(model_endpoints) == 0:
            raise ObjectNotFoundException

        if len(model_endpoints) > 1:
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

        if (
            model_endpoint.record.endpoint_type is not ModelEndpointType.SYNC
            and model_endpoint.record.endpoint_type is not ModelEndpointType.STREAMING
        ):
            raise EndpointUnsupportedInferenceTypeException(
                f"Endpoint {model_endpoint_name} does not serve sync requests."
            )

        inference_gateway = self.model_endpoint_service.get_sync_model_endpoint_inference_gateway()
        endpoint_content = _model_endpoint_entity_to_get_llm_model_endpoint_response(model_endpoint)
        if endpoint_content.inference_framework == LLMInferenceFramework.DEEPSPEED:
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
        elif (
            endpoint_content.inference_framework == LLMInferenceFramework.TEXT_GENERATION_INFERENCE
        ):
            outputs = []

            for prompt in request.prompts:
                tgi_args: Any = {
                    "inputs": prompt,
                    "parameters": {
                        "max_new_tokens": request.max_new_tokens,
                        "temperature": request.temperature,
                        "decoder_input_details": True,
                    },
                }
                inference_request = EndpointPredictV1Request(args=tgi_args)
                predict_result = await inference_gateway.predict(
                    topic=model_endpoint.record.destination, predict_request=inference_request
                )

                if predict_result.status != TaskStatus.SUCCESS or predict_result.result is None:
                    return CompletionSyncV1Response(
                        status=predict_result.status,
                        outputs=[],
                        traceback=predict_result.traceback,
                    )

                outputs.append(json.loads(predict_result.result["result"]))

            return CompletionSyncV1Response(
                status=predict_result.status,
                outputs=[
                    self.model_output_to_completion_output(output, model_endpoint)
                    for output in outputs
                ],
            )
        else:
            raise EndpointUnsupportedInferenceTypeException(
                f"Unsupported inference framework {endpoint_content.inference_framework}"
            )


class CompletionStreamV1UseCase:
    """
    Use case for running a stream prompt completion on an LLM endpoint.
    """

    def __init__(
        self,
        model_endpoint_service: ModelEndpointService,
        llm_model_endpoint_service: LLMModelEndpointService,
    ):
        self.model_endpoint_service = model_endpoint_service
        self.llm_model_endpoint_service = llm_model_endpoint_service
        self.authz_module = ScaleAuthorizationModule()

    async def execute(
        self, user: User, model_endpoint_name: str, request: CompletionStreamV1Request
    ) -> AsyncIterable[CompletionStreamV1Response]:
        """
        Runs the use case to create a stream inference task.

        Args:
            user: The user who is creating the stream inference task.
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

        if len(model_endpoints) == 0:
            raise ObjectNotFoundException

        if len(model_endpoints) > 1:
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

        if model_endpoint.record.endpoint_type != ModelEndpointType.STREAMING:
            raise EndpointUnsupportedInferenceTypeException(
                f"Endpoint {model_endpoint_name} is not a streaming endpoint."
            )

        inference_gateway = (
            self.model_endpoint_service.get_streaming_model_endpoint_inference_gateway()
        )

        model_content = _model_endpoint_entity_to_get_llm_model_endpoint_response(model_endpoint)

        args: Any = None
        if model_content.inference_framework == LLMInferenceFramework.DEEPSPEED:
            args = {
                "prompts": [request.prompt],
                "token_probs": True,
                "generate_kwargs": {
                    "do_sample": True,
                    "temperature": request.temperature,
                    "max_new_tokens": request.max_new_tokens,
                },
                "serialize_results_as_string": False,
            }
        elif model_content.inference_framework == LLMInferenceFramework.TEXT_GENERATION_INFERENCE:
            args = {
                "inputs": request.prompt,
                "parameters": {
                    "max_new_tokens": request.max_new_tokens,
                    "temperature": request.temperature,
                },
            }
        inference_request = EndpointPredictV1Request(args=args)

        predict_result = inference_gateway.streaming_predict(
            topic=model_endpoint.record.destination, predict_request=inference_request
        )

        num_completion_tokens = 0
        async for res in predict_result:
            result = res.result
            if model_content.inference_framework == LLMInferenceFramework.DEEPSPEED:
                if res.status == TaskStatus.SUCCESS and result is not None:
                    if "token" in result["result"]:
                        yield CompletionStreamV1Response(
                            status=res.status,
                            output=CompletionStreamOutput(
                                text=result["result"]["token"],
                                finished=False,
                                num_prompt_tokens=None,
                                num_completion_tokens=None,
                            ),
                        )
                    else:
                        completion_token_count = len(
                            result["result"]["response"][0]["token_probs"]["tokens"]
                        )
                        total_token_count = result["result"]["response"][0]["tokens_consumed"]
                        yield CompletionStreamV1Response(
                            status=res.status,
                            output=CompletionStreamOutput(
                                text=result["result"]["response"][0]["text"],
                                finished=True,
                                num_prompt_tokens=total_token_count - completion_token_count,
                                num_completion_tokens=completion_token_count,
                            ),
                        )
                else:
                    yield CompletionStreamV1Response(
                        status=res.status,
                        output=None,
                        traceback=res.traceback,
                    )
            elif (
                model_content.inference_framework == LLMInferenceFramework.TEXT_GENERATION_INFERENCE
            ):
                if res.status == TaskStatus.SUCCESS and result is not None:
                    if result["result"].get("generated_text") is not None:
                        finished = True
                    else:
                        finished = False

                    num_completion_tokens += 1

                    yield CompletionStreamV1Response(
                        status=res.status,
                        output=CompletionStreamOutput(
                            text=result["result"]["token"]["text"],
                            finished=finished,
                            num_prompt_tokens=None,
                            num_completion_tokens=num_completion_tokens,
                        ),
                    )
                else:
                    yield CompletionStreamV1Response(
                        status=res.status,
                        output=None,
                        traceback=res.traceback,
                    )
            else:
                raise EndpointUnsupportedInferenceTypeException(
                    f"Unsupported inference framework {model_content.inference_framework}"
                )
