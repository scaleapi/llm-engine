from typing import Dict, List, Optional

from llmengine.api_engine import DEFAULT_TIMEOUT, APIEngine, assert_self_hosted
from llmengine.data_types import (
    CreateLLMEndpointRequest,
    CreateLLMEndpointResponse,
    DeleteLLMEndpointResponse,
    GetLLMEndpointResponse,
    GpuType,
    ListLLMEndpointsResponse,
    LLMInferenceFramework,
    LLMSource,
    ModelDownloadRequest,
    ModelDownloadResponse,
    ModelEndpointType,
    PostInferenceHooks,
    Quantization,
)


class Model(APIEngine):
    """
    Model API. This API is used to get, list, and delete models. Models include both base
    models built into LLM Engine, and fine-tuned models that you create through the
    [FineTune.create()](./#llmengine.fine_tuning.FineTune.create) API.

    See [Model Zoo](../../model_zoo) for the list of publicly available base models.
    """

    @classmethod
    @assert_self_hosted
    def create(
        cls,
        name: str,
        # LLM specific fields
        model: str,
        inference_framework_image_tag: str,
        source: LLMSource = LLMSource.HUGGING_FACE,
        inference_framework: LLMInferenceFramework = LLMInferenceFramework.VLLM,
        num_shards: int = 1,
        quantize: Optional[Quantization] = None,
        checkpoint_path: Optional[str] = None,
        # General endpoint fields
        cpus: int = 8,
        memory: str = "24Gi",
        storage: str = "40Gi",
        gpus: int = 1,
        min_workers: int = 0,
        max_workers: int = 1,
        per_worker: int = 2,
        endpoint_type: ModelEndpointType = ModelEndpointType.STREAMING,
        gpu_type: Optional[str] = "nvidia-ampere-a10",
        high_priority: Optional[bool] = False,
        post_inference_hooks: Optional[List[PostInferenceHooks]] = None,
        default_callback_url: Optional[str] = None,
        public_inference: Optional[bool] = True,
        labels: Optional[Dict[str, str]] = None,
    ) -> CreateLLMEndpointResponse:
        """
        Create an LLM model. Note: This API is only available for self-hosted users.

        Args:
            name (`str`):
                Name of the endpoint

            model (`str`):
                Name of the base model

            inference_framework_image_tag (`str`):
                Image tag for the inference framework. Use "latest" for the most recent image

            source (`LLMSource`):
                Source of the LLM. Currently only HuggingFace is supported

            inference_framework (`LLMInferenceFramework`):
                Inference framework for the LLM. Current supported frameworks are
                LLMInferenceFramework.DEEPSPEED, LLMInferenceFramework.TEXT_GENERATION_INFERENCE,
                LLMInferenceFramework.VLLM and LLMInferenceFramework.LIGHTLLM

            num_shards (`int`):
                Number of shards for the LLM. When bigger than 1, LLM will be sharded
                to multiple GPUs. Number of GPUs must be equal or larger than num_shards.

            quantize (`Optional[Quantization]`):
                Quantization method for the LLM. `text_generation_inference` supports `bitsandbytes` and `vllm` supports `awq`.

            checkpoint_path (`Optional[str]`):
                Remote path to the checkpoint for the LLM. LLM engine must have permission to access the given path.
                Can be either a folder or a tar file. Folder is preferred since we don't need to untar and model loads faster.
                For model weights, safetensors are preferred but PyTorch checkpoints are also accepted (model loading will be longer).

            cpus (`int`):
                Number of cpus each worker should get, e.g. 1, 2, etc. This must be greater
                than or equal to 1. Recommendation is set it to 8 * GPU count.

            memory (`str`):
                Amount of memory each worker should get, e.g. "4Gi", "512Mi", etc. This must
                be a positive amount of memory. Recommendation is set it to 24Gi * GPU count.

            storage (`str`):
                Amount of local ephemeral storage each worker should get, e.g. "4Gi",
                "512Mi", etc. This must be a positive amount of storage.
                Recommendataion is 40Gi for 7B models, 80Gi for 13B models and 200Gi for 70B models.

            gpus (`int`):
                Number of gpus each worker should get, e.g. 0, 1, etc.

            min_workers (`int`):
                The minimum number of workers. Must be greater than or equal to 0. This
                should be determined by computing the minimum throughput of your workload and
                dividing it by the throughput of a single worker. When this number is 0,
                max_workers must be 1, and the endpoint will autoscale between
                0 and 1 pods. When this number is greater than 0, max_workers can be any number
                greater or equal to min_workers.

            max_workers (`int`):
                The maximum number of workers. Must be greater than or equal to 0,
                and as well as greater than or equal to ``min_workers``. This should be determined by
                computing the maximum throughput of your workload and dividing it by the throughput
                of a single worker

            per_worker (`int`):
                The maximum number of concurrent requests that an individual worker can
                service. LLM engine automatically scales the number of workers for the endpoint so that
                each worker is processing ``per_worker`` requests, subject to the limits defined by
                ``min_workers`` and ``max_workers``
                - If the average number of concurrent requests per worker is lower than
                ``per_worker``, then the number of workers will be reduced. - Otherwise,
                if the average number of concurrent requests per worker is higher than
                ``per_worker``, then the number of workers will be increased to meet the elevated
                traffic.
                Here is our recommendation for computing ``per_worker``:
                1. Compute ``min_workers`` and ``max_workers`` per your minimum and maximum
                throughput requirements. 2. Determine a value for the maximum number of
                concurrent requests in the workload. Divide this number by ``max_workers``. Doing
                this ensures that the number of workers will "climb" to ``max_workers``.

            endpoint_type (`ModelEndpointType`):
                Currently only ``"streaming"`` endpoints are supported.

            gpu_type (`Optional[str]`):
                If specifying a non-zero number of gpus, this controls the type of gpu
                requested. Here are the supported values:

                - ``nvidia-tesla-t4``
                - ``nvidia-ampere-a10``
                - ``nvidia-ampere-a100``
                - ``nvidia-ampere-a100e``

            high_priority (`Optional[bool]`):
                Either ``True`` or ``False``. Enabling this will allow the created
                endpoint to leverage the shared pool of prewarmed nodes for faster spinup time

            post_inference_hooks (`Optional[List[PostInferenceHooks]]`):
                List of hooks to trigger after inference tasks are served

            default_callback_url (`Optional[str]`):
                The default callback url to use for sync completion requests.
                This can be overridden in the task parameters for each individual task.
                post_inference_hooks must contain "callback" for the callback to be triggered

            public_inference (`Optional[bool]`):
                If ``True``, this endpoint will be available to all user IDs for
                inference

            labels (`Optional[Dict[str, str]]`):
                An optional dictionary of key/value pairs to associate with this endpoint
        Returns:
            CreateLLMEndpointResponse: creation task ID of the created Model. Currently not used.

        === "Create Llama 2 7B model in Python"
            ```python
            from llmengine import Model

            response = Model.create(
                name="llama-2-7b-test"
                model="llama-2-7b",
                inference_framework_image_tag="0.2.1.post1",
                inference_framework=LLMInferenceFramework.VLLM,
                num_shards=1,
                checkpoint_path="s3://path/to/checkpoint",
                cpus=8,
                memory="24Gi",
                storage="40Gi",
                gpus=1,
                min_workers=0,
                max_workers=1,
                per_worker=10,
                endpoint_type=ModelEndpointType.STREAMING,
                gpu_type="nvidia-ampere-a10",
                public_inference=False,
            )

            print(response.json())
            ```

        === "Create Llama 2 13B model in Python"
            ```python
            from llmengine import Model

            response = Model.create(
                name="llama-2-13b-test"
                model="llama-2-13b",
                inference_framework_image_tag="0.2.1.post1",
                inference_framework=LLMInferenceFramework.VLLM,
                num_shards=2,
                checkpoint_path="s3://path/to/checkpoint",
                cpus=16,
                memory="48Gi",
                storage="80Gi",
                gpus=2,
                min_workers=0,
                max_workers=1,
                per_worker=10,
                endpoint_type=ModelEndpointType.STREAMING,
                gpu_type="nvidia-ampere-a10",
                public_inference=False,
            )

            print(response.json())
            ```

        === "Create Llama 2 70B model with 8bit quantization in Python"
            ```python
            from llmengine import Model

            response = Model.create(
                name="llama-2-70b-test"
                model="llama-2-70b",
                inference_framework_image_tag="0.9.4",
                inference_framework=LLMInferenceFramework.TEXT_GENERATION_INFERENCE,
                num_shards=4,
                quantize="bitsandbytes",
                checkpoint_path="s3://path/to/checkpoint",
                cpus=40,
                memory="96Gi",
                storage="200Gi",
                gpus=4,
                min_workers=0,
                max_workers=1,
                per_worker=10,
                endpoint_type=ModelEndpointType.STREAMING,
                gpu_type="nvidia-ampere-a10",
                public_inference=False,
            )

            print(response.json())
            ```
        """
        post_inference_hooks_strs = None
        if post_inference_hooks is not None:
            post_inference_hooks_strs = []
            for hook in post_inference_hooks:
                if isinstance(hook, PostInferenceHooks):
                    post_inference_hooks_strs.append(hook.value)
                else:
                    post_inference_hooks_strs.append(hook)

        request = CreateLLMEndpointRequest(
            name=name,
            model_name=model,
            source=source,
            inference_framework=inference_framework,
            inference_framework_image_tag=inference_framework_image_tag,
            num_shards=num_shards,
            quantize=quantize,
            checkpoint_path=checkpoint_path,
            cpus=cpus,
            endpoint_type=ModelEndpointType(endpoint_type),
            gpus=gpus,
            gpu_type=GpuType(gpu_type) if gpu_type is not None else None,
            labels=labels or {},
            max_workers=max_workers,
            memory=memory,
            metadata={},
            min_workers=min_workers,
            per_worker=per_worker,
            high_priority=high_priority,
            post_inference_hooks=post_inference_hooks_strs,
            # Pydantic automatically validates the url
            default_callback_url=default_callback_url,  # type: ignore
            storage=storage,
            public_inference=public_inference,
        )
        response = cls.post_sync(
            resource_name="v1/llm/model-endpoints",
            data=request.dict(),
            timeout=DEFAULT_TIMEOUT,
        )
        return CreateLLMEndpointResponse.parse_obj(response)

    @classmethod
    def get(
        cls,
        model: str,
    ) -> GetLLMEndpointResponse:
        """
        Get information about an LLM model.

        This API can be used to get information about a Model's source and inference framework.
        For self-hosted users, it returns additional information about number of shards, quantization, infra settings, etc.
        The function takes as a single parameter the name `model`
        and returns a
        [GetLLMEndpointResponse](../../api/data_types/#llmengine.GetLLMEndpointResponse)
        object.

        Args:
            model (`str`):
                Name of the model

        Returns:
            GetLLMEndpointResponse: object representing the LLM and configurations

        === "Accessing model in Python"
            ```python
            from llmengine import Model

            response = Model.get("llama-2-7b.suffix.2023-07-18-12-00-00")

            print(response.json())
            ```

        === "Response in JSON"
            ```json
            {
                "id": null,
                "name": "llama-2-7b.suffix.2023-07-18-12-00-00",
                "model_name": null,
                "source": "hugging_face",
                "status": "READY",
                "inference_framework": "text_generation_inference",
                "inference_framework_tag": null,
                "num_shards": null,
                "quantize": null,
                "spec": null
            }
            ```
        """
        response = cls._get(f"v1/llm/model-endpoints/{model}", timeout=DEFAULT_TIMEOUT)
        return GetLLMEndpointResponse.parse_obj(response)

    @classmethod
    def list(cls) -> ListLLMEndpointsResponse:
        """
        List LLM models available to call inference on.

        This API can be used to list all available models, including both publicly
        available models and user-created fine-tuned models.
        It returns a list of
        [GetLLMEndpointResponse](../../api/data_types/#llmengine.GetLLMEndpointResponse)
        objects for all models. The most important field is the model `name`.

        Returns:
            ListLLMEndpointsResponse: list of models

        === "Listing available modes in Python"
            ```python
            from llmengine import Model

            response = Model.list()
            print(response.json())
            ```

        === "Response in JSON"
            ```json
            {
                "model_endpoints": [
                    {
                        "id": null,
                        "name": "llama-2-7b.suffix.2023-07-18-12-00-00",
                        "model_name": null,
                        "source": "hugging_face",
                        "inference_framework": "text_generation_inference",
                        "inference_framework_tag": null,
                        "num_shards": null,
                        "quantize": null,
                        "spec": null
                    },
                    {
                        "id": null,
                        "name": "llama-2-7b",
                        "model_name": null,
                        "source": "hugging_face",
                        "inference_framework": "text_generation_inference",
                        "inference_framework_tag": null,
                        "num_shards": null,
                        "quantize": null,
                        "spec": null
                    },
                    {
                        "id": null,
                        "name": "llama-13b-deepspeed-sync",
                        "model_name": null,
                        "source": "hugging_face",
                        "inference_framework": "deepspeed",
                        "inference_framework_tag": null,
                        "num_shards": null,
                        "quantize": null,
                        "spec": null
                    },
                    {
                        "id": null,
                        "name": "falcon-40b",
                        "model_name": null,
                        "source": "hugging_face",
                        "inference_framework": "text_generation_inference",
                        "inference_framework_tag": null,
                        "num_shards": null,
                        "quantize": null,
                        "spec": null
                    }
                ]
            }
            ```
        """
        response = cls._get("v1/llm/model-endpoints", timeout=DEFAULT_TIMEOUT)
        return ListLLMEndpointsResponse.parse_obj(response)

    @classmethod
    def delete(cls, model_endpoint_name: str) -> DeleteLLMEndpointResponse:
        """
        Deletes an LLM model.

        This API can be used to delete a fine-tuned model. It takes
        as parameter the name of the `model` and returns a response
        object which has a `deleted` field confirming if the deletion
        was successful. If called on a base model included with LLM
        Engine, an error will be thrown.

        Args:
            model_endpoint_name (`str`):
                Name of the model endpoint to be deleted

        Returns:
            response: whether the model endpoint was successfully deleted

        === "Deleting model in Python"
            ```python
            from llmengine import Model

            response = Model.delete("llama-2-7b.suffix.2023-07-18-12-00-00")
            print(response.json())
            ```

        === "Response in JSON"
            ```json
            {
                "deleted": true
            }
            ```
        """
        response = cls._delete(
            f"v1/llm/model-endpoints/{model_endpoint_name}", timeout=DEFAULT_TIMEOUT
        )
        return DeleteLLMEndpointResponse.parse_obj(response)

    @classmethod
    def download(
        cls,
        model_name: str,
        download_format: str = "hugging_face",
    ) -> ModelDownloadResponse:
        """
        Download a fine-tuned model.

        This API can be used to download the resulting model from a fine-tuning job.
        It takes the `model_name` and `download_format` as parameter and returns a
        response object which contains a dictonary of filename, url pairs associated
        with the fine-tuned model. The user can then download these urls to obtain
        the fine-tuned model. If called on a nonexistent model, an error will be thrown.

        Args:
            model_name (`str`):
                name of the fine-tuned model
            download_format (`str`):
                download format requested (default=hugging_face)
        Returns:
            DownloadModelResponse: an object that contains a dictionary of filenames, urls from which to download the model weights.
            The urls are presigned urls that grant temporary access and expire after an hour.

        === "Downloading model in Python"
            ```python
            from llmengine import Model

            response = Model.download("llama-2-7b.suffix.2023-07-18-12-00-00", download_format="hugging_face")
            print(response.json())
            ```

        === "Response in JSON"
            ```json
            {
                "urls": {"my_model_file": "https://url-to-my-model-weights"}
            }
            ```
        """

        request = ModelDownloadRequest(model_name=model_name, download_format=download_format)
        response = cls.post_sync(
            resource_name="v1/llm/model-endpoints/download",
            data=request.dict(),
            timeout=DEFAULT_TIMEOUT,
        )
        return ModelDownloadResponse.parse_obj(response)
