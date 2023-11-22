import pytest
from model_engine_server.common.dtos.batch_jobs import (
    CreateDockerImageBatchJobBundleV1Request,
    CreateDockerImageBatchJobResourceRequests,
)
from model_engine_server.common.dtos.llms import (
    CompletionStreamV1Request,
    CompletionSyncV1Request,
    CreateLLMModelEndpointV1Request,
    UpdateLLMModelEndpointV1Request,
)
from model_engine_server.common.dtos.model_bundles import (
    CreateModelBundleV1Request,
    CreateModelBundleV2Request,
)
from model_engine_server.common.dtos.model_endpoints import (
    CreateModelEndpointV1Request,
    UpdateModelEndpointV1Request,
)
from model_engine_server.domain.entities import (
    GpuType,
    LLMInferenceFramework,
    ModelBundle,
    ModelBundleEnvironmentParams,
    ModelBundleFrameworkType,
    ModelBundlePackagingType,
    ModelEndpointType,
    Quantization,
    StreamingEnhancedRunnableImageFlavor,
)


@pytest.fixture
def create_model_bundle_request() -> CreateModelBundleV1Request:
    env_params = ModelBundleEnvironmentParams(
        framework_type=ModelBundleFrameworkType.CUSTOM,
        ecr_repo="test_repo",
        image_tag="test_tag",
    )
    return CreateModelBundleV1Request(
        name="test_bundle_name",
        location="test_location",
        requirements=["numpy==0.0.0"],
        env_params=env_params,
        packaging_type=ModelBundlePackagingType.CLOUDPICKLE,
        metadata=None,
        app_config=None,
    )


@pytest.fixture
def create_model_bundle_v2_request() -> CreateModelBundleV2Request:
    return CreateModelBundleV2Request(
        name="test_bundle_name",
        metadata=None,
        schema_location="s3://test-bucket/test-key",
        flavor=StreamingEnhancedRunnableImageFlavor(
            flavor="streaming_enhanced_runnable_image",
            repository="test_repo",
            tag="test_tag",
            command=["test_command"],
            env={"test_key": "test_value"},
            protocol="http",
            readiness_initial_delay_seconds=30,
            streaming_command=["test_streaming_command"],
            streaming_predict_route="/test_streaming_predict_route",
        ),
    )


@pytest.fixture
def create_model_endpoint_request_sync(
    model_bundle_1: ModelBundle,
) -> CreateModelEndpointV1Request:
    return CreateModelEndpointV1Request(
        name="test_endpoint_name_1",
        model_bundle_id=model_bundle_1.id,
        endpoint_type=ModelEndpointType.SYNC,
        metadata={},
        post_inference_hooks=["billing"],
        cpus=1,
        gpus=1,
        memory="8G",
        gpu_type=GpuType.NVIDIA_TESLA_T4,
        storage=None,
        min_workers=1,
        max_workers=3,
        per_worker=2,
        labels={"team": "infra", "product": "my_product"},
        aws_role="test_aws_role",
        results_s3_bucket="test_s3_bucket",
    )


@pytest.fixture
def create_model_endpoint_request_streaming(
    model_bundle_5: ModelBundle,
) -> CreateModelEndpointV1Request:
    return CreateModelEndpointV1Request(
        name="test_endpoint_name_2",
        model_bundle_id=model_bundle_5.id,
        endpoint_type=ModelEndpointType.STREAMING,
        metadata={},
        post_inference_hooks=["billing"],
        cpus=1,
        gpus=1,
        memory="8G",
        gpu_type=GpuType.NVIDIA_TESLA_T4,
        storage="10G",
        min_workers=1,
        max_workers=3,
        per_worker=1,
        labels={"team": "infra", "product": "my_product"},
        aws_role="test_aws_role",
        results_s3_bucket="test_s3_bucket",
    )


@pytest.fixture
def create_model_endpoint_request_async(
    model_bundle_1: ModelBundle,
) -> CreateModelEndpointV1Request:
    return CreateModelEndpointV1Request(
        name="test_endpoint_name_2",
        model_bundle_id=model_bundle_1.id,
        endpoint_type=ModelEndpointType.ASYNC,
        metadata={},
        post_inference_hooks=["billing"],
        cpus=1,
        gpus=1,
        memory="8G",
        gpu_type=GpuType.NVIDIA_TESLA_T4,
        storage="10G",
        min_workers=1,
        max_workers=3,
        per_worker=2,
        labels={"team": "infra", "product": "my_product"},
        aws_role="test_aws_role",
        results_s3_bucket="test_s3_bucket",
    )


@pytest.fixture
def update_model_endpoint_request(
    model_bundle_2: ModelBundle,
) -> UpdateModelEndpointV1Request:
    return UpdateModelEndpointV1Request(
        model_bundle_id=model_bundle_2.id,
        metadata={"test_new_key": "test_new_value"},
        cpus=2,
        memory="16G",
        min_workers=1,
        max_workers=4,
        per_worker=2,
    )


@pytest.fixture
def create_docker_image_batch_job_bundle_request() -> CreateDockerImageBatchJobBundleV1Request:
    return CreateDockerImageBatchJobBundleV1Request(
        name="name",
        image_repository="repo",
        image_tag="tag",
        command=["sudo", "rn", "-rf"],
        env=dict(hi="hi", bye="bye"),
        mount_location=None,
        resource_requests=CreateDockerImageBatchJobResourceRequests(
            cpus=1, memory=None, storage=None, gpus=None, gpu_type=None
        ),
    )


@pytest.fixture
def create_llm_model_endpoint_request_sync() -> CreateLLMModelEndpointV1Request:
    return CreateLLMModelEndpointV1Request(
        name="test_llm_endpoint_name_sync",
        model_name="mpt-7b",
        source="hugging_face",
        inference_framework="deepspeed",
        inference_framework_image_tag="test_tag",
        num_shards=2,
        endpoint_type=ModelEndpointType.SYNC,
        metadata={},
        post_inference_hooks=["billing"],
        cpus=1,
        gpus=2,
        memory="8G",
        gpu_type=GpuType.NVIDIA_TESLA_T4,
        storage=None,
        min_workers=1,
        max_workers=3,
        per_worker=2,
        labels={"team": "infra", "product": "my_product"},
        aws_role="test_aws_role",
        results_s3_bucket="test_s3_bucket",
    )


@pytest.fixture
def create_llm_model_endpoint_request_async() -> CreateLLMModelEndpointV1Request:
    return CreateLLMModelEndpointV1Request(
        name="test_llm_endpoint_name_async",
        model_name="mpt-7b",
        source="hugging_face",
        inference_framework="deepspeed",
        inference_framework_image_tag="test_tag",
        num_shards=2,
        endpoint_type=ModelEndpointType.ASYNC,
        metadata={},
        post_inference_hooks=["billing"],
        cpus=1,
        gpus=2,
        memory="8G",
        gpu_type=GpuType.NVIDIA_TESLA_T4,
        storage=None,
        min_workers=0,
        max_workers=3,
        per_worker=2,
        labels={"team": "infra", "product": "my_product"},
        aws_role="test_aws_role",
        results_s3_bucket="test_s3_bucket",
    )


@pytest.fixture
def create_llm_model_endpoint_request_streaming() -> CreateLLMModelEndpointV1Request:
    return CreateLLMModelEndpointV1Request(
        name="test_llm_endpoint_name_streaming",
        model_name="mpt-7b",
        source="hugging_face",
        inference_framework="deepspeed",
        inference_framework_image_tag="test_tag",
        num_shards=2,
        endpoint_type=ModelEndpointType.STREAMING,
        metadata={},
        post_inference_hooks=["billing"],
        cpus=1,
        gpus=2,
        memory="8G",
        gpu_type=GpuType.NVIDIA_TESLA_T4,
        storage=None,
        min_workers=1,
        max_workers=3,
        per_worker=2,
        labels={"team": "infra", "product": "my_product"},
        aws_role="test_aws_role",
        results_s3_bucket="test_s3_bucket",
    )


@pytest.fixture
def update_llm_model_endpoint_request() -> UpdateLLMModelEndpointV1Request:
    return UpdateLLMModelEndpointV1Request(
        checkpoint_path="s3://test_checkpoint_path",
        memory="4G",
        min_workers=0,
        max_workers=1,
    )


@pytest.fixture
def create_llm_model_endpoint_request_llama_2() -> CreateLLMModelEndpointV1Request:
    return CreateLLMModelEndpointV1Request(
        name="test_llm_endpoint_name_llama_2",
        model_name="llama-2-7b",
        source="hugging_face",
        inference_framework="text_generation_inference",
        inference_framework_image_tag="0.9.4",
        num_shards=2,
        endpoint_type=ModelEndpointType.STREAMING,
        metadata={},
        post_inference_hooks=["billing"],
        cpus=1,
        gpus=2,
        memory="8G",
        gpu_type=GpuType.NVIDIA_TESLA_T4,
        storage=None,
        min_workers=1,
        max_workers=3,
        per_worker=2,
        labels={"team": "infra", "product": "my_product"},
        aws_role="test_aws_role",
        results_s3_bucket="test_s3_bucket",
    )


@pytest.fixture
def create_llm_model_endpoint_text_generation_inference_request_streaming() -> (
    CreateLLMModelEndpointV1Request
):
    return CreateLLMModelEndpointV1Request(
        name="test_llm_endpoint_name_tgi_streaming",
        model_name="mpt-7b",
        source="hugging_face",
        inference_framework="deepspeed",
        inference_framework_image_tag="test_tag",
        num_shards=2,
        endpoint_type=ModelEndpointType.STREAMING,
        metadata={},
        post_inference_hooks=["billing"],
        cpus=1,
        gpus=2,
        memory="8G",
        gpu_type=GpuType.NVIDIA_TESLA_T4,
        storage=None,
        min_workers=1,
        max_workers=3,
        per_worker=2,
        labels={"team": "infra", "product": "my_product"},
        aws_role="test_aws_role",
        results_s3_bucket="test_s3_bucket",
    )


@pytest.fixture
def create_llm_model_endpoint_text_generation_inference_request_async() -> (
    CreateLLMModelEndpointV1Request
):
    return CreateLLMModelEndpointV1Request(
        name="test_llm_endpoint_name_tgi_async",
        model_name="mpt-7b",
        source="hugging_face",
        inference_framework="text_generation_inference",
        inference_framework_image_tag="0.9.4",
        num_shards=2,
        quantize=Quantization.BITSANDBYTES,
        endpoint_type=ModelEndpointType.ASYNC,
        metadata={},
        post_inference_hooks=["billing"],
        cpus=1,
        gpus=2,
        memory="8G",
        gpu_type=GpuType.NVIDIA_TESLA_T4,
        storage=None,
        min_workers=1,
        max_workers=3,
        per_worker=2,
        labels={"team": "infra", "product": "my_product"},
        aws_role="test_aws_role",
        results_s3_bucket="test_s3_bucket",
    )


@pytest.fixture
def create_llm_model_endpoint_trt_llm_request_streaming() -> CreateLLMModelEndpointV1Request:
    return CreateLLMModelEndpointV1Request(
        name="test_llm_endpoint_name_trt_llm_streaming",
        model_name="llama-2-7b",
        source="hugging_face",
        inference_framework="tensorrt_llm",
        inference_framework_image_tag="23.10",
        num_shards=2,
        endpoint_type=ModelEndpointType.STREAMING,
        metadata={},
        post_inference_hooks=["billing"],
        cpus=1,
        gpus=2,
        memory="8G",
        gpu_type=GpuType.NVIDIA_TESLA_T4,
        storage=None,
        min_workers=1,
        max_workers=3,
        per_worker=2,
        labels={"team": "infra", "product": "my_product"},
        aws_role="test_aws_role",
        results_s3_bucket="test_s3_bucket",
        checkpoint_path="s3://test_checkpoint_path",
    )


@pytest.fixture
def create_llm_model_endpoint_trt_llm_request_async() -> CreateLLMModelEndpointV1Request:
    return CreateLLMModelEndpointV1Request(
        name="test_llm_endpoint_name_tgi_async",
        model_name="llama-2-7b",
        source="hugging_face",
        inference_framework="tensorrt_llm",
        inference_framework_image_tag="23.10",
        num_shards=2,
        quantize=Quantization.BITSANDBYTES,
        endpoint_type=ModelEndpointType.ASYNC,
        metadata={},
        post_inference_hooks=["billing"],
        cpus=1,
        gpus=2,
        memory="8G",
        gpu_type=GpuType.NVIDIA_TESLA_T4,
        storage=None,
        min_workers=1,
        max_workers=3,
        per_worker=2,
        labels={"team": "infra", "product": "my_product"},
        aws_role="test_aws_role",
        results_s3_bucket="test_s3_bucket",
        checkpoint_path="s3://test_checkpoint_path",
    )


@pytest.fixture
def create_llm_model_endpoint_request_invalid_model_name() -> CreateLLMModelEndpointV1Request:
    return CreateLLMModelEndpointV1Request(
        name="test_llm_endpoint_name_1",
        model_name="nonexist",
        source="hugging_face",
        inference_framework="deepspeed",
        inference_framework_image_tag="test_tag",
        num_shards=2,
        endpoint_type=ModelEndpointType.SYNC,
        metadata={},
        post_inference_hooks=["billing"],
        cpus=1,
        gpus=2,
        memory="8G",
        gpu_type=GpuType.NVIDIA_TESLA_T4,
        storage=None,
        min_workers=1,
        max_workers=3,
        per_worker=2,
        labels={"team": "infra", "product": "my_product"},
        aws_role="test_aws_role",
        results_s3_bucket="test_s3_bucket",
    )


@pytest.fixture
def create_llm_model_endpoint_request_invalid_quantization() -> CreateLLMModelEndpointV1Request:
    return CreateLLMModelEndpointV1Request(
        name="test_llm_endpoint_name_1",
        model_name="nonexist",
        source="hugging_face",
        inference_framework=LLMInferenceFramework.VLLM,
        inference_framework_image_tag="test_tag",
        num_shards=2,
        quantize=Quantization.BITSANDBYTES,
        endpoint_type=ModelEndpointType.SYNC,
        metadata={},
        post_inference_hooks=["billing"],
        cpus=1,
        gpus=2,
        memory="8G",
        gpu_type=GpuType.NVIDIA_TESLA_T4,
        storage=None,
        min_workers=1,
        max_workers=3,
        per_worker=2,
        labels={"team": "infra", "product": "my_product"},
        aws_role="test_aws_role",
        results_s3_bucket="test_s3_bucket",
    )


@pytest.fixture
def completion_sync_request() -> CompletionSyncV1Request:
    return CompletionSyncV1Request(
        prompt="What is machine learning?",
        max_new_tokens=10,
        temperature=0.5,
        return_token_log_probs=True,
    )


@pytest.fixture
def completion_stream_request() -> CompletionStreamV1Request:
    return CompletionStreamV1Request(
        prompt="What is machine learning?",
        max_new_tokens=10,
        temperature=0.5,
    )
