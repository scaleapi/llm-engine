import datetime
from typing import Any, Dict, Iterator, Tuple

import pytest
from fastapi import Depends, HTTPException
from fastapi.security import HTTPBasicCredentials
from fastapi.testclient import TestClient
from model_engine_server.api.app import app
from model_engine_server.api.dependencies import (
    AUTH,
    get_external_interfaces,
    get_external_interfaces_read_only,
    verify_authentication,
)
from model_engine_server.core.auth.authentication_repository import AuthenticationRepository, User
from model_engine_server.core.auth.fake_authentication_repository import (
    FakeAuthenticationRepository,
)
from model_engine_server.domain.entities import (
    BatchJob,
    BatchJobProgress,
    BatchJobRecord,
    BatchJobStatus,
    CallbackAuth,
    CallbackBasicAuth,
    CloudpickleArtifactFlavor,
    GpuType,
    ModelBundle,
    ModelBundleEnvironmentParams,
    ModelBundleFrameworkType,
    ModelBundlePackagingType,
    ModelEndpoint,
    ModelEndpointConfig,
    ModelEndpointDeploymentState,
    ModelEndpointInfraState,
    ModelEndpointRecord,
    ModelEndpointResourceState,
    ModelEndpointStatus,
    ModelEndpointType,
    ModelEndpointUserConfigState,
    PytorchFramework,
    StreamingEnhancedRunnableImageFlavor,
    TensorflowFramework,
    Trigger,
    ZipArtifactFlavor,
)
from model_engine_server.domain.entities.batch_job_entity import DockerImageBatchJob
from model_engine_server.domain.entities.docker_image_batch_job_bundle_entity import (
    DockerImageBatchJobBundle,
)

USER_TEAM_OVERRIDE = dict(
    test_user_id_on_other_team="test_team", test_user_id_on_other_team_2="test_team"
)


def get_test_auth_repository() -> Iterator[AuthenticationRepository]:
    try:
        yield FakeAuthenticationRepository(USER_TEAM_OVERRIDE)
    finally:
        pass


def fake_verify_authentication(
    credentials: HTTPBasicCredentials = Depends(AUTH),
    auth_repo: AuthenticationRepository = Depends(get_test_auth_repository),
) -> User:
    """
    Verifies the authentication headers and returns a (user_id, team_id) auth tuple. Otherwise,
    raises a 401.
    """
    auth_username = credentials.username if credentials is not None else None
    if not auth_username:
        raise HTTPException(status_code=401, detail="No authentication was passed in")

    auth = auth_repo.get_auth_from_username(username=auth_username)
    if not auth:
        raise HTTPException(status_code=401, detail="Could not authenticate user")

    return auth


@pytest.fixture(autouse=True)
def fake_auth():
    try:
        # It doesn't seem like FastAPI allows you to override sub-dependencies?
        app.dependency_overrides[verify_authentication] = fake_verify_authentication
        yield
    finally:
        app.dependency_overrides[verify_authentication] = {}


@pytest.fixture
def get_test_client_wrapper(get_repositories_generator_wrapper):
    def get_test_client(
        fake_docker_repository_image_always_exists=True,
        fake_model_bundle_repository_contents=None,
        fake_model_endpoint_record_repository_contents=None,
        fake_model_endpoint_infra_gateway_contents=None,
        fake_batch_job_record_repository_contents=None,
        fake_batch_job_progress_gateway_contents=None,
        fake_docker_image_batch_job_bundle_repository_contents=None,
        fake_docker_image_batch_job_gateway_contents=None,
        fake_llm_fine_tuning_service_contents=None,
        fake_file_storage_gateway_contents=None,
        fake_file_system_gateway_contents=None,
        fake_trigger_repository_contents=None,
        fake_cron_job_gateway_contents=None,
    ) -> TestClient:
        if fake_docker_image_batch_job_gateway_contents is None:
            fake_docker_image_batch_job_gateway_contents = {}
        if fake_docker_image_batch_job_bundle_repository_contents is None:
            fake_docker_image_batch_job_bundle_repository_contents = {}
        if fake_batch_job_progress_gateway_contents is None:
            fake_batch_job_progress_gateway_contents = {}
        if fake_batch_job_record_repository_contents is None:
            fake_batch_job_record_repository_contents = {}
        if fake_model_endpoint_infra_gateway_contents is None:
            fake_model_endpoint_infra_gateway_contents = {}
        if fake_model_endpoint_record_repository_contents is None:
            fake_model_endpoint_record_repository_contents = {}
        if fake_model_bundle_repository_contents is None:
            fake_model_bundle_repository_contents = {}
        if fake_llm_fine_tuning_service_contents is None:
            fake_llm_fine_tuning_service_contents = {}
        if fake_file_storage_gateway_contents is None:
            fake_file_storage_gateway_contents = {}
        if fake_file_system_gateway_contents is None:
            fake_file_system_gateway_contents = {}
        if fake_trigger_repository_contents is None:
            fake_trigger_repository_contents = {}
        if fake_cron_job_gateway_contents is None:
            fake_cron_job_gateway_contents = {}
        app.dependency_overrides[get_external_interfaces] = get_repositories_generator_wrapper(
            fake_docker_repository_image_always_exists=fake_docker_repository_image_always_exists,
            fake_model_bundle_repository_contents=fake_model_bundle_repository_contents,
            fake_model_endpoint_record_repository_contents=fake_model_endpoint_record_repository_contents,
            fake_model_endpoint_infra_gateway_contents=fake_model_endpoint_infra_gateway_contents,
            fake_batch_job_record_repository_contents=fake_batch_job_record_repository_contents,
            fake_batch_job_progress_gateway_contents=fake_batch_job_progress_gateway_contents,
            fake_docker_image_batch_job_bundle_repository_contents=fake_docker_image_batch_job_bundle_repository_contents,
            fake_docker_image_batch_job_gateway_contents=fake_docker_image_batch_job_gateway_contents,
            fake_llm_fine_tuning_service_contents=fake_llm_fine_tuning_service_contents,
            fake_file_storage_gateway_contents=fake_file_storage_gateway_contents,
            fake_file_system_gateway_contents=fake_file_system_gateway_contents,
            fake_trigger_repository_contents=fake_trigger_repository_contents,
            fake_cron_job_gateway_contents=fake_cron_job_gateway_contents,
        )
        app.dependency_overrides[get_external_interfaces_read_only] = app.dependency_overrides[
            get_external_interfaces
        ]
        client = TestClient(app)
        return client

    return get_test_client


@pytest.fixture
def simple_client(get_test_client_wrapper) -> TestClient:
    """Returns a Client with no initial contents and a Docker repository that always returns True"""
    client = get_test_client_wrapper(
        fake_docker_repository_image_always_exists=True,
        fake_model_bundle_repository_contents={},
        fake_model_endpoint_record_repository_contents={},
        fake_model_endpoint_infra_gateway_contents={},
        fake_batch_job_record_repository_contents={},
        fake_batch_job_progress_gateway_contents={},
        fake_docker_image_batch_job_bundle_repository_contents={},
        fake_trigger_repository_contents={},
    )
    return client


@pytest.fixture
def create_model_bundle_request_pytorch() -> Dict[str, Any]:
    return {
        "v1": {
            "name": "my_model_bundle_1",
            "location": "test_location",
            "requirements": ["numpy==0.0.0"],
            "env_params": {
                "framework_type": "pytorch",
                "pytorch_image_tag": "test_image_tag",
            },
            "packaging_type": "cloudpickle",
        },
        "v2": {
            "name": "my_model_bundle_1",
            "schema_location": "test_schema_location",
            "flavor": {
                "flavor": "cloudpickle_artifact",
                "location": "test_location",
                "requirements": ["numpy==0.0.0"],
                "load_predict_fn": "test_load_predict_fn",
                "load_model_fn": "test_load_model_fn",
                "framework": {
                    "framework_type": "pytorch",
                    "pytorch_image_tag": "test_image_tag",
                },
            },
        },
    }


@pytest.fixture
def create_model_bundle_request_custom() -> Dict[str, Any]:
    return {
        "v1": {
            "name": "my_model_bundle_2",
            "location": "test_location",
            "requirements": ["numpy==0.0.0"],
            "env_params": {
                "framework_type": "custom_base_image",
                "ecr_repo": "test_repo",
                "image_tag": "test_tag",
            },
            "packaging_type": "zip",
        },
        "v2": {
            "name": "my_model_bundle_2",
            "schema_location": "test_schema_location",
            "flavor": {
                "flavor": "zip_artifact",
                "location": "test_location",
                "requirements": ["numpy==0.0.0"],
                "load_predict_fn_module_path": "test_load_predict_fn_module_path",
                "load_model_fn_module_path": "test_load_model_fn_module_path",
                "framework": {
                    "framework_type": "custom_base_image",
                    "image_repository": "test_repo",
                    "image_tag": "test_tag",
                },
            },
        },
    }


@pytest.fixture
def model_bundle_1_v1(test_api_key: str) -> Tuple[ModelBundle, Any]:
    model_bundle = ModelBundle(
        id="test_model_bundle_id_1",
        name="test_model_bundle_name_1",
        created_by=test_api_key,
        owner=test_api_key,
        created_at=datetime.datetime(2022, 1, 1),
        model_artifact_ids=["test_model_artifact_id"],
        schema_location="test_schema_location",
        metadata={},
        flavor=CloudpickleArtifactFlavor(
            flavor="cloudpickle_artifact",
            framework=PytorchFramework(
                framework_type="pytorch",
                pytorch_image_tag="test_tag",
            ),
            requirements=["numpy==0.0.0"],
            location="test_location",
            app_config=None,
            load_predict_fn="test_load_predict_fn",
            load_model_fn="test_load_model_fn",
        ),
        # LEGACY FIELDS
        location="test_location",
        requirements=["numpy==0.0.0"],
        env_params=ModelBundleEnvironmentParams(
            framework_type=ModelBundleFrameworkType.PYTORCH, pytorch_image_tag="test_tag"
        ),
        packaging_type=ModelBundlePackagingType.CLOUDPICKLE,
        app_config=None,
    )
    model_bundle_json_v1: Dict[str, Any] = {
        "app_config": None,
        "metadata": {},
        "created_at": "2022-01-01T00:00:00",
        "env_params": {
            "ecr_repo": None,
            "framework_type": "pytorch",
            "image_tag": None,
            "pytorch_image_tag": "test_tag",
            "tensorflow_version": None,
        },
        "id": "test_model_bundle_id_1",
        "location": "test_location",
        "name": "test_model_bundle_name_1",
        "packaging_type": "cloudpickle",
        "requirements": ["numpy==0.0.0"],
        "model_artifact_ids": ["test_model_artifact_id"],
        "schema_location": "test_schema_location",
    }
    model_bundle_json_v2: Dict[str, Any] = {
        "metadata": {},
        "created_at": "2022-01-01T00:00:00",
        "flavor": {
            "app_config": None,
            "flavor": "cloudpickle_artifact",
            "framework": {
                "framework_type": "pytorch",
                "pytorch_image_tag": "test_tag",
            },
            "requirements": ["numpy==0.0.0"],
            "load_model_fn": "test_load_model_fn",
            "load_predict_fn": "test_load_predict_fn",
            "location": "test_location",
        },
        "id": "test_model_bundle_id_1",
        "name": "test_model_bundle_name_1",
        "model_artifact_ids": ["test_model_artifact_id"],
        "schema_location": "test_schema_location",
    }
    return model_bundle, {"v1": model_bundle_json_v1, "v2": model_bundle_json_v2}


@pytest.fixture
def model_bundle_1_v2(test_api_key: str) -> Tuple[ModelBundle, Any]:
    model_bundle = ModelBundle(
        id="test_model_bundle_id_2",
        name="test_model_bundle_name_1",
        created_by=test_api_key,
        owner=test_api_key,
        created_at=datetime.datetime(2022, 1, 2),
        model_artifact_ids=["test_model_artifact_id"],
        schema_location=None,
        metadata={},
        flavor=ZipArtifactFlavor(
            flavor="zip_artifact",
            framework=TensorflowFramework(
                framework_type="tensorflow",
                tensorflow_version="2.0.4",
            ),
            requirements=["numpy==0.0.0"],
            location="test_location",
            app_config=None,
            load_predict_fn_module_path="test_load_predict_fn_module_path",
            load_model_fn_module_path="test_load_model_fn_module_path",
        ),
        # LEGACY FIELDS
        location="test_location",
        requirements=["numpy==0.0.0"],
        env_params=ModelBundleEnvironmentParams(
            framework_type=ModelBundleFrameworkType.TENSORFLOW, tensorflow_version="2.0.4"
        ),
        packaging_type=ModelBundlePackagingType.ZIP,
        app_config=None,
    )
    model_bundle_json_v1: Dict[str, Any] = {
        "app_config": None,
        "metadata": {},
        "created_at": "2022-01-02T00:00:00",
        "env_params": {
            "ecr_repo": None,
            "framework_type": "tensorflow",
            "image_tag": None,
            "pytorch_image_tag": None,
            "tensorflow_version": "2.0.4",
        },
        "id": "test_model_bundle_id_2",
        "location": "test_location",
        "name": "test_model_bundle_name_1",
        "packaging_type": "zip",
        "requirements": ["numpy==0.0.0"],
        "model_artifact_ids": ["test_model_artifact_id"],
        "schema_location": None,
    }
    model_bundle_json_v2: Dict[str, Any] = {
        "metadata": {},
        "created_at": "2022-01-02T00:00:00",
        "flavor": {
            "app_config": None,
            "flavor": "zip_artifact",
            "framework": {
                "framework_type": "tensorflow",
                "tensorflow_version": "2.0.4",
            },
            "requirements": ["numpy==0.0.0"],
            "load_model_fn_module_path": "test_load_model_fn_module_path",
            "load_predict_fn_module_path": "test_load_predict_fn_module_path",
            "location": "test_location",
        },
        "id": "test_model_bundle_id_2",
        "name": "test_model_bundle_name_1",
        "model_artifact_ids": ["test_model_artifact_id"],
        "schema_location": None,
    }
    return model_bundle, {"v1": model_bundle_json_v1, "v2": model_bundle_json_v2}


@pytest.fixture
def model_bundle_2_v1(test_api_key: str) -> Tuple[ModelBundle, Any]:
    model_bundle = ModelBundle(
        id="test_model_bundle_id_3",
        name="test_model_bundle_name_2",
        created_by=test_api_key,
        owner=test_api_key,
        created_at=datetime.datetime(2022, 1, 3),
        model_artifact_ids=["test_model_artifact_id"],
        schema_location="test_schema_location",
        metadata={},
        flavor=CloudpickleArtifactFlavor(
            flavor="cloudpickle_artifact",
            framework=PytorchFramework(
                framework_type="pytorch",
                pytorch_image_tag="test_tag",
            ),
            requirements=["numpy==0.0.0"],
            location="test_location",
            app_config=None,
            load_predict_fn="test_load_predict_fn",
            load_model_fn="test_load_model_fn",
        ),
        # LEGACY FIELDS
        location="test_location",
        requirements=["numpy==0.0.0"],
        env_params=ModelBundleEnvironmentParams(
            framework_type=ModelBundleFrameworkType.PYTORCH, pytorch_image_tag="test_tag"
        ),
        packaging_type=ModelBundlePackagingType.CLOUDPICKLE,
        app_config=None,
    )
    model_bundle_json_v1: Dict[str, Any] = {
        "app_config": None,
        "metadata": {},
        "created_at": "2022-01-03T00:00:00",
        "env_params": {
            "ecr_repo": None,
            "framework_type": "pytorch",
            "image_tag": None,
            "pytorch_image_tag": "test_tag",
            "tensorflow_version": None,
        },
        "id": "test_model_bundle_id_3",
        "location": "test_location",
        "name": "test_model_bundle_name_2",
        "packaging_type": "cloudpickle",
        "requirements": ["numpy==0.0.0"],
        "model_artifact_ids": ["test_model_artifact_id"],
        "schema_location": "test_schema_location",
    }
    model_bundle_json_v2: Dict[str, Any] = {
        "metadata": {},
        "created_at": "2022-01-03T00:00:00",
        "flavor": {
            "app_config": None,
            "flavor": "cloudpickle_artifact",
            "framework": {
                "framework_type": "pytorch",
                "pytorch_image_tag": "test_tag",
            },
            "requirements": ["numpy==0.0.0"],
            "load_model_fn": "test_load_model_fn",
            "load_predict_fn": "test_load_predict_fn",
            "location": "test_location",
        },
        "id": "test_model_bundle_id_3",
        "name": "test_model_bundle_name_2",
        "model_artifact_ids": ["test_model_artifact_id"],
        "schema_location": "test_schema_location",
    }
    return model_bundle, {"v1": model_bundle_json_v1, "v2": model_bundle_json_v2}


@pytest.fixture
def model_bundle_3_v1(test_api_key: str) -> Tuple[ModelBundle, Any]:
    model_bundle = ModelBundle(
        id="test_model_bundle_id_4",
        name="test_model_bundle_name_3",
        created_by=test_api_key,
        owner=test_api_key,
        created_at=datetime.datetime(2022, 1, 3),
        model_artifact_ids=["test_model_artifact_id"],
        schema_location="test_schema_location",
        metadata={},
        flavor=StreamingEnhancedRunnableImageFlavor(
            flavor="streaming_enhanced_runnable_image",
            repository="test_repository",
            tag="test_tag",
            env={"test_env_key": "test_env_value"},
            protocol="http",
            streaming_command=["test_streaming_command"],
        ),
        # LEGACY FIELDS
        location="test_location",
        requirements=["numpy==0.0.0"],
        env_params=ModelBundleEnvironmentParams(
            framework_type=ModelBundleFrameworkType.PYTORCH, pytorch_image_tag="test_tag"
        ),
        packaging_type=ModelBundlePackagingType.CLOUDPICKLE,
        app_config=None,
    )
    model_bundle_json_v2: Dict[str, Any] = {
        "metadata": {},
        "created_at": "2022-01-03T00:00:00",
        "flavor": {
            "flavor": "streaming_enhanced_runnable_image",
            "repository": "test_repository",
            "tag": "test_tag",
            "command": [],
            "env": {"test_env_key": "test_env_value"},
            "protocol": "http",
            "streaming_command": ["test_streaming_command"],
        },
        "id": "test_model_bundle_id_3",
        "name": "test_model_bundle_name_2",
        "model_artifact_ids": ["test_model_artifact_id"],
        "schema_location": "test_schema_location",
    }
    return model_bundle, {"v2": model_bundle_json_v2}


@pytest.fixture
def create_model_endpoint_request_async(
    model_bundle_1_v1: Tuple[ModelBundle, Any]
) -> Dict[str, Any]:
    return {
        "name": "test_model_endpoint_name_1",
        "model_bundle_id": model_bundle_1_v1[0].id,
        "endpoint_type": "async",
        "metadata": {},
        "post_inference_hooks": ["callback"],
        "default_callback_url": "http://www.example.com",
        "default_callback_auth": {
            "kind": "basic",
            "username": "test_username",
            "password": "test_password",
        },
        "cpus": 1,
        "gpus": 1,
        "memory": "1G",
        "gpu_type": "nvidia-tesla-t4",
        "storage": None,
        "min_workers": 0,
        "max_workers": 5,
        "per_worker": 3,
        "labels": {"team": "infra", "product": "my_product"},
        "aws_role": "test_aws_role",
        "results_s3_bucket": "test_s3_bucket",
    }


@pytest.fixture
def create_model_endpoint_request_sync(
    model_bundle_1_v1: Tuple[ModelBundle, Any]
) -> Dict[str, Any]:
    return {
        "name": "test_model_endpoint_name_2",
        "model_bundle_id": model_bundle_1_v1[0].id,
        "endpoint_type": "sync",
        "metadata": {},
        "post_inference_hooks": None,
        "default_callback_url": None,
        "default_callback_auth": None,
        "cpus": 1,
        "gpus": 1,
        "memory": "1G",
        "gpu_type": "nvidia-ampere-a10",
        "storage": None,
        "min_workers": 1,
        "max_workers": 5,
        "per_worker": 3,
        "labels": {"team": "infra", "product": "my_product"},
        "aws_role": "test_aws_role",
        "results_s3_bucket": "test_s3_bucket",
    }


@pytest.fixture
def create_model_endpoint_request_streaming(
    model_bundle_3_v1: Tuple[ModelBundle, Any]
) -> Dict[str, Any]:
    return {
        "name": "test_model_endpoint_name_2",
        "model_bundle_id": model_bundle_3_v1[0].id,
        "endpoint_type": "streaming",
        "metadata": {},
        "post_inference_hooks": None,
        "default_callback_url": None,
        "default_callback_auth": None,
        "cpus": 1,
        "gpus": 1,
        "memory": "1G",
        "gpu_type": "nvidia-ampere-a10",
        "storage": None,
        "min_workers": 1,
        "max_workers": 5,
        "per_worker": 1,
        "labels": {"team": "infra", "product": "my_product"},
        "aws_role": "test_aws_role",
        "results_s3_bucket": "test_s3_bucket",
    }


@pytest.fixture
def create_model_endpoint_request_streaming_invalid_bundle(
    model_bundle_1_v1: Tuple[ModelBundle, Any]
) -> Dict[str, Any]:
    return {
        "name": "test_model_endpoint_name_2",
        "model_bundle_id": model_bundle_1_v1[0].id,
        "endpoint_type": "streaming",
        "metadata": {},
        "post_inference_hooks": None,
        "default_callback_url": None,
        "default_callback_auth": None,
        "cpus": 1,
        "gpus": 1,
        "memory": "1G",
        "gpu_type": "nvidia-ampere-a10",
        "storage": None,
        "min_workers": 1,
        "max_workers": 5,
        "per_worker": 1,
        "labels": {"team": "infra", "product": "my_product"},
        "aws_role": "test_aws_role",
        "results_s3_bucket": "test_s3_bucket",
    }


@pytest.fixture
def create_model_endpoint_request_sync_invalid_streaming_bundle(
    model_bundle_3_v1: Tuple[ModelBundle, Any]
) -> Dict[str, Any]:
    return {
        "name": "test_model_endpoint_name_2",
        "model_bundle_id": model_bundle_3_v1[0].id,
        "endpoint_type": "sync",
        "metadata": {},
        "post_inference_hooks": None,
        "default_callback_url": None,
        "default_callback_auth": None,
        "cpus": 1,
        "gpus": 1,
        "memory": "1G",
        "gpu_type": "nvidia-ampere-a10",
        "storage": None,
        "min_workers": 1,
        "max_workers": 5,
        "per_worker": 1,
        "labels": {"team": "infra", "product": "my_product"},
        "aws_role": "test_aws_role",
        "results_s3_bucket": "test_s3_bucket",
    }


@pytest.fixture
def update_model_endpoint_request(
    model_endpoint_1: Tuple[ModelEndpoint, Any],
    model_bundle_1_v1: Tuple[ModelBundle, Any],
) -> Dict[str, Any]:
    return {
        "model_endpoint_id": model_endpoint_1[0].record.id,
        "model_bundle_id": model_bundle_1_v1[0].id,
        "metadata": {"test_new_metadata_key": "test_new_metadata_value"},
        "cpus": 3,
        "min_workers": 3,
        "max_workers": 8,
    }


@pytest.fixture
def model_endpoint_1(
    test_api_key: str, model_bundle_1_v1: Tuple[ModelBundle, Any]
) -> Tuple[ModelEndpoint, Any]:
    model_endpoint = ModelEndpoint(
        record=ModelEndpointRecord(
            id="test_model_endpoint_id_1",
            name="test_model_endpoint_name_1",
            created_by=test_api_key,
            created_at=datetime.datetime(2022, 1, 3),
            last_updated_at=datetime.datetime(2022, 1, 3),
            metadata={},
            creation_task_id="test_creation_task_id",
            endpoint_type=ModelEndpointType.ASYNC,
            destination="test_destination",
            status=ModelEndpointStatus.READY,
            current_model_bundle=model_bundle_1_v1[0],
            owner=test_api_key,
            public_inference=False,
        ),
        infra_state=ModelEndpointInfraState(
            deployment_name=f"{test_api_key}-test_model_endpoint_name_1",
            aws_role="test_aws_role",
            results_s3_bucket="test_s3_bucket",
            child_fn_info=None,
            labels={},
            prewarm=True,
            high_priority=False,
            deployment_state=ModelEndpointDeploymentState(
                min_workers=1,
                max_workers=3,
                per_worker=2,
                available_workers=1,
                unavailable_workers=1,
            ),
            resource_state=ModelEndpointResourceState(
                cpus=1,
                gpus=1,
                memory="1G",
                gpu_type=GpuType.NVIDIA_TESLA_T4,
                storage="10G",
                optimize_costs=True,
            ),
            user_config_state=ModelEndpointUserConfigState(
                app_config=model_bundle_1_v1[0].app_config,
                endpoint_config=ModelEndpointConfig(
                    bundle_name=model_bundle_1_v1[0].name,
                    endpoint_name="test_model_endpoint_name_1",
                    post_inference_hooks=["callback"],
                    default_callback_url="http://www.example.com",
                    default_callback_auth=CallbackAuth(
                        __root__=CallbackBasicAuth(
                            kind="basic",
                            username="test_username",
                            password="test_password",
                        ),
                    ),
                ),
            ),
            num_queued_items=1,
            image="test_image",
        ),
    )
    model_endpoint_json: Dict[str, Any] = {
        "id": "test_model_endpoint_id_1",
        "name": "test_model_endpoint_name_1",
        "endpoint_type": "async",
        "destination": "test_destination",
        "deployment_name": f"{test_api_key}-test_model_endpoint_name_1",
        "metadata": {},
        "bundle_name": "test_model_bundle_name_1",
        "status": "READY",
        "post_inference_hooks": ["callback"],
        "default_callback_url": "http://www.example.com",
        "default_callback_auth": {
            "kind": "basic",
            "username": "test_username",
            "password": "test_password",
        },
        "labels": {},
        "aws_role": "test_aws_role",
        "results_s3_bucket": "test_s3_bucket",
        "created_by": test_api_key,
        "created_at": "2022-01-03T00:00:00",
        "last_updated_at": "2022-01-03T00:00:00",
        "deployment_state": {
            "min_workers": 1,
            "max_workers": 3,
            "per_worker": 2,
            "available_workers": 1,
            "unavailable_workers": 1,
        },
        "resource_state": {
            "cpus": "1",
            "gpus": 1,
            "memory": "1G",
            "gpu_type": "nvidia-tesla-t4",
            "storage": "10G",
            "optimize_costs": True,
        },
        "num_queued_items": 1,
        "public_inference": False,
    }
    return model_endpoint, model_endpoint_json


@pytest.fixture
def model_endpoint_2(
    test_api_key: str, model_bundle_1_v1: Tuple[ModelBundle, Any]
) -> Tuple[ModelEndpoint, Any]:
    model_endpoint = ModelEndpoint(
        record=ModelEndpointRecord(
            id="test_model_endpoint_id_2",
            name="test_model_endpoint_name_2",
            created_by=test_api_key,
            created_at=datetime.datetime(2022, 1, 3),
            last_updated_at=datetime.datetime(2022, 1, 3),
            metadata={},
            creation_task_id="test_creation_task_id",
            endpoint_type=ModelEndpointType.SYNC,
            destination="test_destination",
            status=ModelEndpointStatus.READY,
            current_model_bundle=model_bundle_1_v1[0],
            owner=test_api_key,
        ),
        infra_state=ModelEndpointInfraState(
            deployment_name=f"{test_api_key}-test_model_endpoint_name_2",
            aws_role="test_aws_role",
            results_s3_bucket="test_s3_bucket",
            child_fn_info=None,
            labels={},
            prewarm=False,
            high_priority=True,
            deployment_state=ModelEndpointDeploymentState(
                min_workers=1,
                max_workers=3,
                per_worker=2,
                available_workers=2,
                unavailable_workers=1,
            ),
            resource_state=ModelEndpointResourceState(
                cpus=1,
                gpus=1,
                memory="1G",
                gpu_type=GpuType.NVIDIA_TESLA_T4,
                storage="10G",
                optimize_costs=False,
            ),
            user_config_state=ModelEndpointUserConfigState(
                app_config=model_bundle_1_v1[0].app_config,
                endpoint_config=ModelEndpointConfig(
                    bundle_name=model_bundle_1_v1[0].name,
                    endpoint_name="test_model_endpoint_name_2",
                    post_inference_hooks=None,
                    default_callback_url=None,
                    default_callback_auth=None,
                    billing_tags={
                        "idempotencyKeyPrefix": "value1",
                        "product": "value2",
                        "type": "hi",
                        "subType": "hi",
                        "tags": {"nested_tag_1": "nested_value_1"},
                        "payee": "hi",
                        "payor": "hi",
                        "reference": {"referenceType": "hi", "referenceId": "hi"},
                    },
                    user_id=test_api_key,
                    billing_queue="some:arn:for:something",
                ),
            ),
            image="test_image_2",
        ),
    )
    model_endpoint_json: Dict[str, Any] = {
        "id": "test_model_endpoint_id_2",
        "name": "test_model_endpoint_name_2",
        "endpoint_type": "sync",
        "destination": "test_destination",
        "deployment_name": f"{test_api_key}-test_model_endpoint_name_2",
        "metadata": {},
        "bundle_name": "test_model_bundle_name_1",
        "status": "READY",
        "post_inference_hooks": None,
        "default_callback_url": None,
        "default_callback_auth": None,
        "labels": {},
        "aws_role": "test_aws_role",
        "results_s3_bucket": "test_s3_bucket",
        "created_by": test_api_key,
        "created_at": "2022-01-03T00:00:00",
        "last_updated_at": "2022-01-03T00:00:00",
        "deployment_state": {
            "min_workers": 1,
            "max_workers": 3,
            "per_worker": 2,
            "available_workers": 2,
            "unavailable_workers": 1,
        },
        "resource_state": {
            "cpus": 1,
            "gpus": 1,
            "memory": "1G",
            "gpu_type": "nvidia-tesla-t4",
            "storage": "10G",
            "optimize_costs": False,
        },
        "image": "test_image_2",
    }
    return model_endpoint, model_endpoint_json


@pytest.fixture
def create_batch_job_request() -> Dict[str, Any]:
    return {
        "model_bundle_id": "test_model_bundle_id_1",
        "input_path": "test_input_path",
        "serialization_format": "JSON",
        "labels": {"team": "infra", "product": "test_product"},
        "resource_requests": {
            "cpus": 2,
            "memory": "2G",
            "gpus": 1,
            "gpu_type": "nvidia-tesla-t4",
            "storage": "10G",
            "max_worker": 4,
            "per_worker": 10,
        },
    }


@pytest.fixture
def batch_job_1(
    test_api_key: str,
    model_bundle_1_v1: Tuple[ModelBundle, Any],
    model_endpoint_1: Tuple[ModelEndpoint, Any],
) -> Tuple[BatchJob, Any]:
    batch_job = BatchJob(
        record=BatchJobRecord(
            id="test_batch_job_id_1",
            created_at=datetime.datetime(2022, 1, 1),
            status=BatchJobStatus.RUNNING,
            created_by=test_api_key,
            owner=test_api_key,
            model_bundle=model_bundle_1_v1[0],
            model_endpoint_id=model_endpoint_1[0].record.id,
            task_ids_location="test_task_ids_location",
            result_location=None,
        ),
        model_endpoint=model_endpoint_1[0],
        progress=BatchJobProgress(
            num_tasks_pending=4,
            num_tasks_completed=3,
        ),
    )
    batch_job_json: Dict[str, Any] = {
        "status": "RUNNING",
        "duration": None,
        "result": None,
        "num_tasks_pending": 4,
        "num_tasks_completed": 3,
    }
    return batch_job, batch_job_json


@pytest.fixture
def batch_job_2(
    test_api_key: str,
    model_bundle_1_v1: Tuple[ModelBundle, Any],
    model_endpoint_1: Tuple[ModelEndpoint, Any],
) -> Tuple[BatchJob, Any]:
    batch_job = BatchJob(
        record=BatchJobRecord(
            id="test_batch_job_id_2",
            created_at=datetime.datetime(2022, 1, 1),
            completed_at=datetime.datetime(2022, 1, 2),
            status=BatchJobStatus.SUCCESS,
            created_by=test_api_key,
            owner=test_api_key,
            model_bundle=model_bundle_1_v1[0],
            task_ids_location="test_task_ids_location",
            result_location="test_result_location",
        ),
        progress=BatchJobProgress(
            num_tasks_pending=0,
            num_tasks_completed=8,
        ),
    )
    batch_job_json: Dict[str, Any] = {
        "status": "RUNNING",
        "result": "test_result_location",
        "duration": "1 day, 0:00:00",
        "num_tasks_pending": 0,
        "num_tasks_completed": 8,
    }
    return batch_job, batch_job_json


@pytest.fixture
def create_docker_image_batch_job_bundle_request() -> Dict[str, Any]:
    return dict(
        name="test_docker_image_batch_job_bundle_1",
        image_repository="image_repository",
        image_tag="image_tag_git_sha",
        command=["python", "script.py", "--arg1"],
        env=dict(ENV1="VAL1", ENV2="VAL2"),
        mount_location="/mount/location/to/config",
        resource_requests=dict(cpus=1),
    )


@pytest.fixture
def docker_image_batch_job_bundle_1_v1(test_api_key) -> Tuple[DockerImageBatchJobBundle, Any]:
    batch_bundle = DockerImageBatchJobBundle(
        id="test_docker_image_batch_job_bundle_id_11",
        created_at=datetime.datetime(2022, 1, 1),
        name="test_docker_image_batch_job_bundle_1",
        created_by=test_api_key,
        owner=test_api_key,
        image_repository="image_repository",
        image_tag="image_tag_git_sha",
        command=["python", "script.py", "--arg1"],
        env=dict(ENV1="VAL1", ENV2="VAL2"),
        mount_location="/mount/location/to/config",
        cpus="1",
        memory=None,
        storage=None,
        gpus=None,
        gpu_type=None,
        public=False,
    )
    batch_bundle_json = {
        "id": "test_docker_image_batch_job_bundle_id_11",
        "name": "test_docker_image_batch_job_bundle_1",
        "created_at": "2022-01-01T00:00:00",
        "image_repository": "image_repository",
        "image_tag": "image_tag_git_sha",
        "command": ["python", "script.py", "--arg1"],
        "env": {"ENV1": "VAL1", "ENV2": "VAL2"},
        "mount_location": "/mount/location/to/config",
        "cpus": "1",
        "memory": None,
        "storage": None,
        "gpus": None,
        "gpu_type": None,
        "public": False,
    }
    return batch_bundle, batch_bundle_json


@pytest.fixture
def docker_image_batch_job_bundle_1_v2(test_api_key) -> Tuple[DockerImageBatchJobBundle, Any]:
    batch_bundle = DockerImageBatchJobBundle(
        id="test_docker_image_batch_job_bundle_id_12",
        created_at=datetime.datetime(2022, 1, 3),
        name="test_docker_image_batch_job_bundle_1",
        created_by=test_api_key,
        owner=test_api_key,
        image_repository="image_repository",
        image_tag="image_tag_git_sha",
        command=["python", "script.py", "--arg2"],
        env=dict(ENV1="VAL3", ENV2="VAL4"),
        mount_location="/mount/location/to/config2",
        cpus="2",
        memory=None,
        storage=None,
        gpus=None,
        gpu_type=None,
        public=True,
    )
    batch_bundle_json = {
        "id": "test_docker_image_batch_job_bundle_id_12",
        "name": "test_docker_image_batch_job_bundle_1",
        "created_at": "2022-01-03T00:00:00",
        "image_repository": "image_repository",
        "image_tag": "image_tag_git_sha",
        "command": ["python", "script.py", "--arg2"],
        "env": {"ENV1": "VAL3", "ENV2": "VAL4"},
        "mount_location": "/mount/location/to/config2",
        "cpus": "2",
        "memory": None,
        "storage": None,
        "gpus": None,
        "gpu_type": None,
        "public": True,
    }
    return batch_bundle, batch_bundle_json


@pytest.fixture
def docker_image_batch_job_bundle_2_v1(test_api_key) -> Tuple[DockerImageBatchJobBundle, Any]:
    batch_bundle = DockerImageBatchJobBundle(
        id="test_docker_image_batch_job_bundle_id_21",
        created_at=datetime.datetime(2022, 1, 2),
        name="test_docker_image_batch_job_bundle_2",
        created_by=test_api_key,
        owner=test_api_key,
        image_repository="image_repository",
        image_tag="image_tag_git_sha",
        command=["python", "script2.py", "--arg1"],
        env=dict(ENV1="VAL1", ENV2="VAL2"),
        mount_location="/mount2/location/to/config",
        cpus="3",
        memory=None,
        storage=None,
        gpus=None,
        gpu_type=None,
        public=None,
    )
    batch_bundle_json = {
        "id": "test_docker_image_batch_job_bundle_id_21",
        "name": "test_docker_image_batch_job_bundle_2",
        "created_at": "2022-01-02T00:00:00",
        "image_repository": "image_repository",
        "image_tag": "image_tag_git_sha",
        "command": ["python", "script2.py", "--arg1"],
        "env": {"ENV1": "VAL1", "ENV2": "VAL2"},
        "mount_location": "/mount2/location/to/config",
        "cpus": "3",
        "memory": None,
        "storage": None,
        "gpus": None,
        "gpu_type": None,
        "public": None,
    }
    return batch_bundle, batch_bundle_json


@pytest.fixture
def docker_image_batch_job_bundle_3_v1(test_api_key) -> Tuple[DockerImageBatchJobBundle, Any]:
    batch_bundle = DockerImageBatchJobBundle(
        id="test_docker_image_batch_job_bundle_id_31",
        created_at=datetime.datetime(2022, 1, 2),
        name="test_docker_image_batch_job_bundle_3",
        created_by=test_api_key,
        owner=test_api_key,
        image_repository="image_repository",
        image_tag="image_tag_git_sha",
        command=["python", "script3.py", "--arg1"],
        env=dict(ENV1="VAL1", ENV2="VAL2"),
        mount_location="/mount2/location/to/config",
        cpus="3",
        memory="5G",
        storage="5G",
        gpus=None,
        gpu_type=None,
        public=None,
    )
    batch_bundle_json = {
        "id": "test_docker_image_batch_job_bundle_id_31",
        "name": "test_docker_image_batch_job_bundle_3",
        "created_at": "2022-01-02T00:00:00",
        "image_repository": "image_repository",
        "image_tag": "image_tag_git_sha",
        "command": ["python", "script3.py", "--arg1"],
        "env": {"ENV1": "VAL1", "ENV2": "VAL2"},
        "mount_location": "/mount2/location/to/config",
        "cpus": "3",
        "memory": "5G",
        "storage": "5G",
        "gpus": None,
        "gpu_type": None,
        "public": None,
    }
    return batch_bundle, batch_bundle_json


@pytest.fixture
def create_docker_image_batch_job_request() -> Dict[str, Any]:
    return dict(
        docker_image_batch_job_bundle_name="test_docker_image_batch_job_bundle_1",
        job_config={"some": {"job": "input"}, "i": "dk"},
        labels=dict(team="infra", product="testing"),
        resource_requests=dict(
            cpus=0.1,
            gpus=0,
            memory="100Mi",
            storage="100Mi",
        ),
    )


@pytest.fixture
def docker_image_batch_job_1(test_api_key) -> Tuple[DockerImageBatchJob, Any]:
    dibatch_job = DockerImageBatchJob(
        id="batch-job-id-abcdef123456",
        created_by=test_api_key,
        owner=test_api_key,
        created_at=datetime.datetime(2022, 1, 4),
        completed_at=None,
        status=BatchJobStatus.RUNNING,
    )
    dibatch_job_json = {
        "status": "RUNNING",
    }
    return dibatch_job, dibatch_job_json


@pytest.fixture
def create_llm_model_endpoint_request_sync() -> Dict[str, Any]:
    return {
        "name": "test_llm_model_endpoint_name_1",
        "model_name": "mpt-7b",
        "source": "hugging_face",
        "inference_framework": "deepspeed",
        "inference_framework_image_tag": "0.0.1",
        "num_shards": 2,
        "endpoint_type": "sync",
        "metadata": {},
        "post_inference_hooks": ["callback"],
        "default_callback_url": "http://www.example.com",
        "default_callback_auth": {
            "kind": "basic",
            "username": "test_username",
            "password": "test_password",
        },
        "cpus": 1,
        "gpus": 2,
        "memory": "1G",
        "gpu_type": "nvidia-tesla-t4",
        "storage": None,
        "min_workers": 1,
        "max_workers": 5,
        "per_worker": 3,
        "labels": {"team": "infra", "product": "my_product"},
        "aws_role": "test_aws_role",
    }


@pytest.fixture
def completion_sync_request() -> Dict[str, Any]:
    return {
        "prompt": "what is 1+1?",
        "max_new_tokens": 10,
        "temperature": 0.1,
    }


@pytest.fixture
def completion_stream_request() -> Dict[str, Any]:
    return {"prompt": "what is 1+1?", "max_new_tokens": 10, "temperature": 0.1}


@pytest.fixture
def create_trigger_request() -> Dict[str, Any]:
    return dict(
        name="test_trigger_1",
        cron_schedule="* * * * *",
        bundle_id="test_docker_image_batch_job_bundle_id_31",
        default_job_config={},
        default_job_metadata=dict(team="infra", product="my_product"),
    )


@pytest.fixture
def update_trigger_request() -> Dict[str, Any]:
    return dict(cron_schedule="0 * * * *", suspend=True)


@pytest.fixture
def trigger_1(test_api_key) -> Tuple[Trigger, Any]:
    trigger = Trigger(
        id="test_trigger_id_1",
        name="test_trigger_1",
        owner=test_api_key,
        created_by=test_api_key,
        created_at=datetime.datetime(2022, 1, 2),
        cron_schedule="* * * * *",
        docker_image_batch_job_bundle_id="test_docker_image_batch_job_bundle_id_11",
        default_job_config={},
        default_job_metadata=dict(team="infra", product="my_product_one"),
    )
    trigger_json = {
        "id": "test_trigger_id_1",
        "name": "test_trigger_1",
        "owner": "test_user_id",
        "created_by": "test_user_id",
        "created_at": "2022-01-02T00:00:00",
        "cron_schedule": "* * * * *",
        "docker_image_batch_job_bundle_id": "test_docker_image_batch_job_bundle_id_11",
        "default_job_config": {},
        "default_job_metadata": {"team": "infra", "product": "my_product_one"},
    }
    return trigger, trigger_json


@pytest.fixture
def trigger_2(test_api_key) -> Tuple[Trigger, Any]:
    trigger = Trigger(
        id="test_trigger_id_2",
        name="test_trigger_2",
        owner=test_api_key,
        created_by=test_api_key,
        created_at=datetime.datetime(2022, 2, 2),
        cron_schedule="0 * * * *",
        docker_image_batch_job_bundle_id="test_docker_image_batch_job_bundle_id_12",
        default_job_config={},
        default_job_metadata=dict(team="infra", product="my_product_two"),
    )
    trigger_json = {
        "id": "test_trigger_id_2",
        "name": "test_trigger_2",
        "owner": "test_user_id",
        "created_by": "test_user_id",
        "created_at": "2022-02-02T00:00:00",
        "cron_schedule": "0 * * * *",
        "docker_image_batch_job_bundle_id": "test_docker_image_batch_job_bundle_id_12",
        "default_job_config": {},
        "default_job_metadata": {"team": "infra", "product": "my_product_two"},
    }
    return trigger, trigger_json
