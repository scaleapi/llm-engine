import datetime
from typing import Callable, Optional, Union

import pytest
from model_engine_server.db.models import BatchJob, Bundle
from model_engine_server.db.models import DockerImageBatchJobBundle as OrmDockerImageBatchJobBundle
from model_engine_server.db.models import Endpoint
from model_engine_server.domain.entities import (
    BatchJobRecord,
    GpuType,
    ModelBundle,
    ModelEndpointDeploymentState,
    ModelEndpointInfraState,
    ModelEndpointRecord,
    ModelEndpointResourceState,
    ModelEndpointStatus,
    ModelEndpointType,
    ModelEndpointUserConfigState,
)
from sqlalchemy.ext.asyncio import AsyncSession


class FakeRedis:
    def __init__(self):
        self.db = {}

    async def set(self, key: str, value: Union[str, bytes, int, float], ex: float = None):
        if not (
            isinstance(value, str)
            or isinstance(value, bytes)
            or isinstance(value, int)
            or isinstance(value, float)
        ):
            raise TypeError(
                f"value must of type str, bytes, int, or float, got {value=}, type={type(value)}"
            )
        self.db[key] = str(value).encode()

    async def get(self, key: str) -> Optional[bytes]:
        return self.db.get(key, None)

    def force_expire_all(self):
        self.db = {}

    async def close(self):
        pass


@pytest.fixture
def fake_redis():
    return FakeRedis()


@pytest.fixture
def orm_model_bundle(test_api_key: str) -> Bundle:
    model_bundle = Bundle(
        name="test_model_bundle_name_1",
        created_by=test_api_key,
        owner=test_api_key,
        model_artifact_ids=["test_model_artifact_id"],
        bundle_metadata={},
        flavor="cloudpickle_artifact",
        artifact_requirements=["numpy==0.0.0"],
        artifact_location="test_location",
        artifact_app_config=None,
        artifact_framework_type="pytorch",
        artifact_pytorch_image_tag="test_tag",
        cloudpickle_artifact_load_predict_fn="test_load_predict_fn",
        cloudpickle_artifact_load_model_fn="test_load_model_fn",
        # LEGACY FIELDS
        location="test_location",
        requirements=["numpy==0.0.0"],
        env_params={"framework_type": "pytorch", "pytorch_image_tag": "test_tag"},
        packaging_type="cloudpickle",
        app_config=None,
    )
    model_bundle.id = "test_model_bundle_id_1"
    model_bundle.created_at = datetime.datetime(2022, 1, 1)
    return model_bundle


@pytest.fixture
def orm_model_bundle_2(test_api_key: str) -> Bundle:
    model_bundle = Bundle(
        name="test_model_bundle_name_2",
        created_by=test_api_key,
        owner=test_api_key,
        model_artifact_ids=["test_model_artifact_id"],
        bundle_metadata={},
        flavor="cloudpickle_artifact",
        artifact_requirements=["numpy==0.0.0"],
        artifact_location="test_location",
        artifact_app_config=None,
        artifact_framework_type="tensorflow",
        artifact_tensorflow_version="0.0.0",
        cloudpickle_artifact_load_predict_fn="test_load_predict_fn",
        cloudpickle_artifact_load_model_fn="test_load_model_fn",
        # LEGACY FIELDS
        location="test_location",
        requirements=["numpy==0.0.0"],
        env_params={"framework_type": "tensorflow", "tensorflow_version": "0.0.0"},
        packaging_type="cloudpickle",
        app_config=None,
    )
    model_bundle.id = "test_model_bundle_id_2"
    model_bundle.created_at = datetime.datetime(2022, 1, 2)
    return model_bundle


@pytest.fixture
def orm_model_bundle_3(test_api_key: str) -> Bundle:
    model_bundle = Bundle(
        name="test_model_bundle_name_3",
        created_by=test_api_key,
        owner=test_api_key,
        model_artifact_ids=["test_model_artifact_id"],
        bundle_metadata={
            "load_predict_fn_module_path": "test_load_predict_fn_module_path",
            "load_model_fn_module_path": "test_load_model_fn_module_path",
        },
        flavor="zip_artifact",
        artifact_requirements=["numpy==0.0.0"],
        artifact_location="test_location",
        artifact_app_config=None,
        artifact_framework_type="custom_base_image",
        artifact_image_repository="test_repo",
        artifact_image_tag="test_tag",
        zip_artifact_load_predict_fn_module_path="test_load_predict_fn_module_path",
        zip_artifact_load_model_fn_module_path="test_load_model_fn_module_path",
        # LEGACY FIELDS
        location="test_location",
        requirements=["numpy==0.0.0"],
        env_params={
            "framework_type": "custom_base_image",
            "ecr_repo": "test_repo",
            "image_tag": "test_tag",
        },
        packaging_type="zip",
        app_config=None,
    )
    model_bundle.id = "test_model_bundle_id_3"
    model_bundle.created_at = datetime.datetime(2022, 1, 2)
    return model_bundle


@pytest.fixture
def orm_model_bundle_4(test_api_key: str) -> Bundle:
    model_bundle = Bundle(
        name="test_model_bundle_name_4",
        created_by=test_api_key,
        owner=test_api_key,
        model_artifact_ids=["test_model_artifact_id_4"],
        bundle_metadata={
            "test_key_2": "test_value_2",
        },
        flavor="runnable_image",
        runnable_image_readiness_initial_delay_seconds=30,
        runnable_image_tag="test_tag",
        runnable_image_repository="test_repo",
        runnable_image_env={"test_key": "test_value"},
        runnable_image_command=["test_command"],
        runnable_image_predict_route="/test_predict_route",
        runnable_image_healthcheck_route="/test_healthcheck_route",
        runnable_image_protocol="http",
        # LEGACY FIELDS
        location="test_location",
        requirements=["numpy==0.0.0"],
        env_params={
            "framework_type": "custom_base_image",
            "ecr_repo": "test_repo",
            "image_tag": "test_tag",
        },
        packaging_type="lira",
        app_config=None,
    )
    model_bundle.id = "test_model_bundle_id_4"
    model_bundle.created_at = datetime.datetime(2022, 1, 3)
    return model_bundle


@pytest.fixture
def orm_model_bundle_5(test_api_key: str) -> Bundle:
    model_bundle = Bundle(
        name="test_model_bundle_name_5",
        created_by=test_api_key,
        owner=test_api_key,
        model_artifact_ids=["test_model_artifact_id_5"],
        bundle_metadata={
            "test_key_2": "test_value_2",
        },
        flavor="streaming_enhanced_runnable_image",
        runnable_image_readiness_initial_delay_seconds=30,
        runnable_image_tag="test_tag",
        runnable_image_repository="test_repo",
        runnable_image_env={"test_key": "test_value"},
        runnable_image_command=["test_command"],
        runnable_image_protocol="http",
        streaming_enhanced_runnable_image_streaming_command=["test_streaming_command"],
        streaming_enhanced_runnable_image_streaming_predict_route="/test_streaming_predict_route",
        # LEGACY FIELDS
        location="test_location",
        requirements=["numpy==0.0.0"],
        env_params={
            "framework_type": "custom_base_image",
            "ecr_repo": "test_repo",
            "image_tag": "test_tag",
        },
        packaging_type="lira",
        app_config=None,
    )
    model_bundle.id = "test_model_bundle_id_5"
    model_bundle.created_at = datetime.datetime(2022, 1, 3)
    return model_bundle


@pytest.fixture
def orm_model_endpoint(orm_model_bundle: Bundle) -> Endpoint:
    model_endpoint = Endpoint(
        name="test_model_endpoint_name_1",
        created_by="test_user_id",
        current_bundle_id=orm_model_bundle.id,
        endpoint_metadata={},
        creation_task_id="test_creation_task_id",
        endpoint_type="async",
        destination="test_destination",
        endpoint_status="READY",
        owner="test_user_id",
        public_inference=True,
    )
    model_endpoint.id = "test_model_endpoint_id_1"
    model_endpoint.created_at = datetime.datetime(2022, 1, 3)
    return model_endpoint


@pytest.fixture
def entity_model_endpoint_record(model_bundle_1: ModelBundle) -> ModelEndpointRecord:
    model_endpoint = ModelEndpointRecord(
        id="test_model_endpoint_id_1",
        name="test_model_endpoint_name_1",
        created_by="test_user_id",
        created_at=datetime.datetime(2022, 1, 3),
        last_updated_at=datetime.datetime(2022, 1, 3),
        metadata={},
        creation_task_id="test_creation_task_id",
        endpoint_type=ModelEndpointType.ASYNC,
        destination="test_destination",
        status=ModelEndpointStatus.READY,
        current_model_bundle=model_bundle_1,
        owner="test_user_id",
        public_inference=True,
    )
    return model_endpoint


@pytest.fixture
def entity_model_endpoint_infra_state() -> ModelEndpointInfraState:
    infra_state = ModelEndpointInfraState(
        deployment_name="some-name-test_model_endpoint_name_1",
        aws_role="some-aws-role",
        results_s3_bucket="test_s3_bucket",
        child_fn_info=None,
        post_inference_hooks=None,
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
            nodes_per_worker=1,
            optimize_costs=False,
        ),
        user_config_state=ModelEndpointUserConfigState(
            app_config={"app": "config"},
            endpoint_config={
                "bundle_name": "some-bundle-name",
                "endpoint_name": "some-endpoint-name",
                "post_inference_hooks": None,
            },
        ),
        image="test_image",
    )
    return infra_state


@pytest.fixture
def orm_batch_job(
    test_api_key: str, orm_model_bundle: Bundle, orm_model_endpoint: Endpoint
) -> BatchJob:
    batch_job = BatchJob(
        batch_job_status="SUCCESS",
        created_by=test_api_key,
        owner=test_api_key,
        model_bundle_id=orm_model_bundle.id,
        model_endpoint_id=orm_model_endpoint.id,
    )
    batch_job.id = "test_batch_job_id_1"
    batch_job.created_at = datetime.datetime(2022, 1, 1)
    return batch_job


@pytest.fixture
def entity_batch_job_record(
    model_bundle_1: ModelBundle, entity_model_endpoint_record: ModelEndpointRecord
) -> BatchJobRecord:
    batch_job = BatchJobRecord(
        id="test_batch_job_id_1",
        created_at=datetime.datetime(2022, 1, 3),
        status="SUCCESS",
        created_by="test_user_id",
        owner="test_user_id",
        model_bundle=model_bundle_1,
        model_endpoint_id=entity_model_endpoint_record.id,
    )
    return batch_job


@pytest.fixture
def orm_docker_image_batch_job_bundle_1_v1(test_api_key: str) -> OrmDockerImageBatchJobBundle:
    batch_bundle = OrmDockerImageBatchJobBundle(
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
    batch_bundle.id = "test_docker_image_batch_job_bundle_id_11"
    batch_bundle.created_at = datetime.datetime(2022, 1, 1)
    return batch_bundle


@pytest.fixture
def orm_docker_image_batch_job_bundle_1_v2(test_api_key: str) -> OrmDockerImageBatchJobBundle:
    batch_bundle = OrmDockerImageBatchJobBundle(
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
    batch_bundle.id = "test_docker_image_batch_job_bundle_id_12"
    batch_bundle.created_at = datetime.datetime(2022, 1, 3)
    return batch_bundle


@pytest.fixture
def orm_docker_image_batch_job_bundle_2_v1(test_api_key: str) -> OrmDockerImageBatchJobBundle:
    batch_bundle = OrmDockerImageBatchJobBundle(
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
    batch_bundle.id = "test_docker_image_batch_job_bundle_id_21"
    batch_bundle.created_at = datetime.datetime(2022, 1, 2)
    return batch_bundle


@pytest.fixture
def dbsession() -> Callable[[], AsyncSession]:
    return lambda: AsyncSession()
