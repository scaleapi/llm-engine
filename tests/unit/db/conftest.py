import datetime
import os
from typing import List

import psycopg2
import pytest
import pytest_asyncio
import testing.postgresql
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.ext.asyncio import create_async_engine

from spellbook_serve.db.base import Session, SessionAsync
from spellbook_serve.db.local_setup import init_database, init_database_and_engine
from spellbook_serve.db.models import (
    BatchJob,
    Bundle,
    DockerImageBatchJobBundle,
    Endpoint,
    Model,
    ModelArtifact,
    ModelVersion,
)


def init_testing_postgresql(postgresql: testing.postgresql.Postgresql) -> None:
    """Initializes local postgresql server."""
    conn = psycopg2.connect(**postgresql.dsn())
    init_database(postgresql.url(), conn)  # type: ignore


@pytest.fixture(scope="session")
def engine() -> Engine:
    if os.getenv("ML_INFRA_DATABASE_URL"):
        url = os.getenv("ML_INFRA_DATABASE_URL")
        db_engine = init_database_and_engine(url)
        yield db_engine
    else:
        Postgresql = testing.postgresql.PostgresqlFactory(
            cache_initialized_db=True,
            on_initialized=init_testing_postgresql,
        )
        postgresql = Postgresql().__enter__()
        yield create_engine(postgresql.url(), echo=False, future=True)


@pytest.fixture(scope="function")
def dbsession(engine: Engine) -> Session:
    """Returns a sqlalchemy session, and after the test tears down everything properly."""
    connection = engine.connect()
    transaction = connection.begin()
    session = Session(bind=connection)

    yield session

    session.close()
    transaction.rollback()
    connection.close()


@pytest_asyncio.fixture(scope="function")
async def dbsession_async(engine: Engine) -> SessionAsync:
    """Returns a sqlalchemy session, and after the test tears down everything properly."""
    url = str(engine.url).replace("postgresql://", "postgresql+asyncpg://")
    engine = create_async_engine(url)
    async with engine.connect() as connection:
        async with connection.begin() as transaction:
            session = SessionAsync(bind=connection)
            yield session
            await session.close()
            await transaction.rollback()
        await connection.close()


@pytest_asyncio.fixture(scope="function")
async def bundles(dbsession_async: SessionAsync) -> List[Bundle]:
    bundle1 = Bundle(
        name="test_bundle_1",
        created_by="test_user_1",
        model_artifact_ids=None,
        schema_location=None,
        owner="test_user_1",
        flavor="cloudpickle_artifact",
        # Artifact fields
        artifact_requirements=["test_requirement_1"],
        artifact_location="test_location_1",
        artifact_app_config=None,
        artifact_framework_type="pytorch",
        artifact_pytorch_image_tag="test_tag_1",
        # Cloudpickle artifact fields
        cloudpickle_artifact_load_predict_fn="test_load_predict_fn",
        cloudpickle_artifact_load_model_fn="test_load_model_fn",
        # Legacy fields
        location="test_location_1",
        version="v0",
        registered_model_name="registered_model_name_1",
        bundle_metadata=None,
        env_params=None,
        packaging_type=None,
        app_config=None,
    )
    bundle2 = Bundle(
        name="test_bundle_2",
        created_by="test_user_1",
        model_artifact_ids=None,
        schema_location=None,
        owner="test_user_1",
        flavor="zip_artifact",
        # Artifact fields
        artifact_requirements=["test_requirement_1"],
        artifact_location="test_location_2",
        artifact_app_config={"test_key": "test_value"},
        artifact_framework_type="custom_base_image",
        artifact_image_repository="test_repo_1",
        artifact_image_tag="test_tag_1",
        # Zip artifact fields
        zip_artifact_load_predict_fn_module_path="test_path_1",
        zip_artifact_load_model_fn_module_path="test_path_2",
        # Legacy fields
        location="test_location_1",
        version="v0",
        registered_model_name="registered_model_name_1",
        bundle_metadata=None,
        env_params=None,
        packaging_type=None,
        app_config=None,
    )
    bundle3 = Bundle(
        name="test_bundle_3",
        created_by="test_user_2",
        model_artifact_ids=None,
        schema_location=None,
        owner="test_user_1",
        flavor="runnable_image",
        # Runnable Image fields
        runnable_image_repository="test_repository_1",
        runnable_image_tag="test_tag_1",
        runnable_image_command=["test_command_1"],
        runnable_image_predict_route="/test_predict_route",
        runnable_image_healthcheck_route="/test_healthcheck_route",
        runnable_image_env={"test_key": "test_value"},
        runnable_image_protocol="http",
        runnable_image_readiness_initial_delay_seconds=300,
        # Legacy fields
        location="test_location_1",
        version="v0",
        registered_model_name="registered_model_name_1",
        bundle_metadata=None,
        env_params=None,
        packaging_type=None,
        app_config=None,
    )
    bundle4 = Bundle(
        name="test_bundle_4",
        created_by="test_user_2",
        model_artifact_ids=None,
        schema_location=None,
        owner="test_user_1",
        flavor="triton_enhanced_runnable_image",
        # Runnable Image fields
        runnable_image_repository="test_repository_1",
        runnable_image_tag="test_tag_1",
        runnable_image_command=["test_command_1"],
        runnable_image_predict_route="/test_predict_route",
        runnable_image_healthcheck_route="/test_healthcheck_route",
        runnable_image_env={"test_key": "test_value"},
        runnable_image_protocol="http",
        runnable_image_readiness_initial_delay_seconds=300,
        # Triton enhanced runnable image fields
        triton_enhanced_runnable_image_model_repository="test_model_repository_1",
        triton_enhanced_runnable_image_model_replicas={"test_model_1": "test_val"},
        triton_enhanced_runnable_image_num_cpu=3.5,
        triton_enhanced_runnable_image_commit_tag="test_commit_tag_1",
        triton_enhanced_runnable_image_storage="test_storage_1",
        triton_enhanced_runnable_image_readiness_initial_delay_seconds=350,
        # Legacy fields
        location="test_location_1",
        version="v0",
        registered_model_name="registered_model_name_1",
        bundle_metadata=None,
        env_params=None,
        packaging_type=None,
        app_config=None,
    )
    bundle5 = Bundle(
        name="test_bundle_5",
        created_by="test_user_2",
        model_artifact_ids=None,
        schema_location=None,
        owner="test_user_1",
        flavor="streaming_enhanced_runnable_image",
        # Runnable Image fields
        runnable_image_repository="test_repository_1",
        runnable_image_tag="test_tag_1",
        runnable_image_command=["test_command_1"],
        runnable_image_predict_route="/test_predict_route",
        runnable_image_healthcheck_route="/test_healthcheck_route",
        runnable_image_env={"test_key": "test_value"},
        runnable_image_protocol="http",
        runnable_image_readiness_initial_delay_seconds=300,
        # Streaming Enhanced Runnable Image fields
        streaming_enhanced_runnable_image_streaming_command=["test_streaming_command_1"],
        streaming_enhanced_runnable_image_streaming_predict_route="/test_streaming_predict_route",
        # Legacy fields
        location="test_location_1",
        version="v0",
        registered_model_name="registered_model_name_1",
        bundle_metadata=None,
        env_params=None,
        packaging_type=None,
        app_config=None,
    )
    bundles = [bundle1, bundle2, bundle3, bundle4, bundle5]
    for bundle in bundles:
        await Bundle.create(dbsession_async, bundle)
    return bundles


@pytest_asyncio.fixture(scope="function")
async def endpoints(dbsession_async: SessionAsync, bundles: List[Bundle]) -> List[Endpoint]:
    endpoint1 = Endpoint(
        name="test_endpoint_1",
        created_by="test_user_1",
        current_bundle_id=bundles[0].id,
        endpoint_metadata=None,
        creation_task_id="test_creation_task_id_1",
        endpoint_type="async",
        destination="test_destination_1",
        endpoint_status="READY",
        owner="test_user_1",
    )
    endpoint2 = Endpoint(
        name="test_endpoint_2",
        created_by="test_user_1",
        current_bundle_id=bundles[0].id,
        endpoint_metadata=None,
        creation_task_id="test_creation_task_id_1",
        endpoint_type="async",
        destination="test_destination_1",
        endpoint_status="READY",
        owner="test_user_1",
    )
    endpoint3 = Endpoint(
        name="test_endpoint_3",
        created_by="test_user_1",
        current_bundle_id=bundles[1].id,
        endpoint_metadata=None,
        creation_task_id="test_creation_task_id_1",
        endpoint_type="async",
        destination="test_destination_1",
        endpoint_status="READY",
        owner="test_user_1",
    )
    endpoints = [endpoint1, endpoint2, endpoint3]
    for endpoint in endpoints:
        await Endpoint.create(dbsession_async, endpoint)
    return endpoints


@pytest_asyncio.fixture(scope="function")
async def batch_jobs(
    dbsession_async: SessionAsync, bundles: List[Bundle], endpoints: List[Endpoint]
) -> List[BatchJob]:
    batch_job1 = BatchJob(
        batch_job_status="READY",
        created_by="test_user_1",
        owner="test_user_1",
        model_bundle_id=bundles[0].id,
        model_endpoint_id=endpoints[0].id,
        task_ids_location=None,
    )
    batch_job2 = BatchJob(
        batch_job_status="READY",
        created_by="test_user_1",
        owner="test_user_1",
        model_bundle_id=bundles[0].id,
        model_endpoint_id=endpoints[0].id,
        task_ids_location=None,
    )
    batch_job3 = BatchJob(
        batch_job_status="READY",
        created_by="test_user_2",
        owner="test_user_2",
        model_bundle_id=bundles[1].id,
        model_endpoint_id=endpoints[2].id,
        task_ids_location=None,
    )
    jobs = [batch_job1, batch_job2, batch_job3]
    for batch_job in jobs:
        await BatchJob.create(dbsession_async, batch_job)
    return jobs


@pytest_asyncio.fixture(scope="function")
async def docker_image_batch_job_bundles(
    dbsession_async: SessionAsync,
) -> List[DockerImageBatchJobBundle]:
    batch_bundle_1 = DockerImageBatchJobBundle(
        name="test_docker_image_batch_job_bundle_1",
        created_by="test_user_1",
        owner="test_user_1",
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
    )
    batch_bundle_2 = DockerImageBatchJobBundle(
        name="test_docker_image_batch_job_bundle_1",
        created_by="test_user_1",
        owner="test_user_1",
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
    )
    batch_bundle_3 = DockerImageBatchJobBundle(
        name="test_docker_image_batch_job_bundle_2",
        created_by="test_user_2",
        owner="test_user_2",
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
    )
    batch_bundle_1.created_at = datetime.datetime(2022, 1, 1)
    batch_bundle_2.created_at = datetime.datetime(2022, 1, 3)
    batch_bundle_3.created_at = datetime.datetime(2022, 1, 2)
    batch_bundles = [batch_bundle_1, batch_bundle_2, batch_bundle_3]
    for batch_bundle in batch_bundles:
        await DockerImageBatchJobBundle.create(dbsession_async, batch_bundle)
    return batch_bundles


@pytest.fixture(scope="function")
def models(dbsession: Session) -> List[Model]:
    model1 = Model(
        name="test_model_1",
        description="test_description_1",
        task_types=["test_task_type_1", "test_task_type_2"],
        created_by="test_user_id_1",
        owner="test_user_id_1",
    )
    model2 = Model(
        name="test_model_2",
        description="test_description_2",
        task_types=["test_task_type_1", "test_task_type_3"],
        created_by="test_user_id_1",
        owner="test_user_id_1",
    )
    model3 = Model(
        name="test_model_1",
        description="test_description_1",
        task_types=["test_task_type_2", "test_task_type_3"],
        created_by="test_user_id_2",
        owner="test_user_id_2",
    )
    models = [model1, model2, model3]
    for model in models:
        Model.create(dbsession, model)
    return models


@pytest_asyncio.fixture(scope="function")
async def model_versions(
    dbsession: Session, models: List[Model], bundles: List[Bundle]
) -> List[ModelVersion]:
    model_version1 = ModelVersion(
        model_id=models[0].id,
        version_number=0,
        tags=["test_tag_1", "test_tag_2"],
        metadata={"key1": "value1"},
        created_by="test_user_id_1",
    )
    model_version2 = ModelVersion(
        model_id=models[0].id,
        version_number=1,
        spellbook_serve_model_bundle_id=bundles[0].id,
        tags=["test_tag_1", "test_tag_3"],
        metadata={"key1": "value2"},
        created_by="test_user_id_1",
    )
    model_version3 = ModelVersion(
        model_id=models[2].id,
        version_number=0,
        spellbook_serve_model_bundle_id=bundles[1].id,
        nucleus_model_id="test_nucleus_model_id_1",
        tags=["test_tag_1", "test_tag_2"],
        metadata={"key2": "value3"},
        created_by="test_user_id_1",
    )
    model_versions = [model_version1, model_version2, model_version3]
    for model_version in model_versions:
        ModelVersion.create(dbsession, model_version)
    return model_versions


@pytest.fixture(scope="function")
def model_artifacts(dbsession: Session) -> List[ModelArtifact]:
    model_artifact1 = ModelArtifact(
        name="test_model_artifact_1",
        description="test_description_1",
        is_public=True,
        created_by="test_user_id_1",
        owner="test_user_id_1",
        input_schema={"test_schema_key": "test_schema_value"},
        output_schema={"test_schema_key": "test_schema_value"},
        config={"test_config_key": "test_config_value"},
        location="test_location",
        format="pytorch",
        format_metadata={"test_format_key": "test_format_value"},
        source="huggingface",
        source_metadata={"test_source_key": "test_source_value"},
    )
    model_artifact2 = ModelArtifact(
        name="test_model_artifact_2",
        description="test_description_2",
        is_public=False,
        created_by="test_user_id_1",
        owner="test_user_id_1",
        input_schema={"test_schema_key": "test_schema_value"},
        output_schema={"test_schema_key": "test_schema_value"},
        config={"test_config_key": "test_config_value"},
        location="test_location",
        format="pytorch",
        format_metadata={"test_format_key": "test_format_value"},
        source="huggingface",
        source_metadata={"test_source_key": "test_source_value"},
    )
    model_artifact3 = ModelArtifact(
        name="test_model_artifact_3",
        description="test_description_3",
        is_public=True,
        created_by="test_user_id_2",
        owner="test_user_id_2",
        input_schema={"test_schema_key": "test_schema_value"},
        output_schema={"test_schema_key": "test_schema_value"},
        config={"test_config_key": "test_config_value"},
        location="test_location",
        format="tensorflow",
        format_metadata={"test_format_key": "test_format_value"},
        source="mlflow",
        source_metadata={"test_source_key": "test_source_value"},
    )
    model_artifacts = [model_artifact1, model_artifact2, model_artifact3]
    for model_artifact in model_artifacts:
        ModelArtifact.create(dbsession, model_artifact)
    return model_artifacts
