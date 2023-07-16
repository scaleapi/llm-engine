import datetime
from typing import Callable
from unittest.mock import AsyncMock

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from llm_engine_server.common.dtos.model_bundles import ModelBundleOrderBy
from llm_engine_server.core.domain_exceptions import ReadOnlyDatabaseException
from llm_engine_server.db.models import DockerImageBatchJobBundle as OrmDockerImageBatchJobBundle
from llm_engine_server.domain.entities import GpuType
from llm_engine_server.domain.entities.docker_image_batch_job_bundle_entity import (
    DockerImageBatchJobBundle,
)
from llm_engine_server.domain.exceptions import CorruptRecordInfraStateException
from llm_engine_server.infra.repositories import DbDockerImageBatchJobBundleRepository
from llm_engine_server.infra.repositories.db_docker_image_batch_job_bundle_repository import (
    translate_docker_image_batch_job_bundle_orm_to_entity,
)


@pytest.mark.asyncio
async def test_create_docker_image_batch_job(
    dbsession: Callable[[], AsyncSession],
    orm_docker_image_batch_job_bundle_1_v1: OrmDockerImageBatchJobBundle,
):
    def mock_batch_bundle_create(session: AsyncSession, batch_bundle: OrmDockerImageBatchJobBundle):
        batch_bundle.id = "batbun_id"
        batch_bundle.created_at = datetime.datetime(2022, 1, 1)

    def mock_batch_bundle_select_by_id(session: AsyncSession, batch_bundle_id: str):
        orm_docker_image_batch_job_bundle_1_v1.id = batch_bundle_id
        return orm_docker_image_batch_job_bundle_1_v1

    OrmDockerImageBatchJobBundle.create = AsyncMock(side_effect=mock_batch_bundle_create)
    OrmDockerImageBatchJobBundle.select_by_id = AsyncMock(
        side_effect=mock_batch_bundle_select_by_id
    )

    repo = DbDockerImageBatchJobBundleRepository(session=dbsession, read_only=False)
    batch_bundle = await repo.create_docker_image_batch_job_bundle(
        name="test_docker_image_batch_job_bundle_1",
        created_by="test_user_id",
        owner="test_user_id",
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
    assert batch_bundle


@pytest.mark.asyncio
async def test_create_docker_image_batch_job_raises_if_read_only(
    dbsession: Callable[[], AsyncSession],
    orm_docker_image_batch_job_bundle_1_v1: OrmDockerImageBatchJobBundle,
):
    def mock_batch_bundle_create(session: AsyncSession, batch_bundle: OrmDockerImageBatchJobBundle):
        batch_bundle.id = "batbun_id"
        batch_bundle.created_at = datetime.datetime(2022, 1, 1)

    def mock_batch_bundle_select_by_id(session: AsyncSession, batch_bundle_id: str):
        orm_docker_image_batch_job_bundle_1_v1.id = batch_bundle_id
        return orm_docker_image_batch_job_bundle_1_v1

    OrmDockerImageBatchJobBundle.create = AsyncMock(side_effect=mock_batch_bundle_create)
    OrmDockerImageBatchJobBundle.select_by_id = AsyncMock(
        side_effect=mock_batch_bundle_select_by_id
    )

    repo = DbDockerImageBatchJobBundleRepository(session=dbsession, read_only=True)
    with pytest.raises(ReadOnlyDatabaseException):
        await repo.create_docker_image_batch_job_bundle(
            name="test_docker_image_batch_job_bundle_1",
            created_by="test_user_id",
            owner="test_user_id",
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


@pytest.mark.asyncio
async def test_list_docker_image_batch_job_bundles(
    dbsession: Callable[[], AsyncSession],
    orm_docker_image_batch_job_bundle_1_v1: OrmDockerImageBatchJobBundle,
    orm_docker_image_batch_job_bundle_1_v2: OrmDockerImageBatchJobBundle,
    orm_docker_image_batch_job_bundle_2_v1: OrmDockerImageBatchJobBundle,
    docker_image_batch_job_bundle_1_v1: DockerImageBatchJobBundle,
    docker_image_batch_job_bundle_1_v2: DockerImageBatchJobBundle,
    docker_image_batch_job_bundle_2_v1: DockerImageBatchJobBundle,
    test_api_key: str,
    test_api_key_team: str,
):

    orm_docker_image_batch_job_bundle_1_v2.created_by = test_api_key_team
    orm_docker_image_batch_job_bundle_1_v2.owner = test_api_key_team
    docker_image_batch_job_bundle_1_v2.created_by = test_api_key_team
    docker_image_batch_job_bundle_1_v2.owner = test_api_key_team

    b1_name = docker_image_batch_job_bundle_1_v1.name
    b2_name = docker_image_batch_job_bundle_2_v1.name

    def mock_batch_bundle_select_all_by_name_owner(session: AsyncSession, owner: str, name: str):
        if (owner, name) == (test_api_key, b1_name):
            return [orm_docker_image_batch_job_bundle_1_v1]
        elif (owner, name) == (test_api_key, b2_name):
            return [orm_docker_image_batch_job_bundle_2_v1]
        elif (owner, name) == (test_api_key_team, b1_name):
            return [orm_docker_image_batch_job_bundle_1_v2]
        else:
            return []

    def mock_batch_bundle_select_all_by_owner(session: AsyncSession, owner: str):
        if owner == test_api_key:
            return [orm_docker_image_batch_job_bundle_1_v1, orm_docker_image_batch_job_bundle_2_v1]
        elif owner == test_api_key_team:
            return [orm_docker_image_batch_job_bundle_1_v2]
        else:
            return []

    OrmDockerImageBatchJobBundle.select_all_by_name_owner = AsyncMock(
        side_effect=mock_batch_bundle_select_all_by_name_owner
    )
    OrmDockerImageBatchJobBundle.select_all_by_owner = AsyncMock(
        side_effect=mock_batch_bundle_select_all_by_owner
    )

    repo = DbDockerImageBatchJobBundleRepository(session=dbsession, read_only=True)
    batch_bundles = await repo.list_docker_image_batch_job_bundles(
        owner=test_api_key, name=None, order_by=ModelBundleOrderBy.OLDEST
    )
    assert len(batch_bundles) == 2
    assert batch_bundles[0].dict() == docker_image_batch_job_bundle_1_v1.dict()
    assert batch_bundles[1].dict() == docker_image_batch_job_bundle_2_v1.dict()

    batch_bundles_2 = await repo.list_docker_image_batch_job_bundles(
        owner=test_api_key_team, name=None, order_by=ModelBundleOrderBy.NEWEST
    )
    assert len(batch_bundles_2) == 1
    assert batch_bundles_2[0].dict() == docker_image_batch_job_bundle_1_v2.dict()

    batch_bundles_3 = await repo.list_docker_image_batch_job_bundles(
        owner=test_api_key, name=b1_name, order_by=None
    )
    assert len(batch_bundles_3) == 1
    assert batch_bundles_3[0].dict() == docker_image_batch_job_bundle_1_v1.dict()


@pytest.mark.asyncio
async def test_get_docker_image_batch_job_bundle_success(
    dbsession: Callable[[], AsyncSession],
    orm_docker_image_batch_job_bundle_1_v1: OrmDockerImageBatchJobBundle,
    docker_image_batch_job_bundle_1_v1: DockerImageBatchJobBundle,
):
    def mock_batch_bundle_select_by_id(session: AsyncSession, batch_bundle_id: str):
        orm_docker_image_batch_job_bundle_1_v1.id = batch_bundle_id
        return orm_docker_image_batch_job_bundle_1_v1

    OrmDockerImageBatchJobBundle.select_by_id = AsyncMock(
        side_effect=mock_batch_bundle_select_by_id
    )

    repo = DbDockerImageBatchJobBundleRepository(session=dbsession, read_only=True)
    batch_bundle = await repo.get_docker_image_batch_job_bundle(
        docker_image_batch_job_bundle_id=orm_docker_image_batch_job_bundle_1_v1.id
    )
    assert batch_bundle.dict() == docker_image_batch_job_bundle_1_v1.dict()


@pytest.mark.asyncio
async def test_get_docker_image_batch_job_bundle_returns_none(
    dbsession: Callable[[], AsyncSession],
):
    def mock_batch_bundle_select_by_id(session: AsyncSession, batch_bundle_id: str):
        return None

    OrmDockerImageBatchJobBundle.select_by_id = AsyncMock(
        side_effect=mock_batch_bundle_select_by_id
    )

    repo = DbDockerImageBatchJobBundleRepository(session=dbsession, read_only=True)
    batch_bundle = await repo.get_docker_image_batch_job_bundle(
        docker_image_batch_job_bundle_id="some_id"
    )
    assert batch_bundle is None


@pytest.mark.asyncio
async def test_get_latest_docker_image_batch_job_bundle(
    dbsession: Callable[[], AsyncSession],
    orm_docker_image_batch_job_bundle_1_v1: OrmDockerImageBatchJobBundle,
    orm_docker_image_batch_job_bundle_1_v2: OrmDockerImageBatchJobBundle,
    orm_docker_image_batch_job_bundle_2_v1: OrmDockerImageBatchJobBundle,
    docker_image_batch_job_bundle_1_v1: DockerImageBatchJobBundle,
    docker_image_batch_job_bundle_1_v2: DockerImageBatchJobBundle,
    docker_image_batch_job_bundle_2_v1: DockerImageBatchJobBundle,
    test_api_key: str,
    test_api_key_team: str,
):
    orm_docker_image_batch_job_bundle_1_v2.created_by = test_api_key_team
    orm_docker_image_batch_job_bundle_1_v2.owner = test_api_key_team
    docker_image_batch_job_bundle_1_v2.created_by = test_api_key_team
    docker_image_batch_job_bundle_1_v2.owner = test_api_key_team

    b1_name = docker_image_batch_job_bundle_1_v1.name
    b2_name = docker_image_batch_job_bundle_2_v1.name

    def mock_batch_bundle_select_by_name_owner(session: AsyncSession, owner: str, name: str):
        if (owner, name) == (test_api_key, b1_name):
            return orm_docker_image_batch_job_bundle_1_v1
        elif (owner, name) == (test_api_key, b2_name):
            return orm_docker_image_batch_job_bundle_2_v1
        elif (owner, name) == (test_api_key_team, b1_name):
            return orm_docker_image_batch_job_bundle_1_v2
        else:
            return None

    OrmDockerImageBatchJobBundle.select_latest_by_name_owner = AsyncMock(
        side_effect=mock_batch_bundle_select_by_name_owner
    )

    repo = DbDockerImageBatchJobBundleRepository(session=dbsession, read_only=True)

    batch_bundle_1 = await repo.get_latest_docker_image_batch_job_bundle(test_api_key, b1_name)
    assert batch_bundle_1 == docker_image_batch_job_bundle_1_v1

    batch_bundle_2 = await repo.get_latest_docker_image_batch_job_bundle(test_api_key, b2_name)
    assert batch_bundle_2 == docker_image_batch_job_bundle_2_v1

    batch_bundle_3 = await repo.get_latest_docker_image_batch_job_bundle(test_api_key_team, b2_name)
    assert batch_bundle_3 is None


def test_translate_orm_to_entity():
    # sanity test to make sure that the automatic parse_obj() conversion works
    orm = OrmDockerImageBatchJobBundle(
        name="name",
        created_by="a",
        owner="a",
        image_repository="b",
        image_tag="c",
        command=["asdf", "asdf"],
        env={},
        mount_location=None,
        cpus=None,
        memory=None,
        storage="1Gi",
        gpus="1",
        gpu_type="nvidia-tesla-t4",
    )
    orm.id = "id"
    orm.created_at = datetime.datetime(2020, 1, 1)
    expected = DockerImageBatchJobBundle(
        id="id",
        created_at=datetime.datetime(2020, 1, 1),
        name="name",
        created_by="a",
        owner="a",
        image_repository="b",
        image_tag="c",
        command=["asdf", "asdf"],
        env=dict(),
        mount_location=None,
        cpus=None,
        memory=None,
        storage="1Gi",
        gpus=1,
        gpu_type=GpuType.NVIDIA_TESLA_T4,
    )
    actual = translate_docker_image_batch_job_bundle_orm_to_entity(orm)
    assert expected == actual

    orm.gpus = None
    expected.gpus = None
    actual = translate_docker_image_batch_job_bundle_orm_to_entity(orm)
    assert expected == actual

    orm.gpu_type = None
    expected.gpu_type = None
    actual = translate_docker_image_batch_job_bundle_orm_to_entity(orm)
    assert expected == actual


def test_translate_bad_orm_raises():
    orm = OrmDockerImageBatchJobBundle(
        name="name",
        created_by="a",
        owner="a",
        image_repository="b",
        image_tag="c",
        command=["asdf", "asdf"],
        env={},
        mount_location=None,
        cpus=None,
        memory=None,
        storage="1Gi",
        gpus="1",
        gpu_type="nvidia-hopper-h9001",
    )
    orm.id = "id"
    orm.created_at = datetime.datetime(2020, 1, 1)
    with pytest.raises(CorruptRecordInfraStateException):
        translate_docker_image_batch_job_bundle_orm_to_entity(orm)
