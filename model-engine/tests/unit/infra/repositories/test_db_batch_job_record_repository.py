import datetime
from typing import Any, Callable, Dict, List, Optional
from unittest.mock import AsyncMock

import pytest
from model_engine_server.core.domain_exceptions import ReadOnlyDatabaseException
from model_engine_server.db.models import BatchJob, Bundle
from model_engine_server.domain.entities import BatchJobRecord
from model_engine_server.infra.repositories.db_batch_job_record_repository import (
    DbBatchJobRecordRepository,
    OrmBatchJob,
)
from sqlalchemy.ext.asyncio import AsyncSession


@pytest.mark.asyncio
async def test_create_batch_job_record(
    orm_model_bundle: Bundle,
    orm_batch_job: BatchJob,
    dbsession: Callable[[], AsyncSession],
):
    def mock_batch_job_create(session: AsyncSession, batch_job: BatchJob) -> None:
        batch_job.id = "test_batch_job_id"
        batch_job.created_at = datetime.datetime(2022, 1, 3)
        batch_job.model_bundle = orm_model_bundle

    def mock_batch_job_select_by_id(session: AsyncSession, batch_job_id: str) -> Optional[BatchJob]:
        orm_batch_job.batch_job_id = batch_job_id
        orm_batch_job.created_at = datetime.datetime(2022, 1, 3)
        orm_batch_job.model_bundle = orm_model_bundle
        return orm_batch_job

    OrmBatchJob.create = AsyncMock(side_effect=mock_batch_job_create)
    OrmBatchJob.select_by_id = AsyncMock(side_effect=mock_batch_job_select_by_id)

    repo = DbBatchJobRecordRepository(session=dbsession, read_only=False)
    batch_job = await repo.create_batch_job_record(
        status="SUCCESS",
        created_by="test_user_id",
        owner="test_user_id",
        model_bundle_id="test_model_bundle_id",
    )

    assert batch_job


@pytest.mark.asyncio
async def test_create_batch_job_record_raises_if_read_only(
    orm_model_bundle: Bundle, dbsession: Callable[[], AsyncSession]
):
    def mock_batch_job_create(session: AsyncSession, batch_job: BatchJob) -> None:
        batch_job.id = "test_batch_job_id"
        batch_job.created_at = datetime.datetime(2022, 1, 3)
        batch_job.model_bundle = orm_model_bundle

    OrmBatchJob.create = AsyncMock(side_effect=mock_batch_job_create)

    repo = DbBatchJobRecordRepository(session=dbsession, read_only=True)
    with pytest.raises(ReadOnlyDatabaseException):
        await repo.create_batch_job_record(
            status="SUCCESS",
            created_by="test_user_id",
            owner="test_user_id",
            model_bundle_id="test_model_bundle_id",
        )


@pytest.mark.asyncio
async def test_list_batch_job_records(
    dbsession: Callable[[], AsyncSession],
    orm_batch_job: BatchJob,
    orm_model_bundle: Bundle,
    entity_batch_job_record: BatchJobRecord,
):
    def mock_batch_job_select_all_by_owner(
        session: AsyncSession, owner: Optional[str]
    ) -> List[BatchJob]:
        orm_batch_job.created_at = datetime.datetime(2022, 1, 3)
        orm_batch_job.model_bundle = orm_model_bundle
        return [orm_batch_job]

    OrmBatchJob.select_all_by_owner = AsyncMock(side_effect=mock_batch_job_select_all_by_owner)

    repo = DbBatchJobRecordRepository(session=dbsession, read_only=True)
    batch_jobs = await repo.list_batch_job_records(owner="test_user_id")
    assert batch_jobs == [entity_batch_job_record]


@pytest.mark.asyncio
async def test_get_batch_job_record_success(
    dbsession: Callable[[], AsyncSession],
    orm_batch_job: BatchJob,
    orm_model_bundle: Bundle,
    entity_batch_job_record: BatchJobRecord,
):
    def mock_batch_job_select_by_id(session: AsyncSession, batch_job_id: str) -> Optional[BatchJob]:
        orm_batch_job.batch_job_id = batch_job_id
        orm_batch_job.created_at = datetime.datetime(2022, 1, 3)
        orm_batch_job.model_bundle = orm_model_bundle
        return orm_batch_job

    OrmBatchJob.select_by_id = AsyncMock(side_effect=mock_batch_job_select_by_id)

    repo = DbBatchJobRecordRepository(session=dbsession, read_only=True)
    batch_job = await repo.get_batch_job_record(batch_job_id="test_batch_job_id")

    assert batch_job == entity_batch_job_record


@pytest.mark.asyncio
async def test_get_batch_job_record_returns_none(dbsession: Callable[[], AsyncSession]):
    OrmBatchJob.select_by_id = AsyncMock(return_value=None)

    repo = DbBatchJobRecordRepository(session=dbsession, read_only=True)
    batch_job = await repo.get_batch_job_record(batch_job_id="test_batch_job_id")

    assert batch_job is None


@pytest.mark.asyncio
async def test_update_batch_job_record_raises_if_read_only(
    dbsession: Callable[[], AsyncSession],
    orm_batch_job: BatchJob,
    orm_model_bundle: Bundle,
):
    def mock_batch_job_select_by_id(session: AsyncSession, batch_job_id: str) -> Optional[BatchJob]:
        orm_batch_job.batch_job_id = batch_job_id
        orm_batch_job.created_at = datetime.datetime(2022, 1, 3)
        orm_batch_job.model_bundle = orm_model_bundle
        return orm_batch_job

    def mock_batch_job_update_by_id(
        session: AsyncSession, batch_job_id: str, kwargs: Dict[str, Any]
    ) -> None:
        orm_batch_job.id = batch_job_id
        for key, value in kwargs.items():
            orm_batch_job.__setattr__(key, value)

    OrmBatchJob.select_by_id = AsyncMock(side_effect=mock_batch_job_select_by_id)
    OrmBatchJob.update_by_id = AsyncMock(side_effect=mock_batch_job_update_by_id)

    update_kwargs = dict(
        model_endpoint_id="test_update_model_endpoint_id",
        task_ids_location="test_update_task_ids_location",
        result_location="test_update_result_location",
    )
    repo = DbBatchJobRecordRepository(session=dbsession, read_only=True)
    with pytest.raises(ReadOnlyDatabaseException):
        await repo.update_batch_job_record(
            batch_job_id="test_batch_job_id",
            **update_kwargs,  # type: ignore
        )


@pytest.mark.asyncio
async def test_update_batch_job_record_success(
    dbsession: Callable[[], AsyncSession],
    orm_batch_job: BatchJob,
    orm_model_bundle: Bundle,
):
    def mock_batch_job_select_by_id(session: AsyncSession, batch_job_id: str) -> Optional[BatchJob]:
        orm_batch_job.batch_job_id = batch_job_id
        orm_batch_job.created_at = datetime.datetime(2022, 1, 3)
        orm_batch_job.model_bundle = orm_model_bundle
        return orm_batch_job

    def mock_batch_job_update_by_id(
        session: AsyncSession, batch_job_id: str, kwargs: Dict[str, Any]
    ) -> None:
        orm_batch_job.id = batch_job_id
        for key, value in kwargs.items():
            orm_batch_job.__setattr__(key, value)

    OrmBatchJob.select_by_id = AsyncMock(side_effect=mock_batch_job_select_by_id)
    OrmBatchJob.update_by_id = AsyncMock(side_effect=mock_batch_job_update_by_id)

    update_kwargs = dict(
        model_endpoint_id="test_update_model_endpoint_id",
        task_ids_location="test_update_task_ids_location",
        result_location="test_update_result_location",
    )

    repo = DbBatchJobRecordRepository(session=dbsession, read_only=False)
    batch_job = await repo.update_batch_job_record(
        batch_job_id="test_batch_job_id",
        **update_kwargs,  # type: ignore
    )

    for key, value in update_kwargs.items():
        assert batch_job.__getattribute__(key) == value


@pytest.mark.asyncio
async def test_update_batch_job_record_returns_none(dbsession: Callable[[], AsyncSession]):
    OrmBatchJob.select_by_id = AsyncMock(return_value=None)

    repo = DbBatchJobRecordRepository(session=dbsession, read_only=False)
    batch_job = await repo.update_batch_job_record(batch_job_id="test_batch_job_id")

    assert batch_job is None


@pytest.mark.asyncio
async def test_unset_model_endpoint_id_batch_job_record_raises_if_read_only(
    dbsession: Callable[[], AsyncSession],
    orm_batch_job: BatchJob,
    orm_model_bundle: Bundle,
):
    def mock_batch_job_select_by_id(session: AsyncSession, batch_job_id: str) -> Optional[BatchJob]:
        orm_batch_job.batch_job_id = batch_job_id
        orm_batch_job.created_at = datetime.datetime(2022, 1, 3)
        orm_batch_job.model_bundle = orm_model_bundle
        return orm_batch_job

    def mock_batch_job_update_by_id(
        session: AsyncSession, batch_job_id: str, kwargs: Dict[str, Any]
    ) -> None:
        orm_batch_job.id = batch_job_id
        for key, value in kwargs.items():
            orm_batch_job.__setattr__(key, value)

    OrmBatchJob.select_by_id = AsyncMock(side_effect=mock_batch_job_select_by_id)
    OrmBatchJob.update_by_id = AsyncMock(side_effect=mock_batch_job_update_by_id)

    repo = DbBatchJobRecordRepository(session=dbsession, read_only=True)
    with pytest.raises(ReadOnlyDatabaseException):
        await repo.unset_model_endpoint_id(batch_job_id="test_batch_job_id")


@pytest.mark.asyncio
async def test_unset_model_endpoint_id_batch_job_record_success(
    dbsession: Callable[[], AsyncSession],
    orm_batch_job: BatchJob,
    orm_model_bundle: Bundle,
):
    def mock_batch_job_select_by_id(session: AsyncSession, batch_job_id: str) -> Optional[BatchJob]:
        orm_batch_job.batch_job_id = batch_job_id
        orm_batch_job.created_at = datetime.datetime(2022, 1, 3)
        orm_batch_job.model_bundle = orm_model_bundle
        return orm_batch_job

    def mock_batch_job_update_by_id(
        session: AsyncSession, batch_job_id: str, kwargs: Dict[str, Any]
    ) -> None:
        orm_batch_job.id = batch_job_id
        for key, value in kwargs.items():
            orm_batch_job.__setattr__(key, value)

    OrmBatchJob.select_by_id = AsyncMock(side_effect=mock_batch_job_select_by_id)
    OrmBatchJob.update_by_id = AsyncMock(side_effect=mock_batch_job_update_by_id)

    repo = DbBatchJobRecordRepository(session=dbsession, read_only=False)
    batch_job = await repo.unset_model_endpoint_id(batch_job_id="test_batch_job_id")

    assert batch_job is not None
    assert batch_job.model_endpoint_id is None


@pytest.mark.asyncio
async def test_unset_model_endpoint_id_batch_job_record_returns_none(
    dbsession: Callable[[], AsyncSession]
):
    OrmBatchJob.select_by_id = AsyncMock(return_value=None)

    repo = DbBatchJobRecordRepository(session=dbsession, read_only=False)
    batch_job = await repo.unset_model_endpoint_id(batch_job_id="test_batch_job_id")

    assert batch_job is None
