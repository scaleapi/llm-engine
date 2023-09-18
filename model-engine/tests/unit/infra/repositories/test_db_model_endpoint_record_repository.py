import datetime
from typing import Any, Callable, Dict, List, Optional
from unittest.mock import AsyncMock, Mock

import pytest
from model_engine_server.common.dtos.model_endpoints import ModelEndpointOrderBy
from model_engine_server.domain.exceptions import ReadOnlyDatabaseException
from model_engine_server.db.models import Bundle, Endpoint
from model_engine_server.domain.entities import ModelEndpointRecord
from model_engine_server.infra.gateways import FakeMonitoringMetricsGateway
from model_engine_server.infra.repositories import db_model_endpoint_record_repository
from model_engine_server.infra.repositories.db_model_endpoint_record_repository import (
    DbModelEndpointRecordRepository,
    OrmModelEndpoint,
)
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession


@pytest.mark.asyncio
async def test_create_model_endpoint_record(
    orm_model_bundle: Bundle,
    orm_model_endpoint: Endpoint,
    dbsession: Callable[[], AsyncSession],
    fake_monitoring_metrics_gateway: FakeMonitoringMetricsGateway,
):
    def mock_model_endpoint_create(session: AsyncSession, endpoint: Endpoint) -> None:
        endpoint.id = "test_model_endpoint_id"
        endpoint.created_at = datetime.datetime(2022, 1, 3)
        endpoint.last_updated_at = datetime.datetime(2022, 1, 3)
        endpoint.current_bundle = orm_model_bundle

    def mock_model_endpoint_select_by_id(
        session: AsyncSession, endpoint_id: str
    ) -> Optional[Endpoint]:
        orm_model_endpoint.endpoint_id = endpoint_id
        orm_model_endpoint.created_at = datetime.datetime(2022, 1, 3)
        orm_model_endpoint.last_updated_at = datetime.datetime(2022, 1, 3)
        orm_model_endpoint.current_bundle = orm_model_bundle
        return orm_model_endpoint

    OrmModelEndpoint.create = AsyncMock(side_effect=mock_model_endpoint_create)
    OrmModelEndpoint.select_by_id = AsyncMock(side_effect=mock_model_endpoint_select_by_id)

    repo = DbModelEndpointRecordRepository(
        monitoring_metrics_gateway=fake_monitoring_metrics_gateway,
        session=dbsession,
        read_only=False,
    )
    model_endpoint = await repo.create_model_endpoint_record(
        name="test_model_endpoint_name",
        created_by="test_user_id",
        model_bundle_id="test_model_bundle_id",
        metadata={},
        endpoint_type="async",
        destination="test_destination",
        creation_task_id="test_creation_task_id",
        status="READY",
        owner="test_user_id",
    )

    assert model_endpoint


@pytest.mark.asyncio
async def test_create_model_endpoint_record_raises_if_read_only(
    orm_model_bundle: Bundle,
    dbsession: Callable[[], AsyncSession],
    fake_monitoring_metrics_gateway: FakeMonitoringMetricsGateway,
):
    def mock_model_endpoint_create(session: AsyncSession, endpoint: Endpoint) -> None:
        endpoint.id = "test_model_endpoint_id"
        endpoint.created_at = datetime.datetime(2022, 1, 3)
        endpoint.last_updated_at = datetime.datetime(2022, 1, 3)
        endpoint.current_bundle = orm_model_bundle

    OrmModelEndpoint.create = AsyncMock(side_effect=mock_model_endpoint_create)

    repo = DbModelEndpointRecordRepository(
        monitoring_metrics_gateway=fake_monitoring_metrics_gateway,
        session=dbsession,
        read_only=True,
    )
    with pytest.raises(ReadOnlyDatabaseException):
        await repo.create_model_endpoint_record(
            name="test_model_endpoint_name",
            created_by="test_user_id",
            model_bundle_id="test_model_bundle_id",
            metadata={},
            endpoint_type="async",
            destination="test_destination",
            creation_task_id="test_creation_task_id",
            status="READY",
            owner="test_user_id",
        )


@pytest.mark.asyncio
async def test_list_model_endpoint_records(
    dbsession: Callable[[], AsyncSession],
    orm_model_endpoint: Endpoint,
    orm_model_bundle: Bundle,
    entity_model_endpoint_record: ModelEndpointRecord,
    fake_monitoring_metrics_gateway: FakeMonitoringMetricsGateway,
):
    def mock_model_endpoint_select_all_by_filters(
        session: AsyncSession, filters: Any
    ) -> List[Endpoint]:
        orm_model_endpoint.created_at = datetime.datetime(2022, 1, 3)
        orm_model_endpoint.last_updated_at = datetime.datetime(2022, 1, 3)
        orm_model_endpoint.current_bundle = orm_model_bundle
        return [orm_model_endpoint]

    OrmModelEndpoint._select_all_by_filters = AsyncMock(
        side_effect=mock_model_endpoint_select_all_by_filters
    )

    repo = DbModelEndpointRecordRepository(
        monitoring_metrics_gateway=fake_monitoring_metrics_gateway,
        session=dbsession,
        read_only=False,
    )
    model_endpoints = await repo.list_model_endpoint_records(
        owner="test_user_id",
        name="test_model_endpoint_name",
        order_by=ModelEndpointOrderBy.NEWEST,
    )
    assert model_endpoints == [entity_model_endpoint_record]

    model_endpoints = await repo.list_model_endpoint_records(
        owner="test_user_id", name=None, order_by=ModelEndpointOrderBy.OLDEST
    )
    assert model_endpoints == [entity_model_endpoint_record]


@pytest.mark.asyncio
async def test_list_llm_model_endpoint_records(
    dbsession: Callable[[], AsyncSession],
    orm_model_endpoint: Endpoint,
    orm_model_bundle: Bundle,
    fake_monitoring_metrics_gateway: FakeMonitoringMetricsGateway,
):
    filter_content = "endpoint_metadata ? '_llm' AND hosted_model_inference.endpoints.name = :name_1 AND (hosted_model_inference.endpoints.owner = :owner_1 OR hosted_model_inference.endpoints.public_inference = true)"

    def mock_llm_model_endpoint_select_all_by_filters(
        session: AsyncSession, filters: Any
    ) -> List[Endpoint]:
        q = select(Endpoint)
        for f in filters:
            q = q.filter(f)
        assert filter_content in str(q)
        orm_model_endpoint.created_at = datetime.datetime(2022, 1, 3)
        orm_model_endpoint.last_updated_at = datetime.datetime(2022, 1, 3)
        orm_model_endpoint.current_bundle = orm_model_bundle
        return [orm_model_endpoint]

    OrmModelEndpoint._select_all_by_filters = AsyncMock(
        side_effect=mock_llm_model_endpoint_select_all_by_filters
    )

    repo = DbModelEndpointRecordRepository(
        monitoring_metrics_gateway=fake_monitoring_metrics_gateway,
        session=dbsession,
        read_only=False,
    )
    await repo.list_llm_model_endpoint_records(
        owner="test_user_id",
        name="test_model_endpoint_name",
        order_by=ModelEndpointOrderBy.NEWEST,
    )

    filter_content = "endpoint_metadata ? '_llm' AND (hosted_model_inference.endpoints.owner = :owner_1 OR hosted_model_inference.endpoints.public_inference = true)"
    await repo.list_llm_model_endpoint_records(
        owner="test_user_id",
        name=None,
        order_by=ModelEndpointOrderBy.NEWEST,
    )


@pytest.mark.asyncio
async def test_list_model_endpoint_records_team(
    dbsession: Callable[[], AsyncSession],
    orm_model_endpoint: Endpoint,
    orm_model_bundle: Bundle,
    entity_model_endpoint_record: ModelEndpointRecord,
    test_api_key_user_on_other_team: str,
    test_api_key_user_on_other_team_2: str,
    test_api_key_team: str,
    fake_monitoring_metrics_gateway: FakeMonitoringMetricsGateway,
):
    orm_model_endpoint.created_at = datetime.datetime(2022, 1, 3)
    orm_model_endpoint.last_updated_at = datetime.datetime(2022, 1, 3)
    orm_model_endpoint.current_bundle = orm_model_bundle
    orm_model_endpoint.created_by = test_api_key_user_on_other_team
    orm_model_endpoint.owner = test_api_key_team
    orm_model_bundle.created_by = test_api_key_user_on_other_team
    orm_model_bundle.owner = test_api_key_team
    entity_model_endpoint_record.created_by = test_api_key_user_on_other_team
    entity_model_endpoint_record.owner = test_api_key_team
    entity_model_endpoint_record.current_model_bundle.created_by = test_api_key_user_on_other_team
    entity_model_endpoint_record.current_model_bundle.owner = test_api_key_team

    def mock_model_endpoint_select_all_by_filters(
        session: AsyncSession, filters: Any
    ) -> List[Endpoint]:
        # we could check which filters are here but that seems brittle
        # correct filters are "owner = 'test_api_key_team'" and "name = 'test_model_endpoint_name'"
        return [orm_model_endpoint]

    OrmModelEndpoint._select_all_by_filters = AsyncMock(
        side_effect=mock_model_endpoint_select_all_by_filters
    )

    repo = DbModelEndpointRecordRepository(
        monitoring_metrics_gateway=fake_monitoring_metrics_gateway,
        session=dbsession,
        read_only=False,
    )
    model_endpoints = await repo.list_model_endpoint_records(
        owner=test_api_key_team,
        name="test_model_endpoint_name",
        order_by=ModelEndpointOrderBy.NEWEST,
    )
    assert model_endpoints == [entity_model_endpoint_record]

    model_endpoints = await repo.list_model_endpoint_records(
        owner=test_api_key_team, name=None, order_by=ModelEndpointOrderBy.OLDEST
    )
    assert model_endpoints == [entity_model_endpoint_record]


@pytest.mark.asyncio
async def test_get_model_endpoint_record_success(
    dbsession: Callable[[], AsyncSession],
    orm_model_endpoint: Endpoint,
    orm_model_bundle: Bundle,
    entity_model_endpoint_record: ModelEndpointRecord,
    fake_monitoring_metrics_gateway: FakeMonitoringMetricsGateway,
):
    def mock_model_endpoint_select_by_id(
        session: AsyncSession, endpoint_id: str
    ) -> Optional[Endpoint]:
        orm_model_endpoint.endpoint_id = endpoint_id
        orm_model_endpoint.created_at = datetime.datetime(2022, 1, 3)
        orm_model_endpoint.last_updated_at = datetime.datetime(2022, 1, 3)
        orm_model_endpoint.current_bundle = orm_model_bundle
        return orm_model_endpoint

    OrmModelEndpoint.select_by_id = AsyncMock(side_effect=mock_model_endpoint_select_by_id)

    repo = DbModelEndpointRecordRepository(
        monitoring_metrics_gateway=fake_monitoring_metrics_gateway,
        session=dbsession,
        read_only=False,
    )
    model_endpoint = await repo.get_model_endpoint_record(
        model_endpoint_id="test_model_endpoint_id"
    )

    assert model_endpoint == entity_model_endpoint_record


@pytest.mark.asyncio
async def test_get_model_endpoint_record_returns_none(
    dbsession: Callable[[], AsyncSession],
    fake_monitoring_metrics_gateway: FakeMonitoringMetricsGateway,
):
    OrmModelEndpoint.select_by_id = AsyncMock(return_value=None)

    repo = DbModelEndpointRecordRepository(
        monitoring_metrics_gateway=fake_monitoring_metrics_gateway,
        session=dbsession,
        read_only=False,
    )
    model_endpoint = await repo.get_model_endpoint_record(
        model_endpoint_id="test_model_endpoint_id_nonexistent"
    )

    assert model_endpoint is None


@pytest.mark.asyncio
async def test_delete_model_endpoint_record_success(
    dbsession: Callable[[], AsyncSession],
    orm_model_endpoint: Endpoint,
    orm_model_bundle: Bundle,
    entity_model_endpoint_record: ModelEndpointRecord,
    fake_monitoring_metrics_gateway: FakeMonitoringMetricsGateway,
):
    def mock_model_endpoint_select_by_id(
        session: AsyncSession, endpoint_id: str
    ) -> Optional[Endpoint]:
        orm_model_endpoint.endpoint_id = endpoint_id
        orm_model_endpoint.created_at = datetime.datetime(2022, 1, 3)
        orm_model_endpoint.last_updated_at = datetime.datetime(2022, 1, 3)
        orm_model_endpoint.current_bundle = orm_model_bundle
        return orm_model_endpoint

    OrmModelEndpoint.select_by_id = AsyncMock(side_effect=mock_model_endpoint_select_by_id)
    OrmModelEndpoint.delete = AsyncMock()

    repo = DbModelEndpointRecordRepository(
        monitoring_metrics_gateway=fake_monitoring_metrics_gateway,
        session=dbsession,
        read_only=False,
    )
    deleted = await repo.delete_model_endpoint_record(model_endpoint_id="test_model_endpoint_id")

    assert deleted
    OrmModelEndpoint.delete.assert_called()


@pytest.mark.asyncio
async def test_delete_model_endpoint_record_raises_if_read_only(
    dbsession: Callable[[], AsyncSession],
    orm_model_endpoint: Endpoint,
    orm_model_bundle: Bundle,
    entity_model_endpoint_record: ModelEndpointRecord,
    fake_monitoring_metrics_gateway: FakeMonitoringMetricsGateway,
):
    def mock_model_endpoint_select_by_id(
        session: AsyncSession, endpoint_id: str
    ) -> Optional[Endpoint]:
        orm_model_endpoint.endpoint_id = endpoint_id
        orm_model_endpoint.created_at = datetime.datetime(2022, 1, 3)
        orm_model_endpoint.last_updated_at = datetime.datetime(2022, 1, 3)
        orm_model_endpoint.current_bundle = orm_model_bundle
        return orm_model_endpoint

    OrmModelEndpoint.select_by_id = AsyncMock(side_effect=mock_model_endpoint_select_by_id)
    OrmModelEndpoint.delete = AsyncMock()

    repo = DbModelEndpointRecordRepository(
        monitoring_metrics_gateway=fake_monitoring_metrics_gateway,
        session=dbsession,
        read_only=True,
    )
    with pytest.raises(ReadOnlyDatabaseException):
        await repo.delete_model_endpoint_record(model_endpoint_id="test_model_endpoint_id")


@pytest.mark.asyncio
async def test_delete_model_endpoint_record_returns_false(
    dbsession: Callable[[], AsyncSession],
    fake_monitoring_metrics_gateway: FakeMonitoringMetricsGateway,
):
    OrmModelEndpoint.select_by_id = AsyncMock(return_value=None)
    OrmModelEndpoint.delete = AsyncMock()

    repo = DbModelEndpointRecordRepository(
        monitoring_metrics_gateway=fake_monitoring_metrics_gateway,
        session=dbsession,
        read_only=False,
    )
    deleted = await repo.delete_model_endpoint_record(model_endpoint_id="test_model_endpoint_id")

    assert not deleted
    OrmModelEndpoint.delete.assert_not_called()


@pytest.mark.asyncio
async def test_update_model_endpoint_record_raises_if_read_only(
    dbsession: Callable[[], AsyncSession],
    orm_model_endpoint: Endpoint,
    orm_model_bundle: Bundle,
    entity_model_endpoint_record: ModelEndpointRecord,
    fake_monitoring_metrics_gateway: FakeMonitoringMetricsGateway,
):
    def mock_model_endpoint_select_by_id(
        session: AsyncSession, endpoint_id: str
    ) -> Optional[Endpoint]:
        orm_model_endpoint.endpoint_id = endpoint_id
        orm_model_endpoint.created_at = datetime.datetime(2022, 1, 3)
        orm_model_endpoint.last_updated_at = datetime.datetime(2022, 1, 3)
        orm_model_endpoint.current_bundle = orm_model_bundle
        return orm_model_endpoint

    def mock_model_endpoint_update_by_name_owner(
        session: AsyncSession, name: str, owner: str, kwargs: Dict[str, Any]
    ) -> None:
        orm_model_endpoint.name = name
        orm_model_endpoint.owner = owner
        for key, value in kwargs.items():
            orm_model_endpoint.__setattr__(key, value)

    OrmModelEndpoint.select_by_id = AsyncMock(side_effect=mock_model_endpoint_select_by_id)
    OrmModelEndpoint.update_by_name_owner = AsyncMock(
        side_effect=mock_model_endpoint_update_by_name_owner
    )

    update_kwargs = dict(
        metadata={"test_key": "test_value"},
        creation_task_id="test_update_creation_task_id",
        status="UPDATE_IN_PROGRESS",
    )
    repo = DbModelEndpointRecordRepository(
        monitoring_metrics_gateway=fake_monitoring_metrics_gateway,
        session=dbsession,
        read_only=True,
    )
    with pytest.raises(ReadOnlyDatabaseException):
        await repo.update_model_endpoint_record(
            model_endpoint_id="test_model_endpoint_id",
            model_bundle_id="test_model_bundle_id",
            **update_kwargs,  # type: ignore
        )


@pytest.mark.asyncio
async def test_update_model_endpoint_record_success(
    dbsession: Callable[[], AsyncSession],
    orm_model_endpoint: Endpoint,
    orm_model_bundle: Bundle,
    entity_model_endpoint_record: ModelEndpointRecord,
    fake_monitoring_metrics_gateway: FakeMonitoringMetricsGateway,
):
    def mock_model_endpoint_select_by_id(
        session: AsyncSession, endpoint_id: str
    ) -> Optional[Endpoint]:
        orm_model_endpoint.endpoint_id = endpoint_id
        orm_model_endpoint.created_at = datetime.datetime(2022, 1, 3)
        orm_model_endpoint.last_updated_at = datetime.datetime(2022, 1, 3)
        orm_model_endpoint.current_bundle = orm_model_bundle
        return orm_model_endpoint

    def mock_model_endpoint_update_by_name_owner(
        session: AsyncSession, name: str, owner: str, kwargs: Dict[str, Any]
    ) -> None:
        orm_model_endpoint.name = name
        orm_model_endpoint.owner = owner
        for key, value in kwargs.items():
            orm_model_endpoint.__setattr__(key, value)

    OrmModelEndpoint.select_by_id = AsyncMock(side_effect=mock_model_endpoint_select_by_id)
    OrmModelEndpoint.update_by_name_owner = AsyncMock(
        side_effect=mock_model_endpoint_update_by_name_owner
    )

    update_kwargs = dict(
        metadata={"test_key": "test_value"},
        creation_task_id="test_update_creation_task_id",
        status="UPDATE_IN_PROGRESS",
        public_inference=True,
    )
    repo = DbModelEndpointRecordRepository(
        monitoring_metrics_gateway=fake_monitoring_metrics_gateway,
        session=dbsession,
        read_only=False,
    )
    model_endpoint = await repo.update_model_endpoint_record(
        model_endpoint_id="test_model_endpoint_id",
        model_bundle_id="test_model_bundle_id",
        **update_kwargs,  # type: ignore
    )

    for key, value in update_kwargs.items():
        assert model_endpoint.__getattribute__(key) == value


@pytest.mark.asyncio
async def test_update_model_endpoint_record_returns_none(
    dbsession: Callable[[], AsyncSession],
    fake_monitoring_metrics_gateway: FakeMonitoringMetricsGateway,
):
    OrmModelEndpoint.select_by_id = AsyncMock(return_value=None)

    repo = DbModelEndpointRecordRepository(
        monitoring_metrics_gateway=fake_monitoring_metrics_gateway,
        session=dbsession,
        read_only=False,
    )
    model_endpoint = await repo.update_model_endpoint_record(
        model_endpoint_id="test_model_endpoint_id"
    )

    assert model_endpoint is None


@pytest.mark.asyncio
async def test_db_lock_context(
    dbsession: Callable[[], AsyncSession],
    fake_monitoring_metrics_gateway: FakeMonitoringMetricsGateway,
):
    mock_lock_context = AsyncMock()
    mock_lock_context.__aenter__ = AsyncMock(return_value=mock_lock_context)
    mock_lock_context.lock_acquired = Mock(return_value=True)
    db_model_endpoint_record_repository.AdvisoryLockContextManager = Mock(
        return_value=mock_lock_context
    )
    mock_session = AsyncMock()
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_sessionmaker = Mock(return_value=mock_session)
    repo = DbModelEndpointRecordRepository(
        monitoring_metrics_gateway=fake_monitoring_metrics_gateway,
        session=dbsession,
        read_only=False,
    )

    # Before entering, the lock should not have been acquired.
    assert not repo.DbLockContext(lock_id=0, session=mock_sessionmaker).lock_acquired()

    async with repo.DbLockContext(lock_id=0, session=mock_sessionmaker) as lock:
        assert lock.lock_acquired()
    mock_session.__aenter__.assert_called_once()
    mock_session.__aexit__.assert_called_once()
    mock_lock_context.__aenter__.assert_called_once()
    mock_lock_context.__aexit__.assert_called_once()


def test_get_lock_context(
    dbsession: Callable[[], AsyncSession],
    entity_model_endpoint_record: ModelEndpointRecord,
    fake_monitoring_metrics_gateway: FakeMonitoringMetricsGateway,
):
    repo = DbModelEndpointRecordRepository(
        monitoring_metrics_gateway=fake_monitoring_metrics_gateway,
        session=dbsession,
        read_only=False,
    )
    lock_context = repo.get_lock_context(entity_model_endpoint_record)
    assert isinstance(lock_context, DbModelEndpointRecordRepository.LockContext)


def test_get_lock_context_raises_if_read_only(
    dbsession: Callable[[], AsyncSession],
    entity_model_endpoint_record: ModelEndpointRecord,
    fake_monitoring_metrics_gateway: FakeMonitoringMetricsGateway,
):
    repo = DbModelEndpointRecordRepository(
        monitoring_metrics_gateway=fake_monitoring_metrics_gateway,
        session=dbsession,
        read_only=True,
    )
    with pytest.raises(ReadOnlyDatabaseException):
        repo.get_lock_context(entity_model_endpoint_record)
