from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional

import pytest
from model_engine_server.domain.entities import (
    ModelEndpoint,
    ModelEndpointStatus,
    ModelEndpointType,
)
from model_engine_server.domain.gateways import DigestGateway
from model_engine_server.infra.gateways.resources.fake_queue_endpoint_resource_delegate import (
    FakeQueueEndpointResourceDelegate,
)
from model_engine_server.infra.services.endpoint_gc_service import (
    GC_EXEMPT_KEY,
    GC_FLAGGED_AT_KEY,
    GC_UNAVAILABLE_SINCE_KEY,
    EndpointGarbageCollectionService,
    EndpointGcConfig,
)

NOW = datetime(2026, 9, 21, 6, 0, tzinfo=timezone.utc)
DELETING = EndpointGcConfig(unavailable_days=30, grace_days=14, delete_cap=20, delete_enabled=True)


def _days_ago(days: int) -> str:
    return (NOW - timedelta(days=days)).isoformat()


class QueueWithActivity(FakeQueueEndpointResourceDelegate):
    def __init__(self, sent: Optional[int]):
        self.sent = sent

    async def messages_sent_since(self, endpoint_id: str, since: datetime) -> Optional[int]:
        return self.sent


class CapturingDigestGateway(DigestGateway):
    def __init__(self):
        self.digests: List[str] = []

    def send_digest(self, text: str) -> None:
        self.digests.append(text)


def _endpoint(
    base: ModelEndpoint,
    *,
    available: int,
    unavailable: int,
    metadata: Optional[Dict] = None,
    endpoint_type: ModelEndpointType = ModelEndpointType.STREAMING,
) -> ModelEndpoint:
    record = base.record.model_copy(
        update={"metadata": metadata or {}, "endpoint_type": endpoint_type}
    )
    deployment_state = base.infra_state.deployment_state.model_copy(
        update={"available_workers": available, "unavailable_workers": unavailable}
    )
    infra_state = base.infra_state.model_copy(update={"deployment_state": deployment_state})
    return ModelEndpoint(record=record, infra_state=infra_state)


def _build(
    fake_model_endpoint_record_repository,
    fake_resource_gateway,
    fake_model_endpoint_service,
    *endpoints: ModelEndpoint,
    queue_sent: Optional[int] = 0,
    config: EndpointGcConfig = DELETING,
    with_resources: bool = True,
):
    for endpoint in endpoints:
        fake_model_endpoint_record_repository.add_model_endpoint_record(endpoint.record)
        fake_model_endpoint_service.add_model_endpoint(endpoint)
        if with_resources:
            fake_resource_gateway.add_resource(endpoint.record.id, endpoint.infra_state)
    digest = CapturingDigestGateway()
    service = EndpointGarbageCollectionService(
        model_endpoint_record_repository=fake_model_endpoint_record_repository,
        resource_gateway=fake_resource_gateway,
        queue_delegate=QueueWithActivity(queue_sent),
        model_endpoint_service=fake_model_endpoint_service,
        digest_gateway=digest,
        config=config,
        now=lambda: NOW,
    )
    return service, digest


@pytest.mark.parametrize(
    "available,unavailable,metadata,bucket,expected_keys",
    [
        pytest.param(0, 1, {}, "observing", {GC_UNAVAILABLE_SINCE_KEY}, id="first-sighting"),
        pytest.param(
            0,
            2,
            {GC_UNAVAILABLE_SINCE_KEY: _days_ago(10)},
            "observing",
            {GC_UNAVAILABLE_SINCE_KEY},
            id="inside-window",
        ),
        pytest.param(
            0,
            1,
            {GC_UNAVAILABLE_SINCE_KEY: _days_ago(30)},
            "flagged_new",
            {GC_UNAVAILABLE_SINCE_KEY, GC_FLAGGED_AT_KEY},
            id="window-elapsed-flags",
        ),
        pytest.param(
            0,
            1,
            {GC_UNAVAILABLE_SINCE_KEY: _days_ago(40), GC_FLAGGED_AT_KEY: _days_ago(5)},
            "in_grace",
            {GC_UNAVAILABLE_SINCE_KEY, GC_FLAGGED_AT_KEY},
            id="inside-grace",
        ),
        pytest.param(
            1,
            0,
            {GC_UNAVAILABLE_SINCE_KEY: _days_ago(40), GC_FLAGGED_AT_KEY: _days_ago(5)},
            "cleared",
            set(),
            id="recovered-clears-state",
        ),
        pytest.param(
            0,
            0,
            {GC_UNAVAILABLE_SINCE_KEY: _days_ago(40)},
            "cleared",
            set(),
            id="scaled-to-zero-clears-state",
        ),
        pytest.param(
            0,
            1,
            {GC_UNAVAILABLE_SINCE_KEY: "not-a-timestamp"},
            "state_invalid",
            {GC_UNAVAILABLE_SINCE_KEY},
            id="unparseable-stamp-left-alone",
        ),
        pytest.param(1, 0, {}, None, set(), id="healthy-untouched"),
        pytest.param(0, 0, {}, None, set(), id="scaled-to-zero-untouched"),
        pytest.param(
            0,
            1,
            {GC_EXEMPT_KEY: True, GC_UNAVAILABLE_SINCE_KEY: _days_ago(90)},
            "exempt",
            {GC_EXEMPT_KEY, GC_UNAVAILABLE_SINCE_KEY},
            id="exempt-bool-untouched",
        ),
        pytest.param(
            0,
            1,
            {GC_EXEMPT_KEY: "true"},
            "exempt",
            {GC_EXEMPT_KEY},
            id="exempt-string-true",
        ),
        pytest.param(
            0,
            1,
            {GC_EXEMPT_KEY: "false"},
            "observing",
            {GC_EXEMPT_KEY, GC_UNAVAILABLE_SINCE_KEY},
            id="exempt-string-false-is-not-exempt",
        ),
    ],
)
@pytest.mark.asyncio
async def test_bookkeeping(
    fake_model_endpoint_record_repository,
    fake_resource_gateway,
    fake_model_endpoint_service,
    model_endpoint_1,
    available,
    unavailable,
    metadata,
    bucket,
    expected_keys,
):
    endpoint = _endpoint(
        model_endpoint_1, available=available, unavailable=unavailable, metadata=metadata
    )
    service, _ = _build(
        fake_model_endpoint_record_repository,
        fake_resource_gateway,
        fake_model_endpoint_service,
        endpoint,
    )
    report = await service.execute()

    stored = await fake_model_endpoint_record_repository.get_model_endpoint_record(
        endpoint.record.id
    )
    assert stored is not None
    assert set(stored.metadata.keys()) == expected_keys
    assert report.deleted == []
    if bucket is not None:
        assert [r.id for r in getattr(report, bucket)] == [endpoint.record.id]
    if bucket == "flagged_new":
        assert stored.metadata[GC_FLAGGED_AT_KEY] == NOW.isoformat()
    if bucket == "observing" and GC_UNAVAILABLE_SINCE_KEY not in metadata:
        assert stored.metadata[GC_UNAVAILABLE_SINCE_KEY] == NOW.isoformat()


@pytest.mark.parametrize(
    "delete_enabled,deleted_expected",
    [
        pytest.param(True, True, id="delete-enabled-deletes"),
        pytest.param(False, False, id="observe-only-defers"),
    ],
)
@pytest.mark.asyncio
async def test_delete_after_grace(
    fake_model_endpoint_record_repository,
    fake_resource_gateway,
    fake_model_endpoint_service,
    model_endpoint_1,
    delete_enabled,
    deleted_expected,
):
    endpoint = _endpoint(
        model_endpoint_1,
        available=0,
        unavailable=1,
        metadata={GC_UNAVAILABLE_SINCE_KEY: _days_ago(50), GC_FLAGGED_AT_KEY: _days_ago(14)},
    )
    service, digest = _build(
        fake_model_endpoint_record_repository,
        fake_resource_gateway,
        fake_model_endpoint_service,
        endpoint,
        config=EndpointGcConfig(
            unavailable_days=30, grace_days=14, delete_cap=20, delete_enabled=delete_enabled
        ),
    )
    report = await service.execute()

    assert ([r.id for r in report.deleted] == [endpoint.record.id]) is deleted_expected
    assert ([r.id for r in report.delete_deferred] == [endpoint.record.id]) is not deleted_expected
    assert (endpoint.record.id in fake_model_endpoint_service.db) is not deleted_expected
    assert len(digest.digests) == 1
    assert ("DELETE ENABLED" in digest.digests[0]) is deleted_expected


@pytest.mark.parametrize(
    "queue_sent,bucket,expected_keys",
    [
        pytest.param(0, "observing", {GC_UNAVAILABLE_SINCE_KEY}, id="idle-queue-keeps-clock"),
        pytest.param(3, "cleared", set(), id="active-queue-resets-clock"),
        pytest.param(
            None, "queue_unknown", {GC_UNAVAILABLE_SINCE_KEY}, id="unknown-queue-leaves-state"
        ),
    ],
)
@pytest.mark.asyncio
async def test_async_requires_idle_queue(
    fake_model_endpoint_record_repository,
    fake_resource_gateway,
    fake_model_endpoint_service,
    model_endpoint_1,
    queue_sent,
    bucket,
    expected_keys,
):
    endpoint = _endpoint(
        model_endpoint_1,
        available=0,
        unavailable=1,
        endpoint_type=ModelEndpointType.ASYNC,
        metadata={GC_UNAVAILABLE_SINCE_KEY: _days_ago(10)},
    )
    service, _ = _build(
        fake_model_endpoint_record_repository,
        fake_resource_gateway,
        fake_model_endpoint_service,
        endpoint,
        queue_sent=queue_sent,
    )
    report = await service.execute()

    stored = await fake_model_endpoint_record_repository.get_model_endpoint_record(
        endpoint.record.id
    )
    assert set(stored.metadata.keys()) == expected_keys
    assert [r.id for r in getattr(report, bucket)] == [endpoint.record.id]


@pytest.mark.asyncio
async def test_delete_cap_oldest_first(
    fake_model_endpoint_record_repository,
    fake_resource_gateway,
    fake_model_endpoint_service,
    model_endpoint_1,
    model_endpoint_2,
):
    older = _endpoint(
        model_endpoint_1,
        available=0,
        unavailable=1,
        metadata={GC_UNAVAILABLE_SINCE_KEY: _days_ago(60), GC_FLAGGED_AT_KEY: _days_ago(20)},
    )
    newer = _endpoint(
        model_endpoint_2,
        available=0,
        unavailable=1,
        metadata={GC_UNAVAILABLE_SINCE_KEY: _days_ago(50), GC_FLAGGED_AT_KEY: _days_ago(15)},
    )
    service, _ = _build(
        fake_model_endpoint_record_repository,
        fake_resource_gateway,
        fake_model_endpoint_service,
        newer,
        older,
        config=EndpointGcConfig(
            unavailable_days=30, grace_days=14, delete_cap=1, delete_enabled=True
        ),
    )

    report = await service.execute()

    assert [r.id for r in report.deleted] == [older.record.id]
    assert [r.id for r in report.delete_deferred] == [newer.record.id]


@pytest.mark.asyncio
async def test_no_deployment_clears_state_and_never_deletes(
    fake_model_endpoint_record_repository,
    fake_resource_gateway,
    fake_model_endpoint_service,
    model_endpoint_1,
):
    endpoint = _endpoint(
        model_endpoint_1,
        available=0,
        unavailable=1,
        metadata={GC_UNAVAILABLE_SINCE_KEY: _days_ago(60), GC_FLAGGED_AT_KEY: _days_ago(20)},
    )
    service, _ = _build(
        fake_model_endpoint_record_repository,
        fake_resource_gateway,
        fake_model_endpoint_service,
        endpoint,
        with_resources=False,
    )
    report = await service.execute()

    stored = await fake_model_endpoint_record_repository.get_model_endpoint_record(
        endpoint.record.id
    )
    assert stored.metadata == {}
    assert [r.id for r in report.no_deployment] == [endpoint.record.id]
    assert report.deleted == []


@pytest.mark.parametrize(
    "status", [ModelEndpointStatus.UPDATE_PENDING, ModelEndpointStatus.UPDATE_IN_PROGRESS]
)
@pytest.mark.asyncio
async def test_in_flight_update_is_never_deleted(
    fake_model_endpoint_record_repository,
    fake_resource_gateway,
    fake_model_endpoint_service,
    model_endpoint_1,
    status,
):
    endpoint = _endpoint(
        model_endpoint_1,
        available=0,
        unavailable=1,
        metadata={GC_UNAVAILABLE_SINCE_KEY: _days_ago(60), GC_FLAGGED_AT_KEY: _days_ago(20)},
    )
    endpoint.record.status = status
    service, _ = _build(
        fake_model_endpoint_record_repository,
        fake_resource_gateway,
        fake_model_endpoint_service,
        endpoint,
    )
    report = await service.execute()

    assert [r.id for r in report.in_flight] == [endpoint.record.id]
    assert report.deleted == [] and report.delete_deferred == []
    assert endpoint.record.id in fake_model_endpoint_service.db


@pytest.mark.asyncio
async def test_owner_update_during_grace_restarts_grace(
    fake_model_endpoint_record_repository,
    fake_resource_gateway,
    fake_model_endpoint_service,
    model_endpoint_1,
):
    endpoint = _endpoint(
        model_endpoint_1,
        available=0,
        unavailable=1,
        metadata={GC_UNAVAILABLE_SINCE_KEY: _days_ago(60), GC_FLAGGED_AT_KEY: _days_ago(20)},
    )
    endpoint.record.last_updated_at = NOW - timedelta(days=3)
    service, _ = _build(
        fake_model_endpoint_record_repository,
        fake_resource_gateway,
        fake_model_endpoint_service,
        endpoint,
    )
    report = await service.execute()

    stored = await fake_model_endpoint_record_repository.get_model_endpoint_record(
        endpoint.record.id
    )
    assert [r.id for r in report.flagged_new] == [endpoint.record.id]
    assert stored.metadata[GC_FLAGGED_AT_KEY] == NOW.isoformat()
    assert report.deleted == []


@pytest.mark.asyncio
async def test_metadata_write_keeps_concurrent_user_keys(
    fake_model_endpoint_record_repository,
    fake_resource_gateway,
    fake_model_endpoint_service,
    model_endpoint_1,
):
    endpoint = _endpoint(
        model_endpoint_1, available=0, unavailable=1, metadata={"_llm": {"model_name": "m"}}
    )
    service, _ = _build(
        fake_model_endpoint_record_repository,
        fake_resource_gateway,
        fake_model_endpoint_service,
        endpoint,
    )
    # A user updates metadata after GC has listed the records but before it writes.
    original_list = fake_model_endpoint_record_repository.list_model_endpoint_records

    async def list_then_mutate(**kwargs):
        records = await original_list(**kwargs)
        stored = fake_model_endpoint_record_repository.db[endpoint.record.id]
        stored.metadata = {**stored.metadata, "user_key": "added-mid-run"}
        return records

    fake_model_endpoint_record_repository.list_model_endpoint_records = list_then_mutate

    await service.execute()

    stored = await fake_model_endpoint_record_repository.get_model_endpoint_record(
        endpoint.record.id
    )
    assert stored.metadata["user_key"] == "added-mid-run"
    assert stored.metadata["_llm"] == {"model_name": "m"}
    assert GC_UNAVAILABLE_SINCE_KEY in stored.metadata


@pytest.mark.asyncio
async def test_locked_endpoint_skips_metadata_write(
    fake_model_endpoint_record_repository,
    fake_resource_gateway,
    fake_model_endpoint_service,
    model_endpoint_1,
):
    endpoint = _endpoint(model_endpoint_1, available=0, unavailable=1)
    service, _ = _build(
        fake_model_endpoint_record_repository,
        fake_resource_gateway,
        fake_model_endpoint_service,
        endpoint,
    )
    fake_model_endpoint_record_repository.force_lock_model_endpoint(endpoint.record)

    report = await service.execute()

    stored = await fake_model_endpoint_record_repository.get_model_endpoint_record(
        endpoint.record.id
    )
    assert stored.metadata == {}
    assert [r.id for r in report.write_skipped] == [endpoint.record.id]


@pytest.mark.asyncio
async def test_digest_lists_owner_fields_and_delete_date(
    fake_model_endpoint_record_repository,
    fake_resource_gateway,
    fake_model_endpoint_service,
    model_endpoint_1,
):
    endpoint = _endpoint(
        model_endpoint_1,
        available=0,
        unavailable=1,
        metadata={GC_UNAVAILABLE_SINCE_KEY: _days_ago(30)},
    )
    service, digest = _build(
        fake_model_endpoint_record_repository,
        fake_resource_gateway,
        fake_model_endpoint_service,
        endpoint,
    )
    await service.execute()

    text = digest.digests[0]
    assert f"created_by={endpoint.record.created_by}" in text
    assert f"owner={endpoint.record.owner}" in text
    assert f"delete_after={(NOW + timedelta(days=14)).strftime('%Y-%m-%d')}" in text
