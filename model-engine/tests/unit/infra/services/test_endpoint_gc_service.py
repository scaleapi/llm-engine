from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Set

import pytest
from model_engine_server.domain.entities import (
    ModelEndpoint,
    ModelEndpointStatus,
    ModelEndpointType,
)
from model_engine_server.domain.gateways import DigestGateway, EndpointTrafficGateway, TrafficKey
from model_engine_server.infra.gateways.resources.fake_queue_endpoint_resource_delegate import (
    FakeQueueEndpointResourceDelegate,
)
from model_engine_server.infra.services.endpoint_gc_service import (
    BROKEN,
    DELETE,
    GC_EXEMPT_KEY,
    GC_LAST_TRAFFIC_AT_KEY,
    GC_OBSERVED_AT_KEY,
    GC_SCALE_TO_ZERO_REQUESTED_AT_KEY,
    GC_SCALE_TO_ZERO_TASK_ID_KEY,
    GC_SEEN_TASK_ID_KEY,
    GC_TOUCHED_AT_KEY,
    GC_UNAVAILABLE_SINCE_KEY,
    IDLE,
    SCALE_TO_ZERO,
    EndpointGarbageCollectionService,
    EndpointGcConfig,
)

NOW = datetime(2026, 9, 23, 6, 0, tzinfo=timezone.utc)
ACTING = EndpointGcConfig(actions_enabled=True)


def _days_ago(days: float) -> str:
    return (NOW - timedelta(days=days)).isoformat()


class FakeQueue(FakeQueueEndpointResourceDelegate):
    def __init__(
        self, sent: Optional[int] = 0, depth: int = 0, in_flight: int = 0, delayed: int = 0
    ):
        self.sent = sent
        self.depth, self.in_flight, self.delayed = depth, in_flight, delayed

    async def get_queue_attributes(self, endpoint_id: str) -> Dict:
        return {
            "Attributes": {
                "ApproximateNumberOfMessages": str(self.depth),
                "ApproximateNumberOfMessagesNotVisible": str(self.in_flight),
                "ApproximateNumberOfMessagesDelayed": str(self.delayed),
            }
        }

    async def messages_sent_since(self, endpoint_id: str, since: datetime) -> Optional[int]:
        return self.sent


class FakeTraffic(EndpointTrafficGateway):
    key = TrafficKey.ENDPOINT_NAME

    def __init__(
        self, active: Optional[Set[str]] = None, history: Optional[Dict[str, datetime]] = None
    ):
        self.active, self.history = active, history

    async def active_keys(self, since: datetime) -> Optional[Set[str]]:
        return None if self.active is None else set(self.active)

    async def last_active_at(self, since: datetime) -> Optional[Dict[str, datetime]]:
        return self.history


class CapturingDigestGateway(DigestGateway):
    def __init__(self):
        self.digests: List[str] = []

    def send_digest(self, text: str) -> bool:
        self.digests.append(text)
        return True


def _endpoint(
    base: ModelEndpoint,
    *,
    available: int,
    unavailable: int,
    metadata: Optional[Dict] = None,
    endpoint_type: ModelEndpointType = ModelEndpointType.STREAMING,
    status: ModelEndpointStatus = ModelEndpointStatus.READY,
    last_updated_at: Optional[datetime] = None,
    min_workers: Optional[int] = None,
) -> ModelEndpoint:
    record = base.record.model_copy(
        update={
            "metadata": metadata or {},
            "endpoint_type": endpoint_type,
            "status": status,
            "last_updated_at": last_updated_at,
        }
    )
    if min_workers is None:
        # GC-parked scenarios carry a request stamp and min_workers 0; otherwise the owner's value.
        min_workers = 0 if (metadata or {}).get(GC_SCALE_TO_ZERO_REQUESTED_AT_KEY) else 1
    deployment_state = base.infra_state.deployment_state.model_copy(
        update={
            "available_workers": available,
            "unavailable_workers": unavailable,
            "min_workers": min_workers,
        }
    )
    infra_state = base.infra_state.model_copy(update={"deployment_state": deployment_state})
    return ModelEndpoint(record=record, infra_state=infra_state)


class Harness:
    def __init__(self, repo, resource_gateway, endpoint_service):
        self.repo, self.resources, self.service = repo, resource_gateway, endpoint_service
        self.digest = CapturingDigestGateway()

    def add(self, endpoint: ModelEndpoint, with_resources: bool = True) -> ModelEndpoint:
        self.repo.add_model_endpoint_record(endpoint.record)
        self.service.add_model_endpoint(endpoint)
        if with_resources:
            self.resources.add_resource(endpoint.record.id, endpoint.infra_state)
        return endpoint

    async def run(
        self,
        *,
        traffic_names: Optional[Set[str]] = frozenset(),
        history: Optional[Dict[str, datetime]] = None,
        queue_sent: Optional[int] = 0,
        queue_depth: int = 0,
        queue_in_flight: int = 0,
        queue_delayed: int = 0,
        config: EndpointGcConfig = ACTING,
    ):
        gc = EndpointGarbageCollectionService(
            model_endpoint_record_repository=self.repo,
            resource_gateway=self.resources,
            queue_delegate=FakeQueue(queue_sent, queue_depth, queue_in_flight, queue_delayed),
            traffic_gateways=[
                FakeTraffic(None if traffic_names is None else set(traffic_names), history)
            ],
            model_endpoint_service=self.service,
            digest_gateway=self.digest,
            config=config,
            now=lambda: NOW,
        )
        return await gc.execute()

    async def stored(self, endpoint: ModelEndpoint) -> Dict:
        record = await self.repo.get_model_endpoint_record(endpoint.record.id)
        return dict(record.metadata or {})


@pytest.fixture
def harness(
    fake_model_endpoint_record_repository, fake_resource_gateway, fake_model_endpoint_service
):
    return Harness(
        fake_model_endpoint_record_repository, fake_resource_gateway, fake_model_endpoint_service
    )


BOOKKEEPING = {GC_TOUCHED_AT_KEY, GC_SEEN_TASK_ID_KEY, GC_OBSERVED_AT_KEY}


def _keys(metadata: Dict) -> Set[str]:
    return {k for k in metadata if k not in BOOKKEEPING}


# ---- broken clock ------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "metadata,expect_bucket,expect_kind,expect_keys",
    [
        pytest.param({}, "tracking", None, {GC_UNAVAILABLE_SINCE_KEY}, id="first-sighting"),
        pytest.param(
            {GC_UNAVAILABLE_SINCE_KEY: _days_ago(16)},
            "upcoming14",
            SCALE_TO_ZERO,
            {GC_UNAVAILABLE_SINCE_KEY},
            id="14-day-notice",
        ),
        pytest.param(
            {GC_UNAVAILABLE_SINCE_KEY: _days_ago(29)},
            "upcoming1",
            SCALE_TO_ZERO,
            {GC_UNAVAILABLE_SINCE_KEY},
            id="1-day-notice",
        ),
        pytest.param(
            {GC_UNAVAILABLE_SINCE_KEY: _days_ago(30)},
            "scaled_to_zero",
            SCALE_TO_ZERO,
            {
                GC_UNAVAILABLE_SINCE_KEY,
                GC_SCALE_TO_ZERO_REQUESTED_AT_KEY,
                GC_SCALE_TO_ZERO_TASK_ID_KEY,
            },
            id="day-30-scales-to-zero",
        ),
    ],
)
@pytest.mark.asyncio
async def test_broken_clock(
    harness, model_endpoint_1, metadata, expect_bucket, expect_kind, expect_keys
):
    endpoint = harness.add(
        _endpoint(model_endpoint_1, available=0, unavailable=1, metadata=metadata)
    )
    report = await harness.run()

    assert _keys(await harness.stored(endpoint)) == expect_keys
    if expect_bucket.startswith("upcoming"):
        actions = report.upcoming[int(expect_bucket[len("upcoming") :])]
    else:
        actions = getattr(report, expect_bucket)
    ids = [a.record.id if hasattr(a, "record") else a.id for a in actions]
    assert ids == [endpoint.record.id]
    if expect_kind:
        assert actions[0].kind == expect_kind and actions[0].reason == BROKEN
    assert report.deleted == []


@pytest.mark.asyncio
async def test_broken_parked_by_gc_is_deleted_at_90(harness, model_endpoint_1):
    endpoint = harness.add(
        _endpoint(
            model_endpoint_1,
            available=0,
            unavailable=0,
            metadata={
                GC_UNAVAILABLE_SINCE_KEY: _days_ago(90),
                GC_SCALE_TO_ZERO_REQUESTED_AT_KEY: _days_ago(60),
                GC_SCALE_TO_ZERO_TASK_ID_KEY: "gc-task",
                GC_SEEN_TASK_ID_KEY: "gc-task",
                GC_TOUCHED_AT_KEY: _days_ago(60),
            },
            last_updated_at=NOW - timedelta(days=60, hours=-1),
        )
    )
    endpoint.record.creation_task_id = "gc-task"
    report = await harness.run()

    assert [a.record.id for a in report.deleted] == [endpoint.record.id]
    assert report.deleted[0].kind == DELETE and report.deleted[0].reason == BROKEN
    assert endpoint.record.id not in harness.service.db


@pytest.mark.asyncio
async def test_broken_update_failed_still_counts_down(harness, model_endpoint_1):
    # Scale-to-zero went through the builder and failed; the Deployment is still up and dead.
    endpoint = harness.add(
        _endpoint(
            model_endpoint_1,
            available=0,
            unavailable=1,
            min_workers=1,
            status=ModelEndpointStatus.UPDATE_FAILED,
            metadata={
                GC_UNAVAILABLE_SINCE_KEY: _days_ago(90),
                GC_SCALE_TO_ZERO_REQUESTED_AT_KEY: _days_ago(60),
                GC_SCALE_TO_ZERO_TASK_ID_KEY: "gc-task",
                GC_SEEN_TASK_ID_KEY: "gc-task",
                GC_TOUCHED_AT_KEY: _days_ago(60),
            },
            last_updated_at=NOW - timedelta(days=59, hours=23),
        )
    )
    endpoint.record.creation_task_id = "gc-task"
    report = await harness.run()

    assert [a.record.id for a in report.deleted] == [endpoint.record.id]


@pytest.mark.parametrize(
    "queue_sent,expected",
    [
        pytest.param(0, "tracking", id="silent-queue-is-broken"),
        pytest.param(5, "recovered", id="active-queue-clears-broken-clock"),
        pytest.param(None, None, id="unknown-queue-freezes"),
    ],
)
@pytest.mark.asyncio
async def test_broken_async_needs_silent_queue(harness, model_endpoint_1, queue_sent, expected):
    endpoint = harness.add(
        _endpoint(
            model_endpoint_1,
            available=0,
            unavailable=1,
            endpoint_type=ModelEndpointType.ASYNC,
            metadata={GC_UNAVAILABLE_SINCE_KEY: _days_ago(10)},
        )
    )
    report = await harness.run(queue_sent=queue_sent)

    stored = await harness.stored(endpoint)
    if expected == "tracking":
        assert [r.id for r in report.tracking] == [endpoint.record.id]
        assert GC_UNAVAILABLE_SINCE_KEY in stored
    elif expected == "recovered":
        assert [r.id for r in report.recovered] == [endpoint.record.id]
        assert GC_UNAVAILABLE_SINCE_KEY not in stored
        assert stored[GC_LAST_TRAFFIC_AT_KEY] == NOW.isoformat()
    else:
        assert report.sources_unknown == []
        assert [r.id for r in report.queue_unknown] == [endpoint.record.id]
        assert stored[GC_UNAVAILABLE_SINCE_KEY] == _days_ago(10)
        assert report.tracking == [] and report.deferred == []


@pytest.mark.asyncio
async def test_sync_attempts_do_not_revive_a_broken_endpoint(harness, model_endpoint_1):
    endpoint = harness.add(
        _endpoint(
            model_endpoint_1,
            available=0,
            unavailable=1,
            metadata={GC_UNAVAILABLE_SINCE_KEY: _days_ago(30)},
        )
    )
    report = await harness.run(traffic_names={endpoint.record.name})

    assert [a.record.id for a in report.scaled_to_zero] == [endpoint.record.id]


# ---- idle clock --------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "metadata,traffic,history_days,expect,expect_last_traffic",
    [
        pytest.param({}, set(), None, "start-now", NOW, id="first-sight-no-history"),
        pytest.param(
            {}, set(), 120, "upcoming14", NOW - timedelta(days=76), id="history-backfill-floored"
        ),
        pytest.param({}, set(), 40, "start-now", NOW - timedelta(days=40), id="history-backfills"),
        pytest.param(
            {GC_LAST_TRAFFIC_AT_KEY: _days_ago(50), GC_OBSERVED_AT_KEY: _days_ago(1)},
            {"hit"},
            None,
            "refreshed",
            NOW,
            id="traffic-refreshes-stamp",
        ),
        pytest.param(
            {GC_LAST_TRAFFIC_AT_KEY: _days_ago(83), GC_OBSERVED_AT_KEY: _days_ago(1)},
            set(),
            None,
            "upcoming7",
            NOW - timedelta(days=83),
            id="7-day-notice",
        ),
        pytest.param(
            {GC_LAST_TRAFFIC_AT_KEY: _days_ago(90), GC_OBSERVED_AT_KEY: _days_ago(1)},
            set(),
            None,
            "scaled",
            NOW - timedelta(days=90),
            id="day-90-scales-to-zero",
        ),
        pytest.param(
            {GC_LAST_TRAFFIC_AT_KEY: _days_ago(90)},
            set(),
            None,
            "start-lookback",
            NOW - timedelta(hours=36),
            id="clock-without-observation-boundary-restarts",
        ),
    ],
)
@pytest.mark.asyncio
async def test_idle_clock(
    harness, model_endpoint_1, metadata, traffic, history_days, expect, expect_last_traffic
):
    endpoint = harness.add(
        _endpoint(model_endpoint_1, available=1, unavailable=0, metadata=metadata)
    )
    names = {endpoint.record.name} if traffic else set()
    history = {endpoint.record.name: NOW - timedelta(days=history_days)} if history_days else None
    report = await harness.run(traffic_names=names, history=history)

    stored = await harness.stored(endpoint)
    assert stored[GC_LAST_TRAFFIC_AT_KEY] == expect_last_traffic.isoformat()
    if expect == "scaled":
        assert [a.record.id for a in report.scaled_to_zero] == [endpoint.record.id]
        assert report.scaled_to_zero[0].reason == IDLE
        assert GC_SCALE_TO_ZERO_REQUESTED_AT_KEY in stored
    elif expect.startswith("upcoming"):
        assert [a.record.id for a in report.upcoming[int(expect[len("upcoming") :])]] == [
            endpoint.record.id
        ]
    else:
        assert report.scaled_to_zero == [] and report.deferred == []


@pytest.mark.asyncio
async def test_idle_parked_by_gc_deleted_at_180(harness, model_endpoint_1):
    endpoint = harness.add(
        _endpoint(
            model_endpoint_1,
            available=0,
            unavailable=0,
            metadata={
                GC_LAST_TRAFFIC_AT_KEY: _days_ago(180),
                GC_OBSERVED_AT_KEY: _days_ago(1),
                GC_SCALE_TO_ZERO_REQUESTED_AT_KEY: _days_ago(90),
                GC_SCALE_TO_ZERO_TASK_ID_KEY: "gc-task",
                GC_SEEN_TASK_ID_KEY: "gc-task",
                GC_TOUCHED_AT_KEY: _days_ago(90),
            },
        )
    )
    endpoint.record.creation_task_id = "gc-task"
    report = await harness.run()

    assert [a.record.id for a in report.deleted] == [endpoint.record.id]
    assert report.deleted[0].reason == IDLE


@pytest.mark.asyncio
async def test_idle_parked_by_gc_with_traffic_recovers(harness, model_endpoint_1):
    endpoint = harness.add(
        _endpoint(
            model_endpoint_1,
            available=0,
            unavailable=0,
            metadata={
                GC_LAST_TRAFFIC_AT_KEY: _days_ago(100),
                GC_OBSERVED_AT_KEY: _days_ago(1),
                GC_SCALE_TO_ZERO_REQUESTED_AT_KEY: _days_ago(10),
                GC_SCALE_TO_ZERO_TASK_ID_KEY: "gc-task",
                GC_SEEN_TASK_ID_KEY: "gc-task",
                GC_TOUCHED_AT_KEY: _days_ago(10),
            },
        )
    )
    endpoint.record.creation_task_id = "gc-task"
    report = await harness.run(traffic_names={endpoint.record.name})

    stored = await harness.stored(endpoint)
    assert stored[GC_LAST_TRAFFIC_AT_KEY] == NOW.isoformat()
    assert GC_SCALE_TO_ZERO_REQUESTED_AT_KEY in stored  # still parked by GC, delete clock reset
    assert report.deleted == [] and report.deferred == []


# ---- guards --------------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_owner_parked_endpoint_is_never_touched(harness, model_endpoint_1):
    endpoint = harness.add(_endpoint(model_endpoint_1, available=0, unavailable=0))
    report = await harness.run()

    assert await harness.stored(endpoint) == {}
    assert report.tracking == [] and report.deferred == []


@pytest.mark.asyncio
async def test_traffic_source_unknown_freezes_everything(
    harness, model_endpoint_1, model_endpoint_2
):
    idle = harness.add(
        _endpoint(
            model_endpoint_1,
            available=1,
            unavailable=0,
            metadata={GC_LAST_TRAFFIC_AT_KEY: _days_ago(90), GC_OBSERVED_AT_KEY: _days_ago(1)},
        )
    )
    fresh = harness.add(_endpoint(model_endpoint_2, available=1, unavailable=0))
    report = await harness.run(traffic_names=None)

    assert report.sources_unknown == ["FakeTraffic"]
    assert report.scaled_to_zero == []
    assert [a.record.id for a in report.deferred] == [idle.record.id]
    assert await harness.stored(fresh) == {}
    assert (await harness.stored(idle))[GC_LAST_TRAFFIC_AT_KEY] == _days_ago(90)


@pytest.mark.asyncio
async def test_owner_edit_clears_state(harness, model_endpoint_1):
    endpoint = harness.add(
        _endpoint(
            model_endpoint_1,
            available=0,
            unavailable=1,
            metadata={GC_UNAVAILABLE_SINCE_KEY: _days_ago(20), GC_TOUCHED_AT_KEY: _days_ago(20)},
            last_updated_at=NOW - timedelta(days=2),
        )
    )
    report = await harness.run()

    assert [r.id for r in report.owner_reset] == [endpoint.record.id]
    assert _keys(await harness.stored(endpoint)) == {GC_LAST_TRAFFIC_AT_KEY}


@pytest.mark.asyncio
async def test_builder_writes_after_scale_request_are_not_owner_edits(harness, model_endpoint_1):
    endpoint = harness.add(
        _endpoint(
            model_endpoint_1,
            available=0,
            unavailable=1,
            metadata={
                GC_UNAVAILABLE_SINCE_KEY: _days_ago(31),
                GC_SCALE_TO_ZERO_REQUESTED_AT_KEY: _days_ago(0.5),
                GC_SCALE_TO_ZERO_TASK_ID_KEY: "gc-task",
                GC_SEEN_TASK_ID_KEY: "gc-task",
                GC_TOUCHED_AT_KEY: _days_ago(0.5),
            },
            last_updated_at=NOW - timedelta(hours=6),
            min_workers=0,
        )
    )
    endpoint.record.creation_task_id = "gc-task"
    report = await harness.run()

    assert report.owner_reset == []
    assert [r.id for r in report.tracking] == [endpoint.record.id]
    assert report.scaled_to_zero == []  # no second request; next step is the delete at day 90


@pytest.mark.parametrize(
    "status", [ModelEndpointStatus.UPDATE_PENDING, ModelEndpointStatus.UPDATE_IN_PROGRESS]
)
@pytest.mark.asyncio
async def test_in_flight_is_skipped(harness, model_endpoint_1, status):
    endpoint = harness.add(
        _endpoint(
            model_endpoint_1,
            available=0,
            unavailable=1,
            status=status,
            metadata={GC_UNAVAILABLE_SINCE_KEY: _days_ago(60)},
        )
    )
    report = await harness.run()

    assert [r.id for r in report.in_flight] == [endpoint.record.id]
    assert report.deferred == [] and report.scaled_to_zero == []


@pytest.mark.parametrize(
    "value,exempt", [(True, True), ("true", True), ("false", False), (1, False)]
)
@pytest.mark.asyncio
async def test_exempt_semantics(harness, model_endpoint_1, value, exempt):
    endpoint = harness.add(
        _endpoint(model_endpoint_1, available=0, unavailable=1, metadata={GC_EXEMPT_KEY: value})
    )
    report = await harness.run()

    assert ([r.id for r in report.exempt] == [endpoint.record.id]) is exempt
    assert (GC_UNAVAILABLE_SINCE_KEY in await harness.stored(endpoint)) is not exempt


@pytest.mark.asyncio
async def test_unreadable_state_left_alone(harness, model_endpoint_1):
    endpoint = harness.add(
        _endpoint(
            model_endpoint_1, available=0, unavailable=1, metadata={GC_UNAVAILABLE_SINCE_KEY: "x"}
        )
    )
    report = await harness.run()

    assert [r.id for r in report.state_invalid] == [endpoint.record.id]
    assert (await harness.stored(endpoint))[GC_UNAVAILABLE_SINCE_KEY] == "x"


@pytest.mark.asyncio
async def test_no_deployment_keeps_state_and_never_acts(harness, model_endpoint_1):
    endpoint = harness.add(
        _endpoint(
            model_endpoint_1,
            available=0,
            unavailable=1,
            metadata={GC_UNAVAILABLE_SINCE_KEY: _days_ago(60)},
        ),
        with_resources=False,
    )
    report = await harness.run()

    assert [r.id for r in report.no_deployment] == [endpoint.record.id]
    assert (await harness.stored(endpoint))[GC_UNAVAILABLE_SINCE_KEY] == _days_ago(60)
    assert report.deleted == [] and report.deferred == []


# ---- actions -------------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_observe_only_defers_but_still_writes_clocks(
    harness, model_endpoint_1, model_endpoint_2
):
    due = harness.add(
        _endpoint(
            model_endpoint_1,
            available=0,
            unavailable=1,
            metadata={GC_UNAVAILABLE_SINCE_KEY: _days_ago(40)},
        )
    )
    fresh = harness.add(_endpoint(model_endpoint_2, available=0, unavailable=1))
    report = await harness.run(config=EndpointGcConfig(actions_enabled=False))

    assert [a.record.id for a in report.deferred] == [due.record.id]
    assert report.scaled_to_zero == []
    assert GC_UNAVAILABLE_SINCE_KEY in await harness.stored(fresh)
    assert "OBSERVE ONLY" in harness.digest.digests[0]


@pytest.mark.asyncio
async def test_action_cap_oldest_due_first(harness, model_endpoint_1, model_endpoint_2):
    older = harness.add(
        _endpoint(
            model_endpoint_1,
            available=0,
            unavailable=1,
            metadata={GC_UNAVAILABLE_SINCE_KEY: _days_ago(50)},
        )
    )
    newer = harness.add(
        _endpoint(
            model_endpoint_2,
            available=0,
            unavailable=1,
            metadata={GC_UNAVAILABLE_SINCE_KEY: _days_ago(31)},
        )
    )
    report = await harness.run(config=EndpointGcConfig(actions_enabled=True, action_cap=1))

    assert [a.record.id for a in report.scaled_to_zero] == [older.record.id]
    assert [a.record.id for a in report.deferred] == [newer.record.id]


@pytest.mark.asyncio
async def test_scale_to_zero_calls_update_with_min_workers_zero(harness, model_endpoint_1):
    endpoint = harness.add(
        _endpoint(
            model_endpoint_1,
            available=0,
            unavailable=1,
            metadata={GC_UNAVAILABLE_SINCE_KEY: _days_ago(30)},
        )
    )
    calls = []
    original = harness.service.update_model_endpoint

    async def spy(**kwargs):
        calls.append(kwargs)
        return await original(**kwargs)

    harness.service.update_model_endpoint = spy
    await harness.run()

    assert calls == [{"model_endpoint_id": endpoint.record.id, "min_workers": 0}]


@pytest.mark.asyncio
async def test_metadata_write_keeps_concurrent_user_keys(harness, model_endpoint_1):
    endpoint = harness.add(
        _endpoint(
            model_endpoint_1, available=0, unavailable=1, metadata={"_llm": {"model_name": "m"}}
        )
    )
    original_list = harness.repo.list_model_endpoint_records

    async def list_then_mutate(**kwargs):
        records = await original_list(**kwargs)
        stored = harness.repo.db[endpoint.record.id]
        stored.metadata = {**stored.metadata, "user_key": "added-mid-run"}
        return records

    harness.repo.list_model_endpoint_records = list_then_mutate
    await harness.run()

    stored = await harness.stored(endpoint)
    assert stored["user_key"] == "added-mid-run"
    assert stored["_llm"] == {"model_name": "m"}
    assert GC_UNAVAILABLE_SINCE_KEY in stored


@pytest.mark.asyncio
async def test_locked_endpoint_skips_metadata_write(harness, model_endpoint_1):
    endpoint = harness.add(_endpoint(model_endpoint_1, available=0, unavailable=1))
    harness.repo.force_lock_model_endpoint(endpoint.record)
    report = await harness.run()

    assert await harness.stored(endpoint) == {}
    assert [r.id for r in report.write_skipped] == [endpoint.record.id]


@pytest.mark.asyncio
async def test_digest_lists_action_with_date_and_owner_fields(harness, model_endpoint_1):
    endpoint = harness.add(
        _endpoint(
            model_endpoint_1,
            available=0,
            unavailable=1,
            metadata={GC_UNAVAILABLE_SINCE_KEY: _days_ago(16)},
        )
    )
    await harness.run()

    text = harness.digest.digests[0]
    assert "In 14 days (1):" in text
    assert f"scale to zero [broken] {(NOW + timedelta(days=14)).strftime('%Y-%m-%d')}" in text
    assert f"created_by={endpoint.record.created_by}" in text
    assert f"owner={endpoint.record.owner}" in text


@pytest.mark.asyncio
async def test_revived_after_scale_request_gets_a_fresh_idle_clock(harness, model_endpoint_1):
    # Owner raised min_workers back above zero within GC's attribution window: that is an owner
    # edit however it is timed, and the old clock must not fire again tomorrow.
    endpoint = harness.add(
        _endpoint(
            model_endpoint_1,
            available=1,
            unavailable=0,
            min_workers=1,
            metadata={
                GC_LAST_TRAFFIC_AT_KEY: _days_ago(100),
                GC_OBSERVED_AT_KEY: _days_ago(1),
                GC_SCALE_TO_ZERO_REQUESTED_AT_KEY: _days_ago(0.5),
                GC_TOUCHED_AT_KEY: _days_ago(0.5),
            },
            last_updated_at=NOW - timedelta(hours=6),
        )
    )
    report = await harness.run()

    assert _keys(await harness.stored(endpoint)) == {GC_LAST_TRAFFIC_AT_KEY}
    assert [r.id for r in report.owner_reset] == [endpoint.record.id]
    assert report.deleted == [] and report.deferred == [] and report.scaled_to_zero == []


@pytest.mark.asyncio
async def test_idle_scale_request_failed_in_builder_still_deletes(harness, model_endpoint_1):
    endpoint = harness.add(
        _endpoint(
            model_endpoint_1,
            available=1,
            unavailable=0,
            min_workers=1,
            status=ModelEndpointStatus.UPDATE_FAILED,
            metadata={
                GC_LAST_TRAFFIC_AT_KEY: _days_ago(180),
                GC_OBSERVED_AT_KEY: _days_ago(1),
                GC_SCALE_TO_ZERO_REQUESTED_AT_KEY: _days_ago(90),
                GC_SCALE_TO_ZERO_TASK_ID_KEY: "gc-task",
                GC_SEEN_TASK_ID_KEY: "gc-task",
                GC_TOUCHED_AT_KEY: _days_ago(90),
            },
            last_updated_at=NOW - timedelta(days=89, hours=23),
        )
    )
    endpoint.record.creation_task_id = "gc-task"
    report = await harness.run()

    assert [a.record.id for a in report.deleted] == [endpoint.record.id]
    assert report.deleted[0].reason == IDLE


@pytest.mark.asyncio
async def test_keda_wake_keeps_gc_parked_endpoint_tracked(harness, model_endpoint_1):
    # A request woke the GC-parked endpoint (pods up, min_workers still 0). It stays ours: the
    # clock refreshes from the traffic and the delete is pushed out, not forgotten.
    endpoint = harness.add(
        _endpoint(
            model_endpoint_1,
            available=1,
            unavailable=0,
            min_workers=0,
            metadata={
                GC_LAST_TRAFFIC_AT_KEY: _days_ago(120),
                GC_OBSERVED_AT_KEY: _days_ago(1),
                GC_SCALE_TO_ZERO_REQUESTED_AT_KEY: _days_ago(30),
                GC_SCALE_TO_ZERO_TASK_ID_KEY: "gc-task",
                GC_SEEN_TASK_ID_KEY: "gc-task",
                GC_TOUCHED_AT_KEY: _days_ago(30),
            },
        )
    )
    endpoint.record.creation_task_id = "gc-task"
    report = await harness.run(traffic_names={endpoint.record.name})

    stored = await harness.stored(endpoint)
    assert stored[GC_LAST_TRAFFIC_AT_KEY] == NOW.isoformat()
    assert GC_SCALE_TO_ZERO_REQUESTED_AT_KEY in stored
    assert report.owner_reset == [] and report.deleted == []


@pytest.mark.asyncio
async def test_seeded_state_gets_touched_stamp(harness, model_endpoint_1):
    endpoint = harness.add(
        _endpoint(
            model_endpoint_1,
            available=0,
            unavailable=1,
            metadata={GC_UNAVAILABLE_SINCE_KEY: _days_ago(40)},
        )
    )
    await harness.run(config=EndpointGcConfig(actions_enabled=False))

    stored = await harness.stored(endpoint)
    assert stored[GC_UNAVAILABLE_SINCE_KEY] == _days_ago(40)
    assert GC_TOUCHED_AT_KEY in stored


@pytest.mark.asyncio
async def test_http_scale_to_zero_unsupported_is_reported_not_acted(harness, model_endpoint_1):
    endpoint = harness.add(
        _endpoint(
            model_endpoint_1,
            available=0,
            unavailable=1,
            metadata={GC_UNAVAILABLE_SINCE_KEY: _days_ago(30)},
        )
    )
    report = await harness.run(
        config=EndpointGcConfig(actions_enabled=True, http_scale_to_zero_supported=False)
    )

    assert [a.record.id for a in report.unsupported] == [endpoint.record.id]
    assert report.scaled_to_zero == []


@pytest.mark.asyncio
async def test_locked_endpoint_does_not_get_scaled_without_its_stamp(harness, model_endpoint_1):
    endpoint = harness.add(
        _endpoint(
            model_endpoint_1,
            available=0,
            unavailable=1,
            metadata={GC_UNAVAILABLE_SINCE_KEY: _days_ago(30), GC_TOUCHED_AT_KEY: _days_ago(30)},
        )
    )
    harness.repo.force_lock_model_endpoint(endpoint.record)
    report = await harness.run()

    assert report.scaled_to_zero == [] and report.deferred == []
    assert [r.id for r in report.write_skipped] == [endpoint.record.id]
    assert GC_SCALE_TO_ZERO_REQUESTED_AT_KEY not in await harness.stored(endpoint)


@pytest.mark.asyncio
async def test_parked_broken_async_with_new_messages_is_not_deleted(harness, model_endpoint_1):
    endpoint = harness.add(
        _endpoint(
            model_endpoint_1,
            available=0,
            unavailable=0,
            endpoint_type=ModelEndpointType.ASYNC,
            metadata={
                GC_UNAVAILABLE_SINCE_KEY: _days_ago(90),
                GC_SCALE_TO_ZERO_REQUESTED_AT_KEY: _days_ago(60),
                GC_SCALE_TO_ZERO_TASK_ID_KEY: "gc-task",
                GC_SEEN_TASK_ID_KEY: "gc-task",
                GC_TOUCHED_AT_KEY: _days_ago(60),
            },
        )
    )
    endpoint.record.creation_task_id = "gc-task"
    report = await harness.run(queue_sent=3)

    stored = await harness.stored(endpoint)
    assert report.deleted == []
    assert GC_UNAVAILABLE_SINCE_KEY not in stored
    assert stored[GC_LAST_TRAFFIC_AT_KEY] == NOW.isoformat()
    assert [r.id for r in report.recovered] == [endpoint.record.id]


@pytest.mark.asyncio
async def test_frozen_run_still_records_traffic_from_sources_that_answered(
    harness, model_endpoint_1
):
    endpoint = harness.add(
        _endpoint(
            model_endpoint_1,
            available=1,
            unavailable=0,
            metadata={
                GC_LAST_TRAFFIC_AT_KEY: _days_ago(80),
                GC_OBSERVED_AT_KEY: _days_ago(1),
                GC_TOUCHED_AT_KEY: _days_ago(80),
            },
        )
    )
    gc = EndpointGarbageCollectionService(
        model_endpoint_record_repository=harness.repo,
        resource_gateway=harness.resources,
        queue_delegate=FakeQueue(0),
        traffic_gateways=[FakeTraffic({endpoint.record.name}), FakeTraffic(None)],
        model_endpoint_service=harness.service,
        digest_gateway=harness.digest,
        config=ACTING,
        now=lambda: NOW,
    )
    report = await gc.execute()

    assert report.sources_unknown == ["FakeTraffic"]
    assert (await harness.stored(endpoint))[GC_LAST_TRAFFIC_AT_KEY] == NOW.isoformat()


@pytest.mark.asyncio
async def test_async_first_sight_ignores_http_history(harness, model_endpoint_1):
    endpoint = harness.add(
        _endpoint(
            model_endpoint_1, available=1, unavailable=0, endpoint_type=ModelEndpointType.ASYNC
        )
    )
    await harness.run(history={endpoint.record.name: NOW - timedelta(days=150)})

    assert (await harness.stored(endpoint))[GC_LAST_TRAFFIC_AT_KEY] == NOW.isoformat()


@pytest.mark.asyncio
async def test_owner_update_with_own_task_id_is_detected(harness, model_endpoint_1):
    # Owner changed the bundle hours after GC's request: the builder task id differs from ours.
    endpoint = harness.add(
        _endpoint(
            model_endpoint_1,
            available=0,
            unavailable=0,
            min_workers=0,
            metadata={
                GC_UNAVAILABLE_SINCE_KEY: _days_ago(40),
                GC_SCALE_TO_ZERO_REQUESTED_AT_KEY: _days_ago(0.5),
                GC_SCALE_TO_ZERO_TASK_ID_KEY: "gc-task",
                GC_SEEN_TASK_ID_KEY: "gc-task",
                GC_TOUCHED_AT_KEY: _days_ago(0.5),
            },
            last_updated_at=NOW - timedelta(hours=6),
        )
    )
    endpoint.record.creation_task_id = "owner-task"
    report = await harness.run()

    assert [r.id for r in report.owner_reset] == [endpoint.record.id]
    assert _keys(await harness.stored(endpoint)) == {GC_LAST_TRAFFIC_AT_KEY}


@pytest.mark.asyncio
async def test_exempt_set_after_listing_blocks_the_action(harness, model_endpoint_1):
    endpoint = harness.add(
        _endpoint(
            model_endpoint_1,
            available=0,
            unavailable=1,
            metadata={GC_UNAVAILABLE_SINCE_KEY: _days_ago(30), GC_TOUCHED_AT_KEY: _days_ago(30)},
        )
    )
    original_list = harness.repo.list_model_endpoint_records

    async def list_then_exempt(**kwargs):
        records = await original_list(**kwargs)
        # Owner PUT lands after the listing: the stored row changes, the listed copy does not.
        stored = harness.repo.db[endpoint.record.id]
        harness.repo.db[endpoint.record.id] = stored.model_copy(
            update={"metadata": {**stored.metadata, GC_EXEMPT_KEY: True}}
        )
        return records

    harness.repo.list_model_endpoint_records = list_then_exempt
    report = await harness.run()

    assert report.scaled_to_zero == []
    assert [a.record.id for a in report.skipped_at_action] == [endpoint.record.id]


@pytest.mark.asyncio
async def test_failed_scale_request_is_reconciled_next_run(harness, model_endpoint_1):
    endpoint = harness.add(
        _endpoint(
            model_endpoint_1,
            available=0,
            unavailable=1,
            metadata={GC_UNAVAILABLE_SINCE_KEY: _days_ago(30), GC_TOUCHED_AT_KEY: _days_ago(30)},
        )
    )

    async def boom(**kwargs):
        raise RuntimeError("builder unreachable")

    original_update = harness.service.update_model_endpoint
    harness.service.update_model_endpoint = boom
    report = await harness.run()

    stored = await harness.stored(endpoint)
    assert [a.record.id for a in report.action_failed] == [endpoint.record.id]
    assert GC_SCALE_TO_ZERO_REQUESTED_AT_KEY in stored  # ambiguous: kept for reconciliation
    assert GC_SCALE_TO_ZERO_TASK_ID_KEY not in stored

    # Next run: the record's task id is unchanged, so the request is dropped and retried.
    harness.service.update_model_endpoint = original_update
    report = await harness.run()
    assert [a.record.id for a in report.scaled_to_zero] == [endpoint.record.id]


@pytest.mark.asyncio
async def test_scale_request_records_builder_task_id(harness, model_endpoint_1):
    endpoint = harness.add(
        _endpoint(
            model_endpoint_1,
            available=0,
            unavailable=1,
            metadata={GC_UNAVAILABLE_SINCE_KEY: _days_ago(30), GC_TOUCHED_AT_KEY: _days_ago(30)},
        )
    )
    await harness.run()

    assert (await harness.stored(endpoint))[GC_SCALE_TO_ZERO_TASK_ID_KEY] == "test_creation_task_id"


@pytest.mark.asyncio
async def test_unsupported_actions_do_not_consume_the_cap(
    harness, model_endpoint_1, model_endpoint_2
):
    http = harness.add(
        _endpoint(
            model_endpoint_1,
            available=0,
            unavailable=1,
            metadata={GC_UNAVAILABLE_SINCE_KEY: _days_ago(50)},
        )
    )
    queue = harness.add(
        _endpoint(
            model_endpoint_2,
            available=0,
            unavailable=1,
            endpoint_type=ModelEndpointType.ASYNC,
            metadata={GC_UNAVAILABLE_SINCE_KEY: _days_ago(31)},
        )
    )
    report = await harness.run(
        config=EndpointGcConfig(
            actions_enabled=True, action_cap=1, http_scale_to_zero_supported=False
        )
    )

    assert [a.record.id for a in report.unsupported] == [http.record.id]
    assert [a.record.id for a in report.scaled_to_zero] == [queue.record.id]


@pytest.mark.asyncio
async def test_owner_edit_two_minutes_after_gc_write_is_detected_by_task_id(
    harness, model_endpoint_1
):
    endpoint = harness.add(
        _endpoint(
            model_endpoint_1,
            available=0,
            unavailable=0,
            min_workers=0,
            metadata={
                GC_UNAVAILABLE_SINCE_KEY: _days_ago(90),
                GC_SCALE_TO_ZERO_REQUESTED_AT_KEY: _days_ago(60),
                GC_SCALE_TO_ZERO_TASK_ID_KEY: "gc-task",
                GC_SEEN_TASK_ID_KEY: "gc-task",
                GC_TOUCHED_AT_KEY: _days_ago(1),
            },
            last_updated_at=NOW - timedelta(days=1) + timedelta(minutes=2),
        )
    )
    endpoint.record.creation_task_id = "owner-task"
    report = await harness.run()

    assert report.deleted == []
    assert [r.id for r in report.owner_reset] == [endpoint.record.id]


@pytest.mark.asyncio
async def test_frozen_run_still_drops_broken_clock_when_workers_are_back(harness, model_endpoint_1):
    endpoint = harness.add(
        _endpoint(
            model_endpoint_1,
            available=1,
            unavailable=0,
            min_workers=0,
            metadata={
                GC_UNAVAILABLE_SINCE_KEY: _days_ago(89),
                GC_SCALE_TO_ZERO_REQUESTED_AT_KEY: _days_ago(59),
                GC_SCALE_TO_ZERO_TASK_ID_KEY: "gc-task",
                GC_SEEN_TASK_ID_KEY: "gc-task",
                GC_TOUCHED_AT_KEY: _days_ago(59),
            },
        )
    )
    endpoint.record.creation_task_id = "gc-task"
    report = await harness.run(traffic_names=None)

    assert GC_UNAVAILABLE_SINCE_KEY not in await harness.stored(endpoint)
    assert [r.id for r in report.recovered] == [endpoint.record.id]


@pytest.mark.asyncio
async def test_observation_gap_pauses_the_idle_clock(harness, model_endpoint_1):
    endpoint = harness.add(
        _endpoint(
            model_endpoint_1,
            available=1,
            unavailable=0,
            metadata={
                GC_LAST_TRAFFIC_AT_KEY: _days_ago(90),
                GC_OBSERVED_AT_KEY: _days_ago(10),
                GC_TOUCHED_AT_KEY: _days_ago(10),
                GC_SEEN_TASK_ID_KEY: "test_creation_task_id",
            },
        )
    )
    report = await harness.run()

    stored = await harness.stored(endpoint)
    # Silence is only known for the current lookback; without history the clock restarts there.
    expected = NOW - timedelta(hours=36)
    assert stored[GC_LAST_TRAFFIC_AT_KEY] == expected.isoformat()
    assert stored[GC_OBSERVED_AT_KEY] == NOW.isoformat()
    assert report.scaled_to_zero == [] and report.deferred == []


@pytest.mark.asyncio
async def test_history_never_predates_the_endpoint(harness, model_endpoint_1):
    endpoint = _endpoint(model_endpoint_1, available=1, unavailable=0)
    endpoint.record.created_at = NOW - timedelta(days=3)
    harness.add(endpoint)
    await harness.run(history={endpoint.record.name: NOW - timedelta(days=150)})

    assert (await harness.stored(endpoint))[GC_LAST_TRAFFIC_AT_KEY] == (
        NOW - timedelta(days=3)
    ).isoformat()


@pytest.mark.asyncio
async def test_ambiguous_scale_failure_never_adopts_a_foreign_task_id(harness, model_endpoint_1):
    endpoint = harness.add(
        _endpoint(
            model_endpoint_1,
            available=0,
            unavailable=1,
            metadata={GC_UNAVAILABLE_SINCE_KEY: _days_ago(30), GC_TOUCHED_AT_KEY: _days_ago(30)},
        )
    )

    async def enqueue_then_fail(**kwargs):
        harness.repo.db[endpoint.record.id].creation_task_id = "builder-task"
        raise RuntimeError("record update failed after enqueue")

    harness.service.update_model_endpoint = enqueue_then_fail
    report = await harness.run()

    stored = await harness.stored(endpoint)
    assert [a.record.id for a in report.action_failed] == [endpoint.record.id]
    assert GC_SCALE_TO_ZERO_REQUESTED_AT_KEY in stored

    # Next run: the task id changed and GC cannot prove the build was its own, so it treats
    # the change as an owner edit and starts over rather than claiming the parked endpoint.
    report = await harness.run()
    stored = await harness.stored(endpoint)
    assert GC_SCALE_TO_ZERO_TASK_ID_KEY not in stored
    assert GC_SCALE_TO_ZERO_REQUESTED_AT_KEY not in stored
    assert [r.id for r in report.owner_reset] == [endpoint.record.id]


@pytest.mark.asyncio
async def test_owner_edit_between_listing_and_write_cancels_the_action(harness, model_endpoint_1):
    endpoint = harness.add(
        _endpoint(
            model_endpoint_1,
            available=1,
            unavailable=0,
            metadata={
                GC_LAST_TRAFFIC_AT_KEY: _days_ago(95),
                GC_OBSERVED_AT_KEY: _days_ago(1),
                GC_TOUCHED_AT_KEY: _days_ago(1),
                GC_SEEN_TASK_ID_KEY: "test_creation_task_id",
            },
        )
    )
    original_list = harness.repo.list_model_endpoint_records

    async def list_then_owner_update(**kwargs):
        records = await original_list(**kwargs)
        harness.repo.db[endpoint.record.id].creation_task_id = "owner-task"
        return records

    harness.repo.list_model_endpoint_records = list_then_owner_update
    report = await harness.run()

    assert report.scaled_to_zero == [] and report.deferred == []
    assert [r.id for r in report.owner_reset] == [endpoint.record.id]
    assert (await harness.stored(endpoint))[GC_LAST_TRAFFIC_AT_KEY] == NOW.isoformat()


@pytest.mark.asyncio
async def test_queue_unknown_does_not_block_infra_recovery(harness, model_endpoint_1):
    endpoint = harness.add(
        _endpoint(
            model_endpoint_1,
            available=1,
            unavailable=0,
            min_workers=0,
            endpoint_type=ModelEndpointType.ASYNC,
            metadata={
                GC_UNAVAILABLE_SINCE_KEY: _days_ago(89),
                GC_SCALE_TO_ZERO_REQUESTED_AT_KEY: _days_ago(59),
                GC_SCALE_TO_ZERO_TASK_ID_KEY: "gc-task",
                GC_SEEN_TASK_ID_KEY: "gc-task",
                GC_TOUCHED_AT_KEY: _days_ago(59),
            },
        )
    )
    endpoint.record.creation_task_id = "gc-task"
    report = await harness.run(queue_sent=None)

    assert GC_UNAVAILABLE_SINCE_KEY not in await harness.stored(endpoint)
    assert [r.id for r in report.queue_unknown] == [endpoint.record.id]
    assert report.deleted == []


@pytest.mark.asyncio
async def test_observation_gap_uses_history_when_available(harness, model_endpoint_1):
    endpoint = harness.add(
        _endpoint(
            model_endpoint_1,
            available=1,
            unavailable=0,
            metadata={
                GC_LAST_TRAFFIC_AT_KEY: _days_ago(89),
                GC_OBSERVED_AT_KEY: _days_ago(4),
                GC_TOUCHED_AT_KEY: _days_ago(4),
                GC_SEEN_TASK_ID_KEY: "test_creation_task_id",
            },
        )
    )
    report = await harness.run(history={endpoint.record.name: NOW - timedelta(days=2)})

    assert (await harness.stored(endpoint))[GC_LAST_TRAFFIC_AT_KEY] == (
        NOW - timedelta(days=2)
    ).isoformat()
    assert report.scaled_to_zero == []


@pytest.mark.asyncio
async def test_owner_edit_during_seeded_state_bookkeeping_cancels_the_action(
    harness, model_endpoint_1
):
    endpoint = harness.add(
        _endpoint(
            model_endpoint_1,
            available=0,
            unavailable=1,
            metadata={GC_UNAVAILABLE_SINCE_KEY: _days_ago(40), GC_TOUCHED_AT_KEY: _days_ago(40)},
            last_updated_at=NOW - timedelta(days=40),
        )
    )
    original_list = harness.repo.list_model_endpoint_records

    async def list_then_owner_update(**kwargs):
        records = await original_list(**kwargs)
        row = harness.repo.db[endpoint.record.id]
        harness.repo.db[endpoint.record.id] = row.model_copy(
            update={"last_updated_at": NOW - timedelta(minutes=1)}
        )
        return records

    harness.repo.list_model_endpoint_records = list_then_owner_update
    report = await harness.run()

    assert report.scaled_to_zero == [] and report.deferred == []
    assert [r.id for r in report.owner_reset] == [endpoint.record.id]


@pytest.mark.asyncio
async def test_owner_update_after_listing_during_pending_reconciliation_resets(
    harness, model_endpoint_1
):
    endpoint = harness.add(
        _endpoint(
            model_endpoint_1,
            available=0,
            unavailable=1,
            min_workers=1,
            metadata={
                GC_UNAVAILABLE_SINCE_KEY: _days_ago(40),
                GC_SCALE_TO_ZERO_REQUESTED_AT_KEY: _days_ago(1),
                GC_SEEN_TASK_ID_KEY: "test_creation_task_id",
                GC_TOUCHED_AT_KEY: _days_ago(1),
            },
        )
    )
    original_list = harness.repo.list_model_endpoint_records

    async def list_then_owner_update(**kwargs):
        records = await original_list(**kwargs)
        row = harness.repo.db[endpoint.record.id]
        harness.repo.db[endpoint.record.id] = row.model_copy(
            update={"creation_task_id": "owner-task"}
        )
        return records

    harness.repo.list_model_endpoint_records = list_then_owner_update
    report = await harness.run()

    assert report.scaled_to_zero == [] and report.deferred == []
    assert [r.id for r in report.owner_reset] == [endpoint.record.id]
    stored = await harness.stored(endpoint)
    assert (
        GC_UNAVAILABLE_SINCE_KEY not in stored and GC_SCALE_TO_ZERO_REQUESTED_AT_KEY not in stored
    )


@pytest.mark.parametrize(
    "depth,in_flight,delayed,deleted",
    [
        pytest.param(0, 0, 0, True, id="empty"),
        pytest.param(3, 0, 0, False, id="visible"),
        pytest.param(0, 3, 0, False, id="in-flight"),
        pytest.param(0, 0, 3, False, id="delayed"),
    ],
)
@pytest.mark.asyncio
async def test_async_delete_requires_an_empty_queue(
    harness, model_endpoint_1, depth, in_flight, delayed, deleted
):
    endpoint = harness.add(
        _endpoint(
            model_endpoint_1,
            available=0,
            unavailable=0,
            endpoint_type=ModelEndpointType.ASYNC,
            metadata={
                GC_UNAVAILABLE_SINCE_KEY: _days_ago(90),
                GC_SCALE_TO_ZERO_REQUESTED_AT_KEY: _days_ago(60),
                GC_SCALE_TO_ZERO_TASK_ID_KEY: "gc-task",
                GC_SEEN_TASK_ID_KEY: "gc-task",
                GC_TOUCHED_AT_KEY: _days_ago(60),
            },
        )
    )
    endpoint.record.creation_task_id = "gc-task"
    report = await harness.run(queue_depth=depth, queue_in_flight=in_flight, queue_delayed=delayed)

    assert ([a.record.id for a in report.deleted] == [endpoint.record.id]) is deleted
    assert ([a.record.id for a in report.skipped_at_action] == [endpoint.record.id]) is not deleted


@pytest.mark.asyncio
async def test_multinode_endpoint_is_never_scaled_to_zero(harness, model_endpoint_1):
    endpoint = _endpoint(
        model_endpoint_1,
        available=1,
        unavailable=0,
        metadata={GC_LAST_TRAFFIC_AT_KEY: _days_ago(90), GC_OBSERVED_AT_KEY: _days_ago(1)},
    )
    resource_state = endpoint.infra_state.resource_state.model_copy(update={"nodes_per_worker": 2})
    endpoint = ModelEndpoint(
        record=endpoint.record,
        infra_state=endpoint.infra_state.model_copy(update={"resource_state": resource_state}),
    )
    harness.add(endpoint)
    report = await harness.run()

    assert report.scaled_to_zero == []
    assert [a.record.id for a in report.unsupported] == [endpoint.record.id]
