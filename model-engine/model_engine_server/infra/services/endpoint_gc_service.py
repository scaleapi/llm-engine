"""Lifecycle garbage collection for model endpoints.

Two clocks, both kept in the endpoint's ``endpoint_metadata`` JSONB so state is deleted with the
row, visible through the API, and seedable with a single UPDATE:

- Broken: ``_gc_unavailable_since`` starts on the first run that sees the Deployment with workers
  but none available (async endpoints must also have a silent queue). Scale to zero at
  ``broken_scale_to_zero_days``, delete at ``broken_delete_days``.
- Idle: ``_gc_last_traffic_at`` is the last time any traffic source saw a request. Healthy
  endpoints are scaled to zero at ``idle_scale_to_zero_days`` of silence and deleted at
  ``idle_delete_days``.

Other keys: ``_gc_scale_to_zero_requested_at`` marks GC's own scale-to-zero update so the resulting
record changes are not read as an owner edit; ``_gc_touched_at`` is GC's last write, likewise;
``_gc_exempt`` opts an endpoint out.

GC only trusts its own observations. Clocks are wall clock from the stamps, not run counts. Any
traffic source that cannot answer freezes every clock for that run. An owner edit clears all
stamps. Scaled-to-zero endpoints without GC stamps are the owner's business and never touched.
"""

import asyncio
from contextlib import AsyncExitStack
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Callable, Dict, List, Optional, Sequence, Set, Tuple

from model_engine_server.common.constants import (
    ENDPOINT_GC_EXEMPT_KEY,
    ENDPOINT_GC_LAST_TRAFFIC_AT_KEY,
    ENDPOINT_GC_OBSERVED_AT_KEY,
    ENDPOINT_GC_PARKED_AT_KEY,
    ENDPOINT_GC_SCALE_TO_ZERO_REQUESTED_AT_KEY,
    ENDPOINT_GC_SCALE_TO_ZERO_TASK_ID_KEY,
    ENDPOINT_GC_SEEN_RESTART_AT_KEY,
    ENDPOINT_GC_SEEN_TASK_ID_KEY,
    ENDPOINT_GC_TOUCHED_AT_KEY,
    ENDPOINT_GC_UNAVAILABLE_SINCE_KEY,
)
from model_engine_server.core.loggers import logger_name, make_logger
from model_engine_server.domain.entities import (
    ModelEndpointInfraState,
    ModelEndpointRecord,
    ModelEndpointStatus,
    ModelEndpointType,
)
from model_engine_server.domain.gateways import DigestGateway, EndpointTrafficGateway, TrafficKey
from model_engine_server.domain.services import ModelEndpointService
from model_engine_server.infra.gateways.resources.endpoint_resource_gateway import (
    EndpointResourceGateway,
)
from model_engine_server.infra.gateways.resources.queue_endpoint_resource_delegate import (
    QueueEndpointResourceDelegate,
)
from model_engine_server.infra.repositories.model_endpoint_record_repository import (
    ModelEndpointRecordRepository,
)

logger = make_logger(logger_name())

GC_UNAVAILABLE_SINCE_KEY = ENDPOINT_GC_UNAVAILABLE_SINCE_KEY
GC_LAST_TRAFFIC_AT_KEY = ENDPOINT_GC_LAST_TRAFFIC_AT_KEY
GC_SCALE_TO_ZERO_REQUESTED_AT_KEY = ENDPOINT_GC_SCALE_TO_ZERO_REQUESTED_AT_KEY
GC_SCALE_TO_ZERO_TASK_ID_KEY = ENDPOINT_GC_SCALE_TO_ZERO_TASK_ID_KEY
GC_SEEN_TASK_ID_KEY = ENDPOINT_GC_SEEN_TASK_ID_KEY
GC_SEEN_RESTART_AT_KEY = ENDPOINT_GC_SEEN_RESTART_AT_KEY
GC_PARKED_AT_KEY = ENDPOINT_GC_PARKED_AT_KEY
GC_OBSERVED_AT_KEY = ENDPOINT_GC_OBSERVED_AT_KEY
GC_TOUCHED_AT_KEY = ENDPOINT_GC_TOUCHED_AT_KEY
GC_EXEMPT_KEY = ENDPOINT_GC_EXEMPT_KEY
# Bookkeeping GC rewrites on every write; not clocks.
GC_BOOKKEEPING_KEYS = (GC_TOUCHED_AT_KEY, GC_SEEN_TASK_ID_KEY, GC_SEEN_RESTART_AT_KEY)
GC_STATE_KEYS = (
    GC_UNAVAILABLE_SINCE_KEY,
    GC_LAST_TRAFFIC_AT_KEY,
    GC_OBSERVED_AT_KEY,
    GC_SCALE_TO_ZERO_REQUESTED_AT_KEY,
    GC_SCALE_TO_ZERO_TASK_ID_KEY,
    GC_PARKED_AT_KEY,
    GC_SEEN_TASK_ID_KEY,
    GC_SEEN_RESTART_AT_KEY,
    GC_TOUCHED_AT_KEY,
)
GC_TIMESTAMP_KEYS = (
    GC_UNAVAILABLE_SINCE_KEY,
    GC_LAST_TRAFFIC_AT_KEY,
    GC_OBSERVED_AT_KEY,
    GC_SCALE_TO_ZERO_REQUESTED_AT_KEY,
    GC_PARKED_AT_KEY,
    GC_SEEN_RESTART_AT_KEY,
    GC_TOUCHED_AT_KEY,
)

# Every owner update through the API enqueues a build and gives the record a new
# creation_task_id, so GC remembers the id it last saw (_gc_seen_task_id) and reads any other
# id as an owner edit. The timestamp rule only covers state seeded outside the API, where GC has
# not yet seen a task id: GC's own writes bump last_updated_at moments after _gc_touched_at.
OWNER_UPDATE_SLACK = timedelta(minutes=5)
QUEUE_LOOKUP_CONCURRENCY = 10
TASK_ID_WRITE_RETRIES = 5
STUCK_REQUEST_AFTER = timedelta(days=3)  # a GC scale-to-zero still in flight after this is stuck
TRAFFIC_LOOKBACK = timedelta(hours=36)  # covers a missed daily run
IN_FLIGHT_STATUSES = {
    ModelEndpointStatus.UPDATE_PENDING,
    ModelEndpointStatus.UPDATE_IN_PROGRESS,
    ModelEndpointStatus.DELETE_IN_PROGRESS,
}
NOTICE_DAYS = (14, 7, 1)
SCALE_TO_ZERO = "scale_to_zero"
DELETE = "delete"
BROKEN = "broken"
IDLE = "idle"


@dataclass(frozen=True)
class EndpointGcConfig:
    broken_scale_to_zero_days: int = 30
    broken_delete_days: int = 90
    idle_scale_to_zero_days: int = 90
    idle_delete_days: int = 180
    action_cap: int = 20  # scale-to-zero and delete actions per run, together
    actions_enabled: bool = False  # False: bookkeeping and digest only
    # Sync and streaming endpoints can only scale from zero when KEDA has a Prometheus source.
    http_scale_to_zero_supported: bool = True


@dataclass(frozen=True)
class PlannedAction:
    record: ModelEndpointRecord
    infra_state: ModelEndpointInfraState
    kind: str  # SCALE_TO_ZERO | DELETE
    reason: str  # BROKEN | IDLE
    due_at: datetime


@dataclass
class EndpointGcReport:
    scaled_to_zero: List[PlannedAction] = field(default_factory=list)
    deleted: List[PlannedAction] = field(default_factory=list)
    action_failed: List[PlannedAction] = field(default_factory=list)
    deferred: List[PlannedAction] = field(default_factory=list)  # due, but cap/disabled/frozen
    unsupported: List[PlannedAction] = field(default_factory=list)  # cluster cannot scale http to 0
    skipped_at_action: List[PlannedAction] = field(default_factory=list)  # changed since judged
    upcoming: Dict[int, List[PlannedAction]] = field(
        default_factory=lambda: {days: [] for days in NOTICE_DAYS}
    )
    tracking: List[ModelEndpointRecord] = field(default_factory=list)  # a clock is running
    owner_reset: List[ModelEndpointRecord] = field(default_factory=list)
    recovered: List[ModelEndpointRecord] = field(default_factory=list)
    no_deployment: List[ModelEndpointRecord] = field(default_factory=list)
    in_flight: List[ModelEndpointRecord] = field(default_factory=list)
    stuck: List[ModelEndpointRecord] = field(default_factory=list)  # GC request in flight too long
    exempt: List[ModelEndpointRecord] = field(default_factory=list)
    queue_unknown: List[ModelEndpointRecord] = field(default_factory=list)
    traffic_unknown: List[ModelEndpointRecord] = field(default_factory=list)  # pods not scraped
    state_invalid: List[ModelEndpointRecord] = field(default_factory=list)
    write_skipped: List[ModelEndpointRecord] = field(default_factory=list)
    judge_failed: List[ModelEndpointRecord] = field(default_factory=list)  # bookkeeping raised
    check_failed: List[PlannedAction] = field(default_factory=list)  # live re-read failed
    sources_unknown: List[str] = field(default_factory=list)
    digest_delivered: bool = True


@dataclass
class _Traffic:
    active: Set[str]  # endpoint ids with a request in the lookback, from sources that answered
    queue_active: Set[str]  # async endpoint ids with messages in the lookback
    queue_unknown: Set[str]  # async endpoint ids whose queue could not be read: skipped
    uncovered: Set[str]  # serving http endpoint ids no traffic source is observing: skipped
    last_seen: Dict[str, datetime]  # from history-capable sources, for first sightings
    frozen: bool  # a cluster-wide source could not answer: no clock starts, no action runs


@dataclass
class _Activity:
    active: Optional[bool]  # None: a source could not answer, or a serving pod is unobserved
    observed_pods: Optional[int] = None  # serving pods the coverage sources see, when asked


@dataclass
class _Attempt:
    made: bool = False  # set right before the destructive call, so cleanup failures still count


def _parse_ts(value: object) -> Optional[datetime]:
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def _is_exempt(metadata: Dict) -> bool:
    value = metadata.get(GC_EXEMPT_KEY)
    return value is True or (isinstance(value, str) and value.lower() == "true")


def _worker_counts(infra_state: ModelEndpointInfraState) -> Tuple[int, int]:
    """(desired, available). Desired is spec.replicas; the status sum is the fallback for infra
    states cached before that field existed."""
    state = infra_state.deployment_state
    available = state.available_workers or 0
    if infra_state.desired_workers is not None:
        desired = max(infra_state.desired_workers, available + (state.unavailable_workers or 0))
    else:
        desired = available + (state.unavailable_workers or 0)
    return desired, available


class EndpointGarbageCollectionService:
    def __init__(
        self,
        model_endpoint_record_repository: ModelEndpointRecordRepository,
        resource_gateway: EndpointResourceGateway,
        queue_delegate: QueueEndpointResourceDelegate,
        traffic_gateways: Sequence[EndpointTrafficGateway],
        model_endpoint_service: ModelEndpointService,
        digest_gateway: DigestGateway,
        config: EndpointGcConfig,
        now: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
    ):
        self.record_repository = model_endpoint_record_repository
        self.resource_gateway = resource_gateway
        self.queue_delegate = queue_delegate
        self.traffic_gateways = list(traffic_gateways)
        self.model_endpoint_service = model_endpoint_service
        self.digest_gateway = digest_gateway
        self.config = config
        self.now = now
        # Infra state of the endpoint being judged; writes record its restart annotation.
        self._infra_for_writes: Optional[ModelEndpointInfraState] = None

    async def execute(self) -> EndpointGcReport:
        run_at = self.now()
        report = EndpointGcReport()
        infra_states, records = await asyncio.gather(
            self.resource_gateway.get_all_resources(),
            self.record_repository.list_model_endpoint_records(
                owner=None, name=None, order_by=None
            ),
        )
        states_by_id: Dict[str, ModelEndpointInfraState] = {
            key: state for key, (is_endpoint_id, state) in infra_states.items() if is_endpoint_id
        }
        traffic = await self._collect_traffic(records, states_by_id, run_at, report)

        due: List[PlannedAction] = []
        for record in records:
            try:
                await self._judge(record, states_by_id.get(record.id), traffic, run_at, due, report)
            except Exception:
                # One endpoint's bookkeeping must not take the run, or the digest, down.
                logger.exception(f"GC bookkeeping failed for {record.id}; skipped this run")
                report.judge_failed.append(record)
        await self._act(due, traffic, run_at, report)
        try:
            report.digest_delivered = self.digest_gateway.send_digest(
                format_digest(report, self.config, run_at, states_by_id)
            )
        except Exception:
            logger.exception("GC digest delivery failed")
            report.digest_delivered = False
        return report

    async def _judge(
        self,
        record: ModelEndpointRecord,
        infra_state: Optional[ModelEndpointInfraState],
        traffic: _Traffic,
        run_at: datetime,
        due: List[PlannedAction],
        report: EndpointGcReport,
    ) -> None:
        self._infra_for_writes = None
        metadata = record.metadata or {}
        has_state = any(key in metadata for key in GC_STATE_KEYS)
        if record.status in IN_FLIGHT_STATUSES:
            if has_state:
                report.in_flight.append(record)
                requested = _parse_ts(metadata.get(GC_SCALE_TO_ZERO_REQUESTED_AT_KEY))
                if requested is not None and run_at - requested > STUCK_REQUEST_AFTER:
                    report.stuck.append(record)
            return
        if _is_exempt(metadata):
            report.exempt.append(record)
            return
        bad = [
            key
            for key in GC_TIMESTAMP_KEYS
            if key in metadata
            and not _parse_ts(metadata[key])
            and not (key == GC_SEEN_RESTART_AT_KEY and metadata[key] == "")
        ]
        if has_state and bad:
            logger.warning(f"GC state on {record.id} is not ISO-8601: {bad}")
            report.state_invalid.append(record)
            return
        if has_state and self._owner_touched(record, metadata):
            # An owner edit is activity: restart from a fresh idle clock, not from history.
            await self._write_gc_state(
                record,
                {
                    GC_LAST_TRAFFIC_AT_KEY: run_at.isoformat(),
                    GC_OBSERVED_AT_KEY: run_at.isoformat(),
                },
                run_at,
                report,
                resetting=True,
            )
            report.owner_reset.append(record)
            return
        if infra_state is None:
            # Missing from the k8s listing: gone, or unreadable this run. Report, keep state.
            if has_state:
                report.no_deployment.append(record)
            return

        # Writes from here on acknowledge the restart annotation seen in this listing.
        self._infra_for_writes = infra_state
        if has_state and self._owner_restarted(infra_state, metadata):
            # `restart_model_endpoint` only touches kubernetes; the annotation it writes is the
            # owner's activity signal.
            await self._write_gc_state(
                record,
                {
                    GC_LAST_TRAFFIC_AT_KEY: run_at.isoformat(),
                    GC_OBSERVED_AT_KEY: run_at.isoformat(),
                },
                run_at,
                report,
                resetting=True,
            )
            report.owner_reset.append(record)
            return

        desired, available = _worker_counts(infra_state)
        unavailable_since = _parse_ts(metadata.get(GC_UNAVAILABLE_SINCE_KEY))
        last_traffic_at = _parse_ts(metadata.get(GC_LAST_TRAFFIC_AT_KEY))
        requested_at = _parse_ts(metadata.get(GC_SCALE_TO_ZERO_REQUESTED_AT_KEY))
        is_async = record.endpoint_type == ModelEndpointType.ASYNC
        if requested_at and GC_SCALE_TO_ZERO_TASK_ID_KEY not in metadata:
            # A scale-to-zero whose task id never got recorded. GC cannot tell its own build
            # from an owner update by the new task id, so it never adopts one: a changed id is
            # treated as an owner edit (fresh clock, request dropped), an unchanged id drops the
            # request so the scale-to-zero is retried. Retrying min_workers=0 is idempotent.
            # Decide on the stored row, not the listing: an owner update may have landed since.
            current = (
                await self.record_repository.get_model_endpoint_record(
                    model_endpoint_id=record.id, refresh=True
                )
                or record
            )
            if (current.creation_task_id or "") != (metadata.get(GC_SEEN_TASK_ID_KEY) or ""):
                await self._write_gc_state(
                    record,
                    {
                        GC_LAST_TRAFFIC_AT_KEY: run_at.isoformat(),
                        GC_OBSERVED_AT_KEY: run_at.isoformat(),
                    },
                    run_at,
                    report,
                    resetting=True,
                )
                report.owner_reset.append(record)
                return
            if not await self._write_gc_state(
                record,
                {
                    k: v
                    for k, v in metadata.items()
                    if k in GC_STATE_KEYS and k != GC_SCALE_TO_ZERO_REQUESTED_AT_KEY
                },
                run_at,
                report,
            ):
                return
            metadata = record.metadata or {}
            requested_at = None
        if available > 0 and unavailable_since:
            # Infrastructure recovered: drop the broken clock now, whatever else is unknown.
            if not await self._write_gc_state(
                record,
                {
                    k: v
                    for k, v in metadata.items()
                    if k in GC_STATE_KEYS and k != GC_UNAVAILABLE_SINCE_KEY
                },
                run_at,
                report,
            ):
                return
            report.recovered.append(record)
            unavailable_since = None
            metadata = record.metadata or {}
        if is_async and record.id in traffic.queue_unknown:
            report.queue_unknown.append(record)
            return
        active = record.id in traffic.active
        queue_active = is_async and record.id in traffic.queue_active
        if record.id in traffic.uncovered:
            report.traffic_unknown.append(record)
            return

        if requested_at and infra_state.deployment_state.min_workers > 0:
            if record.status == ModelEndpointStatus.UPDATE_FAILED:
                # GC's scale-to-zero never took (the builder failed); the request stays and the
                # schedule ends in the delete.
                pass
            else:
                # min_workers is back above zero: the owner revived it. Fresh idle clock.
                await self._write_gc_state(
                    record,
                    {
                        GC_LAST_TRAFFIC_AT_KEY: run_at.isoformat(),
                        GC_OBSERVED_AT_KEY: run_at.isoformat(),
                    },
                    run_at,
                    report,
                    resetting=True,
                )
                report.owner_reset.append(record)
                return
        # Parked by GC means min_workers is 0 on our request, whether or not KEDA has a pod up.
        gc_parked = requested_at is not None
        keep_request = {
            key: metadata[key]
            for key in (
                GC_SCALE_TO_ZERO_REQUESTED_AT_KEY,
                GC_SCALE_TO_ZERO_TASK_ID_KEY,
                GC_PARKED_AT_KEY,
            )
            if key in metadata
        }
        if gc_parked and desired == 0 and GC_PARKED_AT_KEY not in metadata:
            # First sighting of the endpoint actually at zero after our request: the promised
            # parked period counts from here.
            keep_request[GC_PARKED_AT_KEY] = run_at.isoformat()
            if not await self._write_gc_state(
                record,
                {**{k: v for k, v in metadata.items() if k in GC_STATE_KEYS}, **keep_request},
                run_at,
                report,
            ):
                return
            metadata = record.metadata or {}
        if has_state and not all(
            key in metadata for key in (GC_TOUCHED_AT_KEY, GC_SEEN_TASK_ID_KEY)
        ):
            # Seeded state: record our write time and the task id we saw, so owner edits
            # become detectable from here on.
            if not await self._write_gc_state(
                record, {k: v for k, v in metadata.items() if k in GC_STATE_KEYS}, run_at, report
            ):
                return
            metadata = record.metadata or {}

        if desired == 0 and not gc_parked:
            # Parked by the owner: never ours to judge. Scaling down is owner activity.
            if has_state:
                await self._write_gc_state(
                    record,
                    {
                        GC_LAST_TRAFFIC_AT_KEY: run_at.isoformat(),
                        GC_OBSERVED_AT_KEY: run_at.isoformat(),
                    },
                    run_at,
                    report,
                    resetting=True,
                )
                report.recovered.append(record)
            return

        broken = desired > 0 and available == 0
        if (active or queue_active) and (available > 0 or gc_parked or (broken and queue_active)):
            # Positive evidence of use, recorded even on a frozen run: a request on a serving
            # endpoint, on one GC parked (KEDA woke it), or work queued for a dead async one.
            # Traffic to a dead sync endpoint that GC has not parked is a caller's problem.
            if unavailable_since:
                report.recovered.append(record)
            if (
                unavailable_since
                or last_traffic_at is None
                or run_at - last_traffic_at > timedelta(hours=12)
            ):
                await self._write_gc_state(
                    record,
                    {
                        GC_LAST_TRAFFIC_AT_KEY: run_at.isoformat(),
                        GC_OBSERVED_AT_KEY: run_at.isoformat(),
                        **keep_request,
                    },
                    run_at,
                    report,
                )
            if gc_parked and not (broken and queue_active):
                # Still parked on our request: the idle clock keeps running toward the delete.
                report.tracking.append(record)
                self._plan(
                    record,
                    infra_state,
                    IDLE,
                    run_at,
                    parked=True,
                    scale_days=self.config.idle_scale_to_zero_days,
                    delete_days=self.config.idle_delete_days,
                    run_at=run_at,
                    due=due,
                    report=report,
                    requested_at=requested_at,
                )
            return

        if broken or (gc_parked and desired == 0 and unavailable_since):
            # Broken, or parked by GC because it was broken.
            if traffic.frozen:
                if unavailable_since is None:
                    return
            else:
                observed_at = _parse_ts(metadata.get(GC_OBSERVED_AT_KEY))
                if unavailable_since is None:
                    unavailable_since = run_at
                elif (is_async or gc_parked) and (
                    observed_at is None or run_at - observed_at > TRAFFIC_LOOKBACK
                ):
                    # Queue activity, or a request that woke a GC-parked endpoint, during an
                    # unobserved stretch would have been missed; silence is only known since
                    # the start of the current lookback.
                    unavailable_since = max(unavailable_since, run_at - TRAFFIC_LOOKBACK)
                if not await self._write_gc_state(
                    record,
                    {
                        GC_UNAVAILABLE_SINCE_KEY: unavailable_since.isoformat(),
                        GC_OBSERVED_AT_KEY: run_at.isoformat(),
                        **keep_request,
                    },
                    run_at,
                    report,
                ):
                    return
            report.tracking.append(record)
            self._plan(
                record,
                infra_state,
                BROKEN,
                unavailable_since,
                parked=gc_parked,
                scale_days=self.config.broken_scale_to_zero_days,
                delete_days=self.config.broken_delete_days,
                run_at=run_at,
                due=due,
                report=report,
                requested_at=requested_at,
            )
            return

        # Serving (available > 0) or parked by GC for idleness, with no request seen: idle clock.
        if traffic.frozen:
            # Unobserved day: nothing starts or ages. Existing plans are shown, not acted on.
            if last_traffic_at is not None:
                report.tracking.append(record)
                self._plan_idle(
                    record, infra_state, last_traffic_at, gc_parked, run_at, due, report
                )
            return
        observed_at = _parse_ts(metadata.get(GC_OBSERVED_AT_KEY))
        if last_traffic_at is None:
            # First sight. History may push the clock back, but never before this endpoint
            # existed, never so far that the first action lands before the first notice, and
            # only for endpoints the history covers.
            floor = run_at - timedelta(days=self.config.idle_scale_to_zero_days - NOTICE_DAYS[0])
            created_at = record.created_at
            if created_at.tzinfo is None:
                created_at = created_at.replace(tzinfo=timezone.utc)
            seen = traffic.last_seen.get(record.id, run_at) if not is_async else run_at
            last_traffic_at = max(seen, floor, created_at)
        elif observed_at is None or run_at - observed_at > TRAFFIC_LOOKBACK:
            # Requests during a gap in observation (or before observation started, for a clock
            # with no boundary) would have been missed. Silence is only known since the start of
            # the current lookback.
            # No single source covers every request path, so history from one of them cannot
            # vouch for the gap.
            last_traffic_at = max(last_traffic_at, run_at - TRAFFIC_LOOKBACK)
        if not await self._write_gc_state(
            record,
            {
                GC_LAST_TRAFFIC_AT_KEY: last_traffic_at.isoformat(),
                GC_OBSERVED_AT_KEY: run_at.isoformat(),
                **keep_request,
            },
            run_at,
            report,
        ):
            return
        report.tracking.append(record)
        self._plan_idle(record, infra_state, last_traffic_at, gc_parked, run_at, due, report)

    def _plan_idle(
        self,
        record: ModelEndpointRecord,
        infra_state: ModelEndpointInfraState,
        last_traffic_at: datetime,
        parked: bool,
        run_at: datetime,
        due: List[PlannedAction],
        report: EndpointGcReport,
    ) -> None:
        self._plan(
            record,
            infra_state,
            IDLE,
            last_traffic_at,
            parked=parked,
            scale_days=self.config.idle_scale_to_zero_days,
            delete_days=self.config.idle_delete_days,
            run_at=run_at,
            due=due,
            report=report,
            requested_at=_parse_ts((record.metadata or {}).get(GC_SCALE_TO_ZERO_REQUESTED_AT_KEY)),
        )

    # ---- traffic -------------------------------------------------------------------------

    async def _collect_traffic(
        self,
        records: List[ModelEndpointRecord],
        states_by_id: Dict[str, ModelEndpointInfraState],
        run_at: datetime,
        report: EndpointGcReport,
    ) -> _Traffic:
        since = run_at - TRAFFIC_LOOKBACK
        by_deployment = {state.deployment_name: eid for eid, state in states_by_id.items()}
        by_name: Dict[str, List[str]] = {}
        for record in records:
            by_name.setdefault(record.name, []).append(record.id)

        def wants_history(record: ModelEndpointRecord) -> bool:
            metadata = record.metadata or {}
            if record.endpoint_type == ModelEndpointType.ASYNC or _is_exempt(metadata):
                return False
            state = states_by_id.get(record.id)
            if state is None or (state.deployment_state.available_workers or 0) == 0:
                return False
            observed_at = _parse_ts(metadata.get(GC_OBSERVED_AT_KEY))
            return (
                GC_LAST_TRAFFIC_AT_KEY not in metadata
                or observed_at is None
                or run_at - observed_at > TRAFFIC_LOOKBACK
            )

        needs_history = any(wants_history(record) for record in records)

        active: Set[str] = set()
        last_seen: Dict[str, datetime] = {}
        observed_pods: Dict[str, int] = {}
        coverage_known = False
        frozen = False
        for gateway in self.traffic_gateways:
            keys = await gateway.active_keys(since)
            if keys is None:
                frozen = True
                report.sources_unknown.append(type(gateway).__name__)
                continue
            active.update(self._resolve(keys, gateway.key, by_deployment, by_name))
            if gateway.reports_coverage:
                counts = await gateway.observed_pod_counts()
                if counts is None:
                    frozen = True
                    report.sources_unknown.append(f"{type(gateway).__name__} coverage")
                else:
                    coverage_known = True
                    self._merge_counts(observed_pods, counts, gateway.key, by_deployment, by_name)
            if not needs_history:
                continue
            history = await gateway.last_active_at(
                run_at - timedelta(days=self.config.idle_delete_days)
            )
            for key, seen in (history or {}).items():
                ids = self._resolve({key}, gateway.key, by_deployment, by_name)
                if len(ids) != 1:
                    # Endpoint names are unique per owner only: an ambiguous key is no history.
                    continue
                (eid,) = ids
                if eid not in last_seen or seen > last_seen[eid]:
                    last_seen[eid] = seen

        queue_active: Set[str] = set()
        queue_unknown: Set[str] = set()
        semaphore = asyncio.Semaphore(QUEUE_LOOKUP_CONCURRENCY)

        async def lookup(endpoint_id: str) -> Tuple[str, Optional[int]]:
            async with semaphore:
                return endpoint_id, await self.queue_delegate.messages_sent_since(
                    endpoint_id, since
                )

        async_ids = [
            record.id
            for record in records
            if record.endpoint_type == ModelEndpointType.ASYNC
            and record.id in states_by_id
            and not _is_exempt(record.metadata or {})
        ]
        for endpoint_id, sent in await asyncio.gather(*(lookup(eid) for eid in async_ids)):
            if sent is None:
                queue_unknown.add(endpoint_id)
            elif sent > 0:
                queue_active.add(endpoint_id)
                active.add(endpoint_id)
        if async_ids and len(queue_unknown) == len(async_ids):
            # Not one queue could be read: the source is down, not the endpoints.
            frozen = True
            report.sources_unknown.append("queue")
        uncovered: Set[str] = set()
        if coverage_known:
            # A serving http endpoint with a pod no source scrapes: its silence is not evidence.
            # (Idle http series expire, so absence of a series alone says nothing.) The pod
            # count comes from the Deployment, not from the source's own discovery.
            uncovered = {
                record.id
                for record in records
                if record.endpoint_type != ModelEndpointType.ASYNC
                and record.id in states_by_id
                and record.id not in active
                and not self._covered(states_by_id[record.id], observed_pods.get(record.id, 0))
            }
        return _Traffic(
            active=active,
            queue_active=queue_active,
            queue_unknown=queue_unknown,
            uncovered=uncovered,
            last_seen=last_seen,
            frozen=frozen,
        )

    @staticmethod
    def _covered(infra_state: ModelEndpointInfraState, observed_pods: int) -> bool:
        """Every serving pod of the Deployment is observed by a coverage-reporting source."""
        return observed_pods >= (infra_state.deployment_state.available_workers or 0)

    def _merge_counts(
        self,
        into: Dict[str, int],
        counts: Dict[str, int],
        kind: TrafficKey,
        by_deployment: Dict[str, str],
        by_name: Dict[str, List[str]],
    ) -> None:
        for key, count in counts.items():
            for endpoint_id in self._resolve({key}, kind, by_deployment, by_name):
                into[endpoint_id] = max(into.get(endpoint_id, 0), count)

    @staticmethod
    def _resolve(
        keys: Set[str],
        kind: TrafficKey,
        by_deployment: Dict[str, str],
        by_name: Dict[str, List[str]],
    ) -> Set[str]:
        ids: Set[str] = set()
        for key in keys:
            if kind == TrafficKey.DEPLOYMENT_NAME:
                if key in by_deployment:
                    ids.add(by_deployment[key])
            else:
                ids.update(by_name.get(key, []))
        return ids

    # ---- planning and actions ------------------------------------------------------------

    @staticmethod
    def _plan(
        record: ModelEndpointRecord,
        infra_state: ModelEndpointInfraState,
        reason: str,
        clock_start: datetime,
        *,
        parked: bool,
        scale_days: int,
        delete_days: int,
        run_at: datetime,
        due: List[PlannedAction],
        report: EndpointGcReport,
        requested_at: Optional[datetime] = None,
    ) -> None:
        # ``parked`` also covers a scale-to-zero that was requested but failed in the endpoint
        # builder: the next and last step is the delete, on the same schedule.
        if parked:
            due_at = clock_start + timedelta(days=delete_days)
            parked_at = _parse_ts((record.metadata or {}).get(GC_PARKED_AT_KEY)) or requested_at
            if parked_at is not None:
                # Whatever delayed the scale-to-zero, the endpoint stays parked for the full
                # gap the schedule promises before it is deleted, notices included.
                due_at = max(due_at, parked_at + timedelta(days=delete_days - scale_days))
            action = PlannedAction(record, infra_state, DELETE, reason, due_at)
        else:
            action = PlannedAction(
                record,
                infra_state,
                SCALE_TO_ZERO,
                reason,
                clock_start + timedelta(days=scale_days),
            )
        if action.due_at <= run_at:
            due.append(action)
            return
        # Notices compare on calendar days: the stamp and the daily run both sit at the same
        # hour, and seconds of scheduler jitter must not skip one.
        days_left = max(1, (action.due_at.date() - run_at.date()).days)
        if days_left in report.upcoming:
            report.upcoming[days_left].append(action)

    async def _act(
        self,
        due: List[PlannedAction],
        traffic: _Traffic,
        run_at: datetime,
        report: EndpointGcReport,
    ) -> None:
        due.sort(key=lambda action: action.due_at)
        self._infra_for_writes = None
        if not due:
            return
        if traffic.frozen or not self.config.actions_enabled:
            report.deferred.extend(due)
            return
        attempted = 0
        for action in due:
            if attempted >= self.config.action_cap:
                report.deferred.append(action)
                continue
            if action.kind == SCALE_TO_ZERO and not self._scale_to_zero_supported(action):
                report.unsupported.append(action)
                continue
            attempt = _Attempt()
            try:
                await self._act_one(action, run_at, report, attempt)
            except Exception:
                if attempt.made:
                    # The action itself was already reported; only the cleanup after it failed.
                    logger.exception(f"GC cleanup after acting on {action.record.id} failed")
                else:
                    # A failed pre-action read must not take the rest of the run (or the
                    # digest) down with it.
                    logger.exception(f"GC pre-action checks failed for {action.record.id}")
                    report.check_failed.append(action)
            attempted += attempt.made

    async def _act_one(
        self,
        action: PlannedAction,
        run_at: datetime,
        report: EndpointGcReport,
        attempt: "_Attempt",
    ) -> None:
        """Re-check one due action against live state and perform it."""
        # The judgement used the run's initial listing; re-check the record before acting,
        # under the endpoint's advisory lock so well-behaved writers wait.
        async with self.record_repository.get_lock_context(action.record) as lock:
            if not lock.lock_acquired():
                report.deferred.append(action)
                return
            fresh = await self.record_repository.get_model_endpoint_record(
                model_endpoint_id=action.record.id, refresh=True
            )
            if (
                fresh is None
                or fresh.status in IN_FLIGHT_STATUSES
                or _is_exempt(fresh.metadata or {})
                or self._owner_touched(fresh, fresh.metadata or {})
            ):
                report.skipped_at_action.append(action)
                return
            live = await self._live_state_allows(action, fresh, run_at, report)
            if live is None:
                return
            # Traffic was collected before bookkeeping; anything since must count. Asked right
            # before this endpoint's action, not once for the batch.
            activity = await self._activity_now(action, fresh, live, run_at, report)
            if activity.active is None:
                report.check_failed.append(action)
                return
            if activity.active:
                metadata = fresh.metadata or {}
                await self._write_gc_state(
                    fresh,
                    {
                        GC_LAST_TRAFFIC_AT_KEY: run_at.isoformat(),
                        GC_OBSERVED_AT_KEY: run_at.isoformat(),
                        **{
                            k: v
                            for k, v in metadata.items()
                            if k
                            in (
                                GC_SCALE_TO_ZERO_REQUESTED_AT_KEY,
                                GC_SCALE_TO_ZERO_TASK_ID_KEY,
                                GC_PARKED_AT_KEY,
                            )
                        },
                    },
                    run_at,
                    report,
                    locked=True,
                    infra_state=live,
                )
                report.recovered.append(fresh)
                report.skipped_at_action.append(action)
                return
            # The telemetry calls took time; restarts and scale-ups do not go through the
            # record lock, so look at the Deployment once more right before touching it.
            live = await self._live_state_allows(action, fresh, run_at, report)
            if live is None:
                return
            if activity.observed_pods is not None and not self._covered(
                live, activity.observed_pods
            ):
                # A pod came up while the sources were being asked; its requests were never
                # observed.
                report.check_failed.append(action)
                return
            if fresh.endpoint_type == ModelEndpointType.ASYNC:
                # Scaling to zero stops workers mid-task; deleting removes the queue with
                # whatever is in it. Either way the queue must be empty right now.
                queued = await self._queued_messages(fresh.id)
                if queued is None:
                    report.check_failed.append(action)
                    return
                if queued > 0:
                    report.skipped_at_action.append(action)
                    return
            attempt.made = True  # counts against the cap whatever happens from here
            try:
                if action.kind == DELETE:
                    await self.model_endpoint_service.delete_model_endpoint(fresh.id)
                    report.deleted.append(action)
                else:
                    await self._scale_to_zero(action, fresh, run_at, report)
            except Exception:
                logger.exception(f"GC {action.kind} failed for {fresh.id} ({fresh.name})")
                report.action_failed.append(action)

    async def _live_state_allows(
        self,
        action: PlannedAction,
        fresh: ModelEndpointRecord,
        run_at: datetime,
        report: EndpointGcReport,
    ) -> Optional[ModelEndpointInfraState]:
        """Read the Deployment now; None (after reporting) when it says not to act."""
        metadata = fresh.metadata or {}
        live = await self.resource_gateway.get_resources(
            endpoint_id=fresh.id,
            deployment_name=action.infra_state.deployment_name,
            endpoint_type=fresh.endpoint_type,
        )
        if self._owner_restarted(live, metadata):
            report.skipped_at_action.append(action)
            return None
        live_desired, live_available = _worker_counts(live)
        gc_parked = GC_SCALE_TO_ZERO_REQUESTED_AT_KEY in metadata
        if (
            live.deployment_state.min_workers > 0
            and gc_parked
            and fresh.status != ModelEndpointStatus.UPDATE_FAILED
        ):
            # The owner raised min_workers since the listing: revived, not ours.
            report.skipped_at_action.append(action)
            return None
        if action.reason == BROKEN and live_available > 0:
            # Recovered since the listing: drop the broken clock, take no action.
            await self._write_gc_state(
                fresh,
                {
                    k: v
                    for k, v in metadata.items()
                    if k in GC_STATE_KEYS
                    and k not in GC_BOOKKEEPING_KEYS
                    and k != GC_UNAVAILABLE_SINCE_KEY
                },
                run_at,
                report,
                locked=True,
                infra_state=live,
            )
            report.recovered.append(fresh)
            report.skipped_at_action.append(action)
            return None
        if action.kind == DELETE:
            if live_desired > 0 and fresh.status != ModelEndpointStatus.UPDATE_FAILED:
                # Something woke or scaled the parked endpoint since the listing. (A failed
                # scale-to-zero leaves the Deployment up by definition and still ends in
                # the delete.)
                report.skipped_at_action.append(action)
                return None
            if live_desired == 0 and gc_parked and GC_PARKED_AT_KEY not in metadata:
                # First sighting at zero after our request, during the final read rather
                # than the listing: the parked period the schedule promises starts now, so
                # the delete is not due today.
                await self._write_gc_state(
                    fresh,
                    {
                        **{
                            k: v
                            for k, v in metadata.items()
                            if k in GC_STATE_KEYS and k not in GC_BOOKKEEPING_KEYS
                        },
                        GC_PARKED_AT_KEY: run_at.isoformat(),
                    },
                    run_at,
                    report,
                    locked=True,
                    infra_state=live,
                )
                report.skipped_at_action.append(action)
                return None
        if action.kind == SCALE_TO_ZERO and live_desired == 0:
            report.skipped_at_action.append(action)
            return None
        return live

    async def _activity_now(
        self,
        action: PlannedAction,
        fresh: ModelEndpointRecord,
        live: ModelEndpointInfraState,
        run_at: datetime,
        report: EndpointGcReport,
    ) -> "_Activity":
        """Whether the endpoint was used in the lookback, asked of every source right now.

        HTTP evidence only revives endpoints GC parked or judged idle; a dead sync endpoint
        that GC has not parked is a caller's problem (design decision). Queue messages revive
        async endpoints in every state. None when any applicable source could not answer or
        does not cover the endpoint's pods: unknown is not silence.
        """
        since = run_at - TRAFFIC_LOOKBACK
        is_async = fresh.endpoint_type == ModelEndpointType.ASYNC
        unknown = False
        if is_async:
            sent = await self.queue_delegate.messages_sent_since(fresh.id, since)
            if sent is None:
                unknown = True
            elif sent > 0:
                return _Activity(active=True)
        if action.reason == IDLE or GC_SCALE_TO_ZERO_REQUESTED_AT_KEY in (fresh.metadata or {}):
            by_deployment = {action.infra_state.deployment_name: fresh.id}
            by_name = {fresh.name: [fresh.id]}
            # Coverage is asked for whenever pods could be serving: the count is checked again
            # against the Deployment read that follows the telemetry calls.
            needs_coverage = not is_async
            coverage_known = False
            observed_pods: Dict[str, int] = {}
            for gateway in self.traffic_gateways:
                keys = await gateway.active_keys(since)
                if keys is None:
                    report.sources_unknown.append(type(gateway).__name__)
                    unknown = True
                    continue
                if fresh.id in self._resolve(keys, gateway.key, by_deployment, by_name):
                    return _Activity(active=True)
                if needs_coverage and gateway.reports_coverage:
                    counts = await gateway.observed_pod_counts()
                    if counts is None:
                        unknown = True
                        continue
                    coverage_known = True
                    self._merge_counts(observed_pods, counts, gateway.key, by_deployment, by_name)
            if needs_coverage and coverage_known:
                observed = observed_pods.get(fresh.id, 0)
                if not self._covered(live, observed):
                    unknown = True
                return _Activity(active=None if unknown else False, observed_pods=observed)
        return _Activity(active=None if unknown else False)

    def _scale_to_zero_supported(self, action: PlannedAction) -> bool:
        if action.record.endpoint_type == ModelEndpointType.ASYNC:
            return True  # the celery autoscaler wakes async endpoints on queue depth
        if not self.config.http_scale_to_zero_supported:
            return False
        # Multinode (LeaderWorkerSet) endpoints get neither an HPA nor a KEDA ScaledObject, so
        # nothing would ever wake them again.
        return (action.infra_state.resource_state.nodes_per_worker or 1) <= 1

    async def _queued_messages(self, endpoint_id: str) -> Optional[int]:
        """Visible plus in-flight plus delayed messages, or None when any count is unreadable."""
        try:
            attributes = (await self.queue_delegate.get_queue_attributes(endpoint_id=endpoint_id))[
                "Attributes"
            ]
            return sum(
                int(attributes[name])
                for name in (
                    "ApproximateNumberOfMessages",
                    "ApproximateNumberOfMessagesNotVisible",
                    "ApproximateNumberOfMessagesDelayed",
                )
            )
        except Exception:
            logger.exception(f"could not read queue depth for {endpoint_id}")
            return None

    async def _scale_to_zero(
        self,
        action: PlannedAction,
        fresh: ModelEndpointRecord,
        run_at: datetime,
        report: EndpointGcReport,
    ) -> None:
        # Stamp first so the builder's record writes are attributed to GC, then record the
        # builder task id the update returns; owner updates get a different task id.
        base = {
            key: value
            for key, value in (action.record.metadata or {}).items()
            if key in GC_STATE_KEYS and key not in GC_BOOKKEEPING_KEYS
        }
        requested = {**base, GC_SCALE_TO_ZERO_REQUESTED_AT_KEY: run_at.isoformat()}
        if not await self._write_gc_state(
            action.record, requested, run_at, report, locked=True, infra_state=action.infra_state
        ):
            report.deferred.append(action)
            return
        try:
            # Passing the current bundle id keeps this a resource patch; without it the service
            # marks the bundle as changed and the delegate replaces the Deployment with 0 replicas.
            # The bundle id comes from the record re-read under the lock, never the listing.
            updated = await self.model_endpoint_service.update_model_endpoint(
                model_endpoint_id=action.record.id,
                model_bundle_id=fresh.current_model_bundle.id,
                min_workers=0,
            )
        except Exception:
            # The build may or may not have been enqueued. Keep the intent without a task id;
            # the next run adopts the builder's task id if one appears, or drops the intent.
            logger.exception(f"scale-to-zero update failed for {action.record.id}; intent kept")
            raise
        task_id = updated.creation_task_id or ""
        for attempt in range(TASK_ID_WRITE_RETRIES):
            # We still hold the endpoint lock from _act; the task id must land, or the
            # builder's own writes will read as an owner edit next run.
            try:
                if await self._write_gc_state(
                    action.record,
                    {**requested, GC_SCALE_TO_ZERO_TASK_ID_KEY: task_id},
                    run_at,
                    report,
                    locked=True,
                ):
                    report.scaled_to_zero.append(action)
                    return
            except Exception:
                logger.exception(
                    f"task id write attempt {attempt + 1} failed for {action.record.id}"
                )
            await asyncio.sleep(2)
        # The intent stays without a task id and is reconciled next run.
        raise RuntimeError(f"could not record the scale-to-zero task id for {action.record.id}")

    # ---- state -----------------------------------------------------------------------------

    @staticmethod
    def _owner_restarted(infra_state: ModelEndpointInfraState, metadata: Dict) -> bool:
        restarted_at = infra_state.restarted_at
        if restarted_at is None:
            return False
        if restarted_at.tzinfo is None:
            restarted_at = restarted_at.replace(tzinfo=timezone.utc)
        if GC_SEEN_RESTART_AT_KEY in metadata:
            # GC acknowledged what it saw before: a timestamp, or "" for no annotation. Anything
            # newer than that is an owner restart.
            seen = _parse_ts(metadata.get(GC_SEEN_RESTART_AT_KEY))
            return seen is None or restarted_at != seen
        touched = _parse_ts(metadata.get(GC_TOUCHED_AT_KEY))
        return touched is not None and restarted_at > touched

    @staticmethod
    def _owner_touched(record: ModelEndpointRecord, metadata: Dict) -> bool:
        seen = metadata.get(GC_SEEN_TASK_ID_KEY)
        if seen is not None:
            if (
                GC_SCALE_TO_ZERO_REQUESTED_AT_KEY in metadata
                and GC_SCALE_TO_ZERO_TASK_ID_KEY not in metadata
            ):
                # GC's own request whose task id was never recorded; reconciled in _judge.
                return False
            # Any API update enqueues a build and replaces creation_task_id.
            return (record.creation_task_id or "") != seen
        updated = record.last_updated_at
        touched = _parse_ts(metadata.get(GC_TOUCHED_AT_KEY))
        if updated is None or touched is None:
            return False
        if updated.tzinfo is None:
            updated = updated.replace(tzinfo=timezone.utc)
        return updated - touched > OWNER_UPDATE_SLACK

    async def _write_gc_state(
        self,
        record: ModelEndpointRecord,
        gc_state: Dict[str, str],
        run_at: datetime,
        report: EndpointGcReport,
        *,
        locked: bool = False,
        resetting: bool = False,
        infra_state: Optional[ModelEndpointInfraState] = None,
    ) -> bool:
        """Replace the GC keys in the endpoint's metadata, leaving every other key as stored.

        Returns True only when the planned state was written. False means the endpoint was
        locked elsewhere (nothing written) or an owner edit landed since this run listed it (a
        fresh idle clock was written instead); callers must not plan on the state they intended.
        """
        # Endpoint updates replace the whole JSONB, so take the same per-endpoint advisory lock
        # they take (unless the caller already holds it) and re-read before writing.
        async with AsyncExitStack() as stack:
            if not locked:
                lock = await stack.enter_async_context(
                    self.record_repository.get_lock_context(record)
                )
                if not lock.lock_acquired():
                    logger.warning(f"GC skipped metadata write for {record.id}: endpoint locked")
                    report.write_skipped.append(record)
                    return False
            fresh = await self.record_repository.get_model_endpoint_record(
                model_endpoint_id=record.id, refresh=True
            )
            current = fresh or record
            current_metadata = current.metadata or {}
            pending_request = (
                GC_SCALE_TO_ZERO_REQUESTED_AT_KEY in current_metadata
                and GC_SCALE_TO_ZERO_TASK_ID_KEY not in current_metadata
                and GC_SEEN_TASK_ID_KEY in current_metadata
                and (current.creation_task_id or "") != current_metadata[GC_SEEN_TASK_ID_KEY]
            )
            edited_since_listing = (current.creation_task_id or "") != (
                record.creation_task_id or ""
            )
            if (
                gc_state
                and not resetting
                and GC_SCALE_TO_ZERO_TASK_ID_KEY not in gc_state
                and (
                    edited_since_listing
                    or (
                        any(key in current_metadata for key in GC_STATE_KEYS)
                        and (self._owner_touched(current, current_metadata) or pending_request)
                    )
                )
            ):
                # The owner edited the endpoint between the listing and this write: drop the
                # planned state and restart from the edit instead.
                logger.info(f"GC found an owner edit on {record.id} before writing; resetting")
                gc_state = {
                    GC_LAST_TRAFFIC_AT_KEY: run_at.isoformat(),
                    GC_OBSERVED_AT_KEY: run_at.isoformat(),
                }
                report.owner_reset.append(record)
                reset = True
            else:
                reset = False
            merged = {
                key: value for key, value in current_metadata.items() if key not in GC_STATE_KEYS
            }
            merged.update({k: v for k, v in gc_state.items() if k not in GC_BOOKKEEPING_KEYS})
            if gc_state:
                # Write time, not run start: a long run must not read its own write as an edit.
                merged[GC_TOUCHED_AT_KEY] = self.now().isoformat()
                merged[GC_SEEN_TASK_ID_KEY] = (
                    gc_state.get(GC_SCALE_TO_ZERO_TASK_ID_KEY) or current.creation_task_id or ""
                )
                if infra_state is None:
                    infra_state = self._infra_for_writes
                if infra_state is not None:
                    # Acknowledge exactly what was seen: a timestamp, or "" for no annotation, so
                    # a restart that appears later is detected whatever GC's write times are.
                    restarted_at = infra_state.restarted_at
                    if restarted_at is not None and restarted_at.tzinfo is None:
                        restarted_at = restarted_at.replace(tzinfo=timezone.utc)
                    merged[GC_SEEN_RESTART_AT_KEY] = (
                        restarted_at.isoformat() if restarted_at is not None else ""
                    )
                elif GC_SEEN_RESTART_AT_KEY in current_metadata:
                    merged[GC_SEEN_RESTART_AT_KEY] = current_metadata[GC_SEEN_RESTART_AT_KEY]
            # Bookkeeping, not an owner edit: last_updated_at stays as the owner left it.
            await self.record_repository.update_model_endpoint_metadata(
                model_endpoint_id=record.id, metadata=merged
            )
            record.metadata = merged  # keep the in-memory record coherent for later steps
            return not reset
        return False  # unreachable: the exit stack does not swallow exceptions


def format_digest(
    report: EndpointGcReport,
    config: EndpointGcConfig,
    run_at: datetime,
    states_by_id: Dict[str, ModelEndpointInfraState],
) -> str:
    mode = "ACTIONS ENABLED" if config.actions_enabled else "OBSERVE ONLY (no actions)"
    lines = [
        f"model-engine endpoint GC {run_at.strftime('%Y-%m-%d %H:%M UTC')} [{mode}] "
        f"broken: zero at {config.broken_scale_to_zero_days}d, delete at "
        f"{config.broken_delete_days}d; idle: zero at {config.idle_scale_to_zero_days}d, "
        f"delete at {config.idle_delete_days}d; cap {config.action_cap}/run",
        f"scaled to zero {len(report.scaled_to_zero)}, deleted {len(report.deleted)}, "
        f"failed {len(report.action_failed)}, checks failed {len(report.check_failed)}, "
        f"deferred {len(report.deferred)}, "
        f"tracking {len(report.tracking)}, recovered {len(report.recovered)}, "
        f"owner reset {len(report.owner_reset)}, no deployment {len(report.no_deployment)}, "
        f"in flight {len(report.in_flight)}, stuck {len(report.stuck)}, exempt {len(report.exempt)}, "
        f"queue unknown {len(report.queue_unknown)}, traffic unknown {len(report.traffic_unknown)}, "
        f"unsupported {len(report.unsupported)}, "
        f"skipped at action {len(report.skipped_at_action)}, "
        f"state invalid {len(report.state_invalid)}, write skipped {len(report.write_skipped)}, "
        f"bookkeeping failed {len(report.judge_failed)}",
    ]
    if report.sources_unknown:
        lines.append(
            "TRAFFIC SOURCE UNAVAILABLE, clocks frozen this run: "
            + ", ".join(sorted(set(report.sources_unknown)))
        )

    def describe(record: ModelEndpointRecord) -> str:
        labels = states_by_id[record.id].labels if record.id in states_by_id else {}
        return (
            f"{record.name} ({record.id}) team={labels.get('team', '?')} "
            f"product={labels.get('product', '?')} owner={record.owner} "
            f"created_by={record.created_by}"
        )

    action_sections: List[Tuple[str, List[PlannedAction]]] = [
        (
            "Scaled to zero today (min_workers set to 0; an owner update restores it)",
            report.scaled_to_zero,
        ),
        ("Deleted today", report.deleted),
        ("Action failed", report.action_failed),
        ("Due, not acted: a live check failed", report.check_failed),
    ]
    for days in NOTICE_DAYS:
        action_sections.append((f"In {days} day{'s' if days != 1 else ''}", report.upcoming[days]))
    action_sections += [
        ("Due, deferred (cap, actions disabled, or a source was unavailable)", report.deferred),
        ("Due, unsupported (cluster cannot scale http endpoints to zero)", report.unsupported),
        (
            "Due, skipped: endpoint changed since it was judged, or its queue is not empty",
            report.skipped_at_action,
        ),
    ]
    for title, actions in action_sections:
        if actions:
            lines.append(f"\n{title} ({len(actions)}):")
            lines.extend(
                f"  - {action.kind.replace('_', ' ')} [{action.reason}] "
                f"{action.due_at.strftime('%Y-%m-%d')}: {describe(action.record)}"
                for action in actions
            )
    record_sections = [
        ("Owner edit detected, GC state cleared", report.owner_reset),
        ("Recovered, clock dropped", report.recovered),
        ("No Deployment in the listing, state kept", report.no_deployment),
        ("Update or delete in flight, skipped", report.in_flight),
        ("GC scale-to-zero stuck in flight for days (builder needs a look)", report.stuck),
        ("Queue activity unknown, skipped", report.queue_unknown),
        ("Pods not scraped by any traffic source, skipped", report.traffic_unknown),
        ("GC state unreadable, skipped (fix the metadata)", report.state_invalid),
        ("Metadata write skipped, endpoint locked", report.write_skipped),
        ("Bookkeeping failed, skipped this run (see the job log)", report.judge_failed),
    ]
    for title, records in record_sections:
        if records:
            lines.append(f"\n{title} ({len(records)}):")
            lines.extend(f"  - {describe(record)}" for record in records)
    return "\n".join(lines)
