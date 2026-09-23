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
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Callable, Dict, List, Optional, Sequence, Set, Tuple

from model_engine_server.common.constants import (
    ENDPOINT_GC_EXEMPT_KEY,
    ENDPOINT_GC_LAST_TRAFFIC_AT_KEY,
    ENDPOINT_GC_SCALE_TO_ZERO_REQUESTED_AT_KEY,
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
GC_TOUCHED_AT_KEY = ENDPOINT_GC_TOUCHED_AT_KEY
GC_EXEMPT_KEY = ENDPOINT_GC_EXEMPT_KEY
GC_STATE_KEYS = (
    GC_UNAVAILABLE_SINCE_KEY,
    GC_LAST_TRAFFIC_AT_KEY,
    GC_SCALE_TO_ZERO_REQUESTED_AT_KEY,
    GC_TOUCHED_AT_KEY,
)

# GC's own metadata writes bump last_updated_at moments after _gc_touched_at; owner edits land
# well after that. A scale-to-zero request goes through the endpoint builder, whose record writes
# in the following hours are GC's doing too.
OWNER_UPDATE_SLACK = timedelta(minutes=5)
SCALE_TO_ZERO_ATTRIBUTION = timedelta(hours=24)
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
    upcoming: Dict[int, List[PlannedAction]] = field(
        default_factory=lambda: {days: [] for days in NOTICE_DAYS}
    )
    tracking: List[ModelEndpointRecord] = field(default_factory=list)  # a clock is running
    owner_reset: List[ModelEndpointRecord] = field(default_factory=list)
    recovered: List[ModelEndpointRecord] = field(default_factory=list)
    no_deployment: List[ModelEndpointRecord] = field(default_factory=list)
    in_flight: List[ModelEndpointRecord] = field(default_factory=list)
    exempt: List[ModelEndpointRecord] = field(default_factory=list)
    queue_unknown: List[ModelEndpointRecord] = field(default_factory=list)
    state_invalid: List[ModelEndpointRecord] = field(default_factory=list)
    write_skipped: List[ModelEndpointRecord] = field(default_factory=list)
    sources_unknown: List[str] = field(default_factory=list)


@dataclass
class _Traffic:
    active: Set[str]  # endpoint ids with a request in the lookback (empty when frozen)
    queue_active: Set[str]  # async endpoint ids with messages in the lookback
    queue_unknown: Set[str]  # async endpoint ids whose queue could not be read: skipped
    last_seen: Dict[str, datetime]  # from history-capable sources, for first sightings
    frozen: bool  # a cluster-wide source could not answer: no clock starts, no action runs


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
    # ModelEndpointDeploymentState carries no desired count; available + unavailable is the
    # Deployment's status.replicas, which is 0 only when the endpoint is scaled to zero.
    state = infra_state.deployment_state
    available = state.available_workers or 0
    return available + (state.unavailable_workers or 0), available


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
            await self._judge(record, states_by_id.get(record.id), traffic, run_at, due, report)
        await self._act(due, traffic, run_at, report)
        try:
            self.digest_gateway.send_digest(
                format_digest(report, self.config, run_at, states_by_id)
            )
        except Exception:
            logger.exception("GC digest delivery failed")
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
        metadata = record.metadata or {}
        has_state = any(key in metadata for key in GC_STATE_KEYS)
        if record.status in IN_FLIGHT_STATUSES:
            if has_state:
                report.in_flight.append(record)
            return
        if _is_exempt(metadata):
            report.exempt.append(record)
            return
        if has_state and not all(
            _parse_ts(metadata[key]) is not None for key in GC_STATE_KEYS if key in metadata
        ):
            logger.warning(f"GC state on {record.id} is not ISO-8601: {metadata}")
            report.state_invalid.append(record)
            return
        if has_state and self._owner_touched(record, metadata):
            await self._write_gc_state(record, {}, run_at, report)
            report.owner_reset.append(record)
            return
        if infra_state is None:
            if has_state:
                await self._write_gc_state(record, {}, run_at, report)
                report.no_deployment.append(record)
            return

        desired, available = _worker_counts(infra_state)
        unavailable_since = _parse_ts(metadata.get(GC_UNAVAILABLE_SINCE_KEY))
        last_traffic_at = _parse_ts(metadata.get(GC_LAST_TRAFFIC_AT_KEY))
        requested_at = _parse_ts(metadata.get(GC_SCALE_TO_ZERO_REQUESTED_AT_KEY))
        is_async = record.endpoint_type == ModelEndpointType.ASYNC
        active = record.id in traffic.active
        if is_async and record.id in traffic.queue_unknown:
            report.queue_unknown.append(record)
            return
        queue_active = is_async and record.id in traffic.queue_active

        if requested_at and infra_state.deployment_state.min_workers > 0:
            if record.status == ModelEndpointStatus.UPDATE_FAILED:
                # GC's scale-to-zero never took (the builder failed); the request stays and the
                # schedule ends in the delete.
                pass
            else:
                # min_workers is back above zero: the owner revived it. Judge fresh.
                await self._write_gc_state(record, {}, run_at, report)
                report.owner_reset.append(record)
                return
        # Parked by GC means min_workers is 0 on our request, whether or not KEDA has a pod up.
        gc_parked = requested_at is not None
        keep_request = (
            {GC_SCALE_TO_ZERO_REQUESTED_AT_KEY: requested_at.isoformat()} if requested_at else {}
        )
        if has_state and GC_TOUCHED_AT_KEY not in metadata:
            # Seeded state: stamp our own write time so owner edits become detectable.
            await self._write_gc_state(
                record, {k: v for k, v in metadata.items() if k in GC_STATE_KEYS}, run_at, report
            )

        if desired == 0 and not gc_parked:
            # Parked by the owner: never ours to judge.
            if has_state:
                await self._write_gc_state(record, {}, run_at, report)
                report.recovered.append(record)
            return

        broken = desired > 0 and available == 0
        if broken and queue_active:
            # Dead async endpoint with work still arriving: not ours to collect.
            if has_state and unavailable_since:
                await self._write_gc_state(
                    record,
                    {GC_LAST_TRAFFIC_AT_KEY: run_at.isoformat(), **keep_request},
                    run_at,
                    report,
                )
                report.recovered.append(record)
            return
        if broken or (gc_parked and desired == 0 and unavailable_since):
            # Broken, or parked by GC because it was broken. Traffic to a dead sync endpoint is a
            # caller's problem, not a sign of life, so only the async queue check above counts.
            if unavailable_since is None:
                if traffic.frozen:
                    return
                unavailable_since = run_at
                await self._write_gc_state(
                    record,
                    {GC_UNAVAILABLE_SINCE_KEY: run_at.isoformat(), **keep_request},
                    run_at,
                    report,
                )
            report.tracking.append(record)
            self._plan(
                record,
                BROKEN,
                unavailable_since,
                parked=gc_parked,
                scale_days=self.config.broken_scale_to_zero_days,
                delete_days=self.config.broken_delete_days,
                run_at=run_at,
                due=due,
                report=report,
            )
            return

        # Serving (available > 0) or parked by GC for idleness: the idle clock.
        if active and not traffic.frozen:
            if unavailable_since:
                report.recovered.append(record)
            if (
                unavailable_since
                or last_traffic_at is None
                or run_at - last_traffic_at > timedelta(hours=12)
            ):
                await self._write_gc_state(
                    record,
                    {GC_LAST_TRAFFIC_AT_KEY: run_at.isoformat(), **keep_request},
                    run_at,
                    report,
                )
            return
        if last_traffic_at is None:
            if traffic.frozen:
                return
            # First sight. History may push the clock back, but never so far that the first
            # action lands before the first notice.
            floor = run_at - timedelta(days=self.config.idle_scale_to_zero_days - NOTICE_DAYS[0])
            last_traffic_at = max(traffic.last_seen.get(record.id, run_at), floor)
            await self._write_gc_state(
                record,
                {GC_LAST_TRAFFIC_AT_KEY: last_traffic_at.isoformat(), **keep_request},
                run_at,
                report,
            )
        elif unavailable_since:
            # Was broken, now serving: keep only the idle clock.
            await self._write_gc_state(
                record,
                {GC_LAST_TRAFFIC_AT_KEY: last_traffic_at.isoformat(), **keep_request},
                run_at,
                report,
            )
            report.recovered.append(record)
        report.tracking.append(record)
        self._plan(
            record,
            IDLE,
            last_traffic_at,
            parked=gc_parked,
            scale_days=self.config.idle_scale_to_zero_days,
            delete_days=self.config.idle_delete_days,
            run_at=run_at,
            due=due,
            report=report,
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
        needs_history = any(
            GC_LAST_TRAFFIC_AT_KEY not in (record.metadata or {}) and record.id in states_by_id
            for record in records
        )

        active: Set[str] = set()
        last_seen: Dict[str, datetime] = {}
        frozen = False
        for gateway in self.traffic_gateways:
            keys = await gateway.active_keys(since)
            if keys is None:
                frozen = True
                report.sources_unknown.append(type(gateway).__name__)
                continue
            active.update(self._resolve(keys, gateway.key, by_deployment, by_name))
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
        for record in records:
            if record.endpoint_type != ModelEndpointType.ASYNC or record.id not in states_by_id:
                continue
            sent = await self.queue_delegate.messages_sent_since(record.id, since)
            if sent is None:
                queue_unknown.add(record.id)
            elif sent > 0:
                queue_active.add(record.id)
                active.add(record.id)
        return _Traffic(
            active=set() if frozen else active,
            queue_active=queue_active,
            queue_unknown=queue_unknown,
            last_seen=last_seen,
            frozen=frozen,
        )

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
        reason: str,
        clock_start: datetime,
        *,
        parked: bool,
        scale_days: int,
        delete_days: int,
        run_at: datetime,
        due: List[PlannedAction],
        report: EndpointGcReport,
    ) -> None:
        # ``parked`` also covers a scale-to-zero that was requested but failed in the endpoint
        # builder: the next and last step is the delete, on the same schedule.
        if parked:
            action = PlannedAction(
                record, DELETE, reason, clock_start + timedelta(days=delete_days)
            )
        else:
            action = PlannedAction(
                record, SCALE_TO_ZERO, reason, clock_start + timedelta(days=scale_days)
            )
        # Compare on calendar days: the stamp and the daily run both sit at the same hour, and
        # seconds of scheduler jitter must not skip a notice.
        days_left = (action.due_at.date() - run_at.date()).days
        if days_left <= 0:
            due.append(action)
        elif days_left in report.upcoming:
            report.upcoming[days_left].append(action)

    async def _act(
        self,
        due: List[PlannedAction],
        traffic: _Traffic,
        run_at: datetime,
        report: EndpointGcReport,
    ) -> None:
        due.sort(key=lambda action: action.due_at)
        for index, action in enumerate(due):
            if index >= self.config.action_cap or not self.config.actions_enabled or traffic.frozen:
                report.deferred.append(action)
                continue
            if (
                action.kind == SCALE_TO_ZERO
                and action.record.endpoint_type != ModelEndpointType.ASYNC
                and not self.config.http_scale_to_zero_supported
            ):
                report.unsupported.append(action)
                continue
            try:
                if action.kind == DELETE:
                    await self.model_endpoint_service.delete_model_endpoint(action.record.id)
                    report.deleted.append(action)
                else:
                    # Stamp first so the builder's record writes are attributed to GC.
                    state = {
                        key: value
                        for key, value in (action.record.metadata or {}).items()
                        if key in GC_STATE_KEYS and key != GC_TOUCHED_AT_KEY
                    }
                    state[GC_SCALE_TO_ZERO_REQUESTED_AT_KEY] = run_at.isoformat()
                    if not await self._write_gc_state(action.record, state, run_at, report):
                        report.deferred.append(action)
                        continue
                    await self.model_endpoint_service.update_model_endpoint(
                        model_endpoint_id=action.record.id, min_workers=0
                    )
                    report.scaled_to_zero.append(action)
            except Exception:
                logger.exception(
                    f"GC {action.kind} failed for {action.record.id} ({action.record.name})"
                )
                report.action_failed.append(action)

    # ---- state -----------------------------------------------------------------------------

    @staticmethod
    def _owner_touched(record: ModelEndpointRecord, metadata: Dict) -> bool:
        updated = record.last_updated_at
        touched = _parse_ts(metadata.get(GC_TOUCHED_AT_KEY))
        if updated is None or touched is None:
            return False
        if updated.tzinfo is None:
            updated = updated.replace(tzinfo=timezone.utc)
        requested = _parse_ts(metadata.get(GC_SCALE_TO_ZERO_REQUESTED_AT_KEY))
        if (
            requested
            and requested - OWNER_UPDATE_SLACK <= updated <= requested + SCALE_TO_ZERO_ATTRIBUTION
        ):
            # The endpoint builder writing the record after GC's own scale-to-zero request.
            return False
        return updated - touched > OWNER_UPDATE_SLACK

    async def _write_gc_state(
        self,
        record: ModelEndpointRecord,
        gc_state: Dict[str, str],
        run_at: datetime,
        report: EndpointGcReport,
    ) -> bool:
        """Replace the GC keys in the endpoint's metadata, leaving every other key as stored."""
        # Endpoint updates replace the whole JSONB, so take the same per-endpoint advisory lock
        # they take and re-read before writing; a concurrent user update is then kept.
        async with self.record_repository.get_lock_context(record) as lock:
            if not lock.lock_acquired():
                logger.warning(f"GC skipped metadata write for {record.id}: endpoint locked")
                report.write_skipped.append(record)
                return False
            fresh = await self.record_repository.get_model_endpoint_record(
                model_endpoint_id=record.id
            )
            merged = {
                key: value
                for key, value in ((fresh or record).metadata or {}).items()
                if key not in GC_STATE_KEYS
            }
            merged.update({k: v for k, v in gc_state.items() if k != GC_TOUCHED_AT_KEY})
            if gc_state:
                # Write time, not run start: a long run must not read its own write as an edit.
                merged[GC_TOUCHED_AT_KEY] = self.now().isoformat()
            await self.record_repository.update_model_endpoint_record(
                model_endpoint_id=record.id, metadata=merged
            )
            record.metadata = merged  # keep the in-memory record coherent for later steps
            return True


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
        f"failed {len(report.action_failed)}, deferred {len(report.deferred)}, "
        f"tracking {len(report.tracking)}, recovered {len(report.recovered)}, "
        f"owner reset {len(report.owner_reset)}, no deployment {len(report.no_deployment)}, "
        f"in flight {len(report.in_flight)}, exempt {len(report.exempt)}, "
        f"queue unknown {len(report.queue_unknown)}, unsupported {len(report.unsupported)}, "
        f"state invalid {len(report.state_invalid)}, write skipped {len(report.write_skipped)}",
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
        ("Scaled to zero today", report.scaled_to_zero),
        ("Deleted today", report.deleted),
        ("Action failed", report.action_failed),
        ("Due, deferred (cap, actions disabled, or a source was unavailable)", report.deferred),
        ("Due, unsupported (cluster cannot scale http endpoints to zero)", report.unsupported),
    ]
    for days in NOTICE_DAYS:
        action_sections.append((f"In {days} day{'s' if days != 1 else ''}", report.upcoming[days]))
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
        ("No Deployment found, GC state cleared (record left as is)", report.no_deployment),
        ("Update or delete in flight, skipped", report.in_flight),
        ("Queue activity unknown, skipped", report.queue_unknown),
        ("GC state unreadable, skipped (fix the metadata)", report.state_invalid),
        ("Metadata write skipped, endpoint locked", report.write_skipped),
    ]
    for title, records in record_sections:
        if records:
            lines.append(f"\n{title} ({len(records)}):")
            lines.extend(f"  - {describe(record)}" for record in records)
    return "\n".join(lines)
