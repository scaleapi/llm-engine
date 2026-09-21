"""Garbage collection for model endpoints whose workers have all been unavailable for a long time.

State lives in the endpoint's ``endpoint_metadata`` JSONB so that it is deleted with the row,
visible to the owner through the API, and seedable with a single UPDATE:

- ``_gc_unavailable_since``: first run on which GC itself saw desired > 0 and available == 0.
- ``_gc_flagged_at``: run on which the unavailable window elapsed; the grace period starts here.
- ``_gc_exempt``: truthy value opts the endpoint out entirely.

GC only trusts its own observations. The clock never starts from a kubernetes condition
timestamp, and any run that sees the endpoint healthy or scaled to zero clears all GC keys. The
window is wall clock from the first observation, not a count of runs: a run that fails or is
skipped neither advances nor resets it.
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Callable, Dict, List, Optional, Tuple

from model_engine_server.common.constants import (
    ENDPOINT_GC_EXEMPT_KEY,
    ENDPOINT_GC_FLAGGED_AT_KEY,
    ENDPOINT_GC_UNAVAILABLE_SINCE_KEY,
)
from model_engine_server.core.loggers import logger_name, make_logger
from model_engine_server.domain.entities import (
    ModelEndpointInfraState,
    ModelEndpointRecord,
    ModelEndpointStatus,
    ModelEndpointType,
)
from model_engine_server.domain.gateways import DigestGateway
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
GC_FLAGGED_AT_KEY = ENDPOINT_GC_FLAGGED_AT_KEY
GC_EXEMPT_KEY = ENDPOINT_GC_EXEMPT_KEY
GC_KEYS = (GC_UNAVAILABLE_SINCE_KEY, GC_FLAGGED_AT_KEY)
# GC's own metadata writes bump last_updated_at moments after the flag stamp; owner edits land
# well after that.
OWNER_UPDATE_SLACK = timedelta(minutes=5)
IN_FLIGHT_STATUSES = {
    ModelEndpointStatus.UPDATE_PENDING,
    ModelEndpointStatus.UPDATE_IN_PROGRESS,
    ModelEndpointStatus.DELETE_IN_PROGRESS,
}


@dataclass(frozen=True)
class EndpointGcConfig:
    unavailable_days: int = 30
    grace_days: int = 14
    delete_cap: int = 20
    delete_enabled: bool = False  # False: bookkeeping and digest only, no deletions.


@dataclass
class EndpointGcReport:
    observing: List[ModelEndpointRecord] = field(default_factory=list)
    flagged_new: List[ModelEndpointRecord] = field(default_factory=list)
    in_grace: List[ModelEndpointRecord] = field(default_factory=list)
    deleted: List[ModelEndpointRecord] = field(default_factory=list)
    delete_failed: List[ModelEndpointRecord] = field(default_factory=list)
    delete_deferred: List[ModelEndpointRecord] = field(default_factory=list)
    cleared: List[ModelEndpointRecord] = field(default_factory=list)
    no_deployment: List[ModelEndpointRecord] = field(default_factory=list)
    in_flight: List[ModelEndpointRecord] = field(default_factory=list)
    exempt: List[ModelEndpointRecord] = field(default_factory=list)
    queue_unknown: List[ModelEndpointRecord] = field(default_factory=list)
    state_invalid: List[ModelEndpointRecord] = field(default_factory=list)
    write_skipped: List[ModelEndpointRecord] = field(default_factory=list)


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


def _all_workers_unavailable(infra_state: ModelEndpointInfraState) -> bool:
    # ModelEndpointDeploymentState carries no desired count; available + unavailable is the
    # Deployment's status.replicas, which is 0 only when the endpoint is scaled to zero.
    state = infra_state.deployment_state
    return (state.available_workers or 0) == 0 and (state.unavailable_workers or 0) > 0


class EndpointGarbageCollectionService:
    def __init__(
        self,
        model_endpoint_record_repository: ModelEndpointRecordRepository,
        resource_gateway: EndpointResourceGateway,
        queue_delegate: QueueEndpointResourceDelegate,
        model_endpoint_service: ModelEndpointService,
        digest_gateway: DigestGateway,
        config: EndpointGcConfig,
        now: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
    ):
        self.record_repository = model_endpoint_record_repository
        self.resource_gateway = resource_gateway
        self.queue_delegate = queue_delegate
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

        ready: List[Tuple[datetime, ModelEndpointRecord]] = []
        for record in records:
            metadata = record.metadata or {}
            has_gc_state = any(key in metadata for key in GC_KEYS)
            if record.status in IN_FLIGHT_STATUSES:
                # A build or delete is running; judge the outcome on a later run.
                if has_gc_state:
                    report.in_flight.append(record)
                continue
            if _is_exempt(metadata):
                report.exempt.append(record)
                continue

            infra_state = states_by_id.get(record.id)
            if infra_state is None:
                if has_gc_state:
                    await self._write_gc_state(record, {}, report)
                    report.no_deployment.append(record)
                continue
            qualifies = _all_workers_unavailable(infra_state)
            if qualifies and record.endpoint_type == ModelEndpointType.ASYNC:
                # A dead async endpoint still accepts messages into its queue; only treat it as
                # unused when nothing was enqueued for the whole window.
                sent = await self.queue_delegate.messages_sent_since(
                    record.id, run_at - timedelta(days=self.config.unavailable_days)
                )
                if sent is None:
                    # Unknown activity: neither advance nor reset the clock this run.
                    report.queue_unknown.append(record)
                    continue
                qualifies = sent == 0
            if not qualifies:
                if has_gc_state:
                    await self._write_gc_state(record, {}, report)
                    report.cleared.append(record)
                continue

            if has_gc_state and not self._state_parses(metadata):
                # Never overwrite a stamp that is present but unreadable; an operator must fix it.
                logger.warning(f"GC state on {record.id} is not ISO-8601: {metadata}")
                report.state_invalid.append(record)
                continue

            since = _parse_ts(metadata.get(GC_UNAVAILABLE_SINCE_KEY))
            if since is None:
                await self._write_gc_state(
                    record, {GC_UNAVAILABLE_SINCE_KEY: run_at.isoformat()}, report
                )
                report.observing.append(record)
                continue
            if run_at - since < timedelta(days=self.config.unavailable_days):
                report.observing.append(record)
                continue

            flagged_at = _parse_ts(metadata.get(GC_FLAGGED_AT_KEY))
            if flagged_at is None:
                await self._write_gc_state(
                    record,
                    {
                        GC_UNAVAILABLE_SINCE_KEY: since.isoformat(),
                        GC_FLAGGED_AT_KEY: run_at.isoformat(),
                    },
                    report,
                )
                report.flagged_new.append(record)
                continue
            if self._owner_updated_since(record, flagged_at):
                # The owner touched the endpoint during grace: restart the grace period.
                await self._write_gc_state(
                    record,
                    {
                        GC_UNAVAILABLE_SINCE_KEY: since.isoformat(),
                        GC_FLAGGED_AT_KEY: run_at.isoformat(),
                    },
                    report,
                )
                report.flagged_new.append(record)
                continue
            if run_at - flagged_at < timedelta(days=self.config.grace_days):
                report.in_grace.append(record)
                continue
            ready.append((flagged_at, record))

        ready.sort(key=lambda item: item[0])
        for index, (_, record) in enumerate(ready):
            if index >= self.config.delete_cap or not self.config.delete_enabled:
                report.delete_deferred.append(record)
                continue
            try:
                await self.model_endpoint_service.delete_model_endpoint(record.id)
                report.deleted.append(record)
            except Exception:
                logger.exception(f"GC failed to delete endpoint {record.id} ({record.name})")
                report.delete_failed.append(record)

        try:
            self.digest_gateway.send_digest(
                format_digest(report, self.config, run_at, states_by_id)
            )
        except Exception:
            logger.exception("GC digest delivery failed")
        return report

    @staticmethod
    def _state_parses(metadata: Dict) -> bool:
        return all(_parse_ts(metadata[key]) is not None for key in GC_KEYS if key in metadata)

    @staticmethod
    def _owner_updated_since(record: ModelEndpointRecord, flagged_at: datetime) -> bool:
        updated = record.last_updated_at
        if updated is None:
            return False
        if updated.tzinfo is None:
            updated = updated.replace(tzinfo=timezone.utc)
        return updated - flagged_at > OWNER_UPDATE_SLACK

    async def _write_gc_state(
        self, record: ModelEndpointRecord, gc_state: Dict[str, str], report: EndpointGcReport
    ) -> None:
        """Replace the GC keys in the endpoint's metadata, leaving every other key as stored."""
        # Endpoint updates replace the whole JSONB, so take the same per-endpoint advisory lock
        # they take and re-read before writing; a concurrent user update is then kept.
        async with self.record_repository.get_lock_context(record) as lock:
            if not lock.lock_acquired():
                logger.warning(f"GC skipped metadata write for {record.id}: endpoint locked")
                report.write_skipped.append(record)
                return
            fresh = await self.record_repository.get_model_endpoint_record(
                model_endpoint_id=record.id
            )
            merged = {
                key: value
                for key, value in ((fresh or record).metadata or {}).items()
                if key not in GC_KEYS
            }
            merged.update(gc_state)
            await self.record_repository.update_model_endpoint_record(
                model_endpoint_id=record.id, metadata=merged
            )


def format_digest(
    report: EndpointGcReport,
    config: EndpointGcConfig,
    run_at: datetime,
    states_by_id: Dict[str, ModelEndpointInfraState],
) -> str:
    mode = "DELETE ENABLED" if config.delete_enabled else "OBSERVE ONLY (no deletions)"
    counts = {
        "deleted": len(report.deleted),
        "failed": len(report.delete_failed),
        "deferred": len(report.delete_deferred),
        "newly flagged": len(report.flagged_new),
        "in grace": len(report.in_grace),
        "observing": len(report.observing),
        "cleared": len(report.cleared),
        "no deployment": len(report.no_deployment),
        "in flight": len(report.in_flight),
        "exempt": len(report.exempt),
        "queue unknown": len(report.queue_unknown),
        "state invalid": len(report.state_invalid),
        "write skipped": len(report.write_skipped),
    }
    lines = [
        f"model-engine endpoint GC {run_at.strftime('%Y-%m-%d %H:%M UTC')} [{mode}] "
        f"unavailable>{config.unavailable_days}d, grace {config.grace_days}d, "
        f"cap {config.delete_cap}/run",
        ", ".join(f"{name} {count}" for name, count in counts.items()),
    ]
    sections = [
        ("Deleted", report.deleted),
        ("Delete failed", report.delete_failed),
        ("Past grace, deferred (cap or deletes disabled)", report.delete_deferred),
        (f"Flagged, delete after {config.grace_days}d of grace", report.flagged_new),
        ("In grace", report.in_grace),
        ("Observing unavailable, clock running", report.observing),
        ("Recovered, GC state cleared", report.cleared),
        ("No Deployment found, GC state cleared (record left as is)", report.no_deployment),
        ("Update or delete in flight, skipped", report.in_flight),
        ("Queue activity unknown, skipped", report.queue_unknown),
        ("GC state unreadable, skipped (fix the metadata)", report.state_invalid),
        ("Metadata write skipped, endpoint locked", report.write_skipped),
    ]
    for title, records in sections:
        if records:
            lines.append(f"\n{title} ({len(records)}):")
            for record in records:
                labels = states_by_id[record.id].labels if record.id in states_by_id else {}
                line = (
                    f"  - {record.name} ({record.id}) team={labels.get('team', '?')} "
                    f"product={labels.get('product', '?')} owner={record.owner} "
                    f"created_by={record.created_by}"
                )
                flagged_at = _parse_ts((record.metadata or {}).get(GC_FLAGGED_AT_KEY))
                if records is report.flagged_new:
                    flagged_at = run_at  # stamped this run; the record still holds the old copy
                if flagged_at is not None:
                    delete_after = flagged_at + timedelta(days=config.grace_days)
                    line += f" delete_after={delete_after.strftime('%Y-%m-%d')}"
                lines.append(line)
    return "\n".join(lines)
