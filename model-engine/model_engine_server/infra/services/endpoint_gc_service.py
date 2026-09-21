"""Garbage collection for model endpoints whose workers have all been unavailable for a long time.

State lives in the endpoint's ``endpoint_metadata`` JSONB so that it is deleted with the row,
visible to the owner through the API, and seedable with a single UPDATE:

- ``_gc_unavailable_since``: first run on which GC itself saw desired > 0 and available == 0.
- ``_gc_flagged_at``: run on which the unavailable window elapsed; the grace period starts here.
- ``_gc_exempt``: truthy value opts the endpoint out entirely.

GC only trusts its own observations. The clock never starts from a kubernetes condition
timestamp, and any run that sees the endpoint healthy or scaled to zero clears all GC keys.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Callable, Dict, List, Optional

from model_engine_server.core.loggers import logger_name, make_logger
from model_engine_server.domain.entities import (
    ModelEndpointInfraState,
    ModelEndpointRecord,
    ModelEndpointStatus,
    ModelEndpointType,
)
from model_engine_server.domain.services import ModelEndpointService
from model_engine_server.infra.gateways.resources.endpoint_resource_gateway import (
    EndpointResourceGateway,
)
from model_engine_server.infra.repositories.model_endpoint_record_repository import (
    ModelEndpointRecordRepository,
)

logger = make_logger(logger_name())

GC_UNAVAILABLE_SINCE_KEY = "_gc_unavailable_since"
GC_FLAGGED_AT_KEY = "_gc_flagged_at"
GC_EXEMPT_KEY = "_gc_exempt"
GC_KEYS = (GC_UNAVAILABLE_SINCE_KEY, GC_FLAGGED_AT_KEY)


class QueueActivityGateway(ABC):
    """Reports whether an async endpoint's queue received messages during a window."""

    @abstractmethod
    async def messages_sent_since(self, endpoint_id: str, since: datetime) -> Optional[int]:
        """Number of messages sent to the endpoint's queue since ``since``.

        Returns None when the answer is unknown; callers must treat that as "possibly active".
        """


class DigestGateway(ABC):
    @abstractmethod
    def send_digest(self, text: str) -> None:
        """Deliver a human-readable run summary."""


@dataclass(frozen=True)
class EndpointGcConfig:
    unavailable_days: int = 30
    grace_days: int = 14
    delete_cap: int = 20
    apply: bool = False  # False: bookkeeping and digest only, no deletions.


@dataclass
class EndpointGcReport:
    observing_new: List[ModelEndpointRecord] = field(default_factory=list)
    observing: List[ModelEndpointRecord] = field(default_factory=list)
    flagged_new: List[ModelEndpointRecord] = field(default_factory=list)
    in_grace: List[ModelEndpointRecord] = field(default_factory=list)
    deleted: List[ModelEndpointRecord] = field(default_factory=list)
    delete_failed: List[ModelEndpointRecord] = field(default_factory=list)
    delete_deferred: List[ModelEndpointRecord] = field(default_factory=list)  # cap or dry run
    cleared: List[ModelEndpointRecord] = field(default_factory=list)
    exempt: List[ModelEndpointRecord] = field(default_factory=list)
    queue_unknown: List[ModelEndpointRecord] = field(default_factory=list)
    # k8s labels (team, product, ...) per endpoint id, for the digest.
    labels: Dict[str, Dict[str, str]] = field(default_factory=dict)


def _parse_ts(value: object) -> Optional[datetime]:
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def _desired_and_available(infra_state: ModelEndpointInfraState) -> tuple[int, int]:
    state = infra_state.deployment_state
    available = state.available_workers or 0
    unavailable = state.unavailable_workers or 0
    return available + unavailable, available


class EndpointGarbageCollectionService:
    def __init__(
        self,
        model_endpoint_record_repository: ModelEndpointRecordRepository,
        resource_gateway: EndpointResourceGateway,
        model_endpoint_service: ModelEndpointService,
        queue_activity_gateway: QueueActivityGateway,
        digest_gateway: DigestGateway,
        config: EndpointGcConfig,
        now: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
    ):
        self.record_repository = model_endpoint_record_repository
        self.resource_gateway = resource_gateway
        self.model_endpoint_service = model_endpoint_service
        self.queue_activity_gateway = queue_activity_gateway
        self.digest_gateway = digest_gateway
        self.config = config
        self.now = now

    async def execute(self) -> EndpointGcReport:
        run_at = self.now()
        report = EndpointGcReport()
        infra_states = await self.resource_gateway.get_all_resources()
        states_by_id: Dict[str, ModelEndpointInfraState] = {
            key: state for key, (is_endpoint_id, state) in infra_states.items() if is_endpoint_id
        }
        report.labels = {key: dict(state.labels or {}) for key, state in states_by_id.items()}
        records = await self.record_repository.list_model_endpoint_records(
            owner=None, name=None, order_by=None
        )

        ready: List[tuple[datetime, ModelEndpointRecord]] = []
        for record in records:
            if record.status == ModelEndpointStatus.DELETE_IN_PROGRESS:
                continue
            metadata = dict(record.metadata or {})
            if metadata.get(GC_EXEMPT_KEY):
                report.exempt.append(record)
                continue

            infra_state = states_by_id.get(record.id)
            qualifies = (
                False
                if infra_state is None
                else await self._qualifies(record, infra_state, run_at, report)
            )
            if qualifies is None:
                # Unknown queue activity: neither advance nor reset the clock this run.
                continue
            if not qualifies:
                if any(key in metadata for key in GC_KEYS):
                    await self._write_metadata(
                        record, {k: v for k, v in metadata.items() if k not in GC_KEYS}
                    )
                    report.cleared.append(record)
                continue

            since = _parse_ts(metadata.get(GC_UNAVAILABLE_SINCE_KEY))
            if since is None:
                metadata[GC_UNAVAILABLE_SINCE_KEY] = run_at.isoformat()
                await self._write_metadata(record, metadata)
                report.observing_new.append(record)
                continue
            if run_at - since < timedelta(days=self.config.unavailable_days):
                report.observing.append(record)
                continue

            flagged_at = _parse_ts(metadata.get(GC_FLAGGED_AT_KEY))
            if flagged_at is None:
                metadata[GC_FLAGGED_AT_KEY] = run_at.isoformat()
                await self._write_metadata(record, metadata)
                report.flagged_new.append(record)
                continue
            if run_at - flagged_at < timedelta(days=self.config.grace_days):
                report.in_grace.append(record)
                continue
            ready.append((flagged_at, record))

        ready.sort(key=lambda item: item[0])
        for index, (_, record) in enumerate(ready):
            if index >= self.config.delete_cap:
                report.delete_deferred.append(record)
                continue
            if not self.config.apply:
                report.delete_deferred.append(record)
                continue
            try:
                await self.model_endpoint_service.delete_model_endpoint(record.id)
                report.deleted.append(record)
            except Exception:
                logger.exception(f"GC failed to delete endpoint {record.id} ({record.name})")
                report.delete_failed.append(record)

        self.digest_gateway.send_digest(format_digest(report, self.config, run_at))
        return report

    async def _qualifies(
        self,
        record: ModelEndpointRecord,
        infra_state: ModelEndpointInfraState,
        run_at: datetime,
        report: EndpointGcReport,
    ) -> Optional[bool]:
        desired, available = _desired_and_available(infra_state)
        if desired == 0 or available > 0:
            return False
        if record.endpoint_type != ModelEndpointType.ASYNC:
            return True
        # A dead async endpoint still accepts messages into its queue; only treat it as unused
        # when nothing was enqueued for the whole window.
        window_start = run_at - timedelta(days=self.config.unavailable_days)
        sent = await self.queue_activity_gateway.messages_sent_since(record.id, window_start)
        if sent is None:
            report.queue_unknown.append(record)
            return None
        return sent == 0

    async def _write_metadata(self, record: ModelEndpointRecord, metadata: Dict) -> None:
        await self.record_repository.update_model_endpoint_record(
            model_endpoint_id=record.id, metadata=metadata
        )


def _describe(record: ModelEndpointRecord, labels: Dict[str, str]) -> str:
    team = labels.get("team", "?")
    product = labels.get("product", "?")
    return f"{record.name} ({record.id}) team={team} product={product} owner={record.owner}"


def format_digest(report: EndpointGcReport, config: EndpointGcConfig, run_at: datetime) -> str:
    mode = "APPLY" if config.apply else "DRY RUN (no deletions)"
    lines = [
        f"model-engine endpoint GC {run_at.strftime('%Y-%m-%d %H:%M UTC')} [{mode}] "
        f"unavailable>{config.unavailable_days}d, grace {config.grace_days}d, cap {config.delete_cap}/run",
        f"deleted {len(report.deleted)}, failed {len(report.delete_failed)}, "
        f"deferred {len(report.delete_deferred)}, newly flagged {len(report.flagged_new)}, "
        f"in grace {len(report.in_grace)}, observing {len(report.observing) + len(report.observing_new)}, "
        f"cleared {len(report.cleared)}, exempt {len(report.exempt)}, queue unknown {len(report.queue_unknown)}",
    ]
    sections = [
        ("Deleted", report.deleted),
        ("Delete failed", report.delete_failed),
        ("Deferred (cap or dry run)", report.delete_deferred),
        (f"Newly flagged, delete after {config.grace_days}d", report.flagged_new),
        ("In grace", report.in_grace),
        ("Newly observed unavailable", report.observing_new),
        ("Recovered, GC state cleared", report.cleared),
        ("Queue activity unknown, skipped", report.queue_unknown),
    ]
    for title, records in sections:
        if records:
            lines.append(f"\n{title} ({len(records)}):")
            lines.extend(f"  - {_describe(r, report.labels.get(r.id, {}))}" for r in records)
    return "\n".join(lines)
