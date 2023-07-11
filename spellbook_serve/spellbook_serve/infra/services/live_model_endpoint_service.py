from typing import Any, Dict, List, Optional

from datadog import statsd

from spellbook_serve.common.dtos.model_endpoints import ModelEndpointOrderBy
from spellbook_serve.common.settings import generate_deployment_name
from spellbook_serve.core.domain_exceptions import (
    ObjectAlreadyExistsException,
    ObjectNotFoundException,
)
from spellbook_serve.core.loggers import filename_wo_ext, make_logger
from spellbook_serve.domain.entities import (
    CallbackAuth,
    CpuSpecificationType,
    GpuType,
    ModelEndpoint,
    ModelEndpointInfraState,
    ModelEndpointRecord,
    ModelEndpointsSchema,
    ModelEndpointStatus,
    ModelEndpointType,
    StorageSpecificationType,
)
from spellbook_serve.domain.exceptions import EndpointDeleteFailedException
from spellbook_serve.domain.gateways import (
    AsyncModelEndpointInferenceGateway,
    ModelEndpointsSchemaGateway,
    StreamingModelEndpointInferenceGateway,
    SyncModelEndpointInferenceGateway,
)
from spellbook_serve.domain.services import ModelEndpointService
from spellbook_serve.infra.gateways import ModelEndpointInfraGateway
from spellbook_serve.infra.repositories import ModelEndpointCacheRepository
from spellbook_serve.infra.repositories.model_endpoint_record_repository import (
    ModelEndpointRecordRepository,
)

logger = make_logger(filename_wo_ext(__file__))

STATSD_CACHE_HIT_NAME = "spellbook_serve.get_infra_state.cache_hit"
STATSD_CACHE_MISS_NAME = "spellbook_serve.get_infra_state.cache_miss"


class LiveModelEndpointService(ModelEndpointService):
    def __init__(
        self,
        model_endpoint_record_repository: ModelEndpointRecordRepository,
        model_endpoint_infra_gateway: ModelEndpointInfraGateway,
        model_endpoint_cache_repository: ModelEndpointCacheRepository,
        async_model_endpoint_inference_gateway: AsyncModelEndpointInferenceGateway,
        streaming_model_endpoint_inference_gateway: StreamingModelEndpointInferenceGateway,
        sync_model_endpoint_inference_gateway: SyncModelEndpointInferenceGateway,
        model_endpoints_schema_gateway: ModelEndpointsSchemaGateway,
    ):
        self.model_endpoint_record_repository = model_endpoint_record_repository
        self.model_endpoint_infra_gateway = model_endpoint_infra_gateway
        self.model_endpoint_cache_repository = model_endpoint_cache_repository
        self.async_model_endpoint_inference_gateway = async_model_endpoint_inference_gateway
        self.streaming_model_endpoint_inference_gateway = streaming_model_endpoint_inference_gateway
        self.sync_model_endpoint_inference_gateway = sync_model_endpoint_inference_gateway
        self.model_endpoints_schema_gateway = model_endpoints_schema_gateway

    def get_async_model_endpoint_inference_gateway(
        self,
    ) -> AsyncModelEndpointInferenceGateway:
        return self.async_model_endpoint_inference_gateway

    def get_sync_model_endpoint_inference_gateway(
        self,
    ) -> SyncModelEndpointInferenceGateway:
        return self.sync_model_endpoint_inference_gateway

    def get_streaming_model_endpoint_inference_gateway(
        self,
    ) -> StreamingModelEndpointInferenceGateway:
        return self.streaming_model_endpoint_inference_gateway

    async def _get_model_endpoint_infra_state(
        self, record: ModelEndpointRecord, use_cache: bool
    ) -> Optional[ModelEndpointInfraState]:
        """
        Gets state of model endpoint infra. Tries cache first, otherwise
        Args:
            record: Endpoint record
            use_cache: Whether it's safe to read from cache

        Returns:

        """
        state = None
        if use_cache:
            deployment_name = generate_deployment_name(record.created_by, record.name)
            state = await self.model_endpoint_cache_repository.read_endpoint_info(
                endpoint_id=record.id, deployment_name=deployment_name
            )
            tags = [
                f"endpoint_name:{record.name}",
                f"user_id:{record.created_by}",
                f"team_id:{record.owner}",
            ]
            if state is None:
                statsd.increment(STATSD_CACHE_MISS_NAME, 1, tags=tags)
                logger.warning(
                    f"Cache miss, reading directly from k8s for {deployment_name}",
                    extra={
                        "endpoint_name": record.name,
                        "endpoint_id": record.id,
                        "user_id": record.created_by,
                        "team_id": record.owner,
                    },
                )
            else:
                statsd.increment(STATSD_CACHE_HIT_NAME, 1, tags=tags)
        if state is None:
            state = await self.model_endpoint_infra_gateway.get_model_endpoint_infra(
                model_endpoint_record=record
            )
            if state is not None:
                await self.model_endpoint_cache_repository.write_endpoint_info(
                    endpoint_id=record.id, endpoint_info=state, ttl_seconds=60
                )
        return state

    async def create_model_endpoint(
        self,
        *,
        name: str,
        created_by: str,
        model_bundle_id: str,
        endpoint_type: ModelEndpointType,
        metadata: Optional[Dict[str, Any]],
        post_inference_hooks: Optional[List[str]],
        child_fn_info: Optional[Dict[str, Any]],
        cpus: CpuSpecificationType,
        gpus: int,
        memory: StorageSpecificationType,
        gpu_type: Optional[GpuType],
        storage: Optional[StorageSpecificationType],
        optimize_costs: bool,
        min_workers: int,
        max_workers: int,
        per_worker: int,
        labels: Dict[str, str],
        aws_role: str,
        results_s3_bucket: str,
        prewarm: bool,
        high_priority: Optional[bool],
        owner: str,
        default_callback_url: Optional[str] = None,
        default_callback_auth: Optional[CallbackAuth],
        public_inference: Optional[bool] = False,
    ) -> ModelEndpointRecord:
        existing_endpoints = (
            await self.model_endpoint_record_repository.list_model_endpoint_records(
                owner=owner, name=name, order_by=None
            )
        )
        if len(existing_endpoints) > 0:
            raise ObjectAlreadyExistsException

        model_endpoint_record = (
            await self.model_endpoint_record_repository.create_model_endpoint_record(
                name=name,
                created_by=created_by,
                model_bundle_id=model_bundle_id,
                metadata=metadata,
                endpoint_type=endpoint_type,
                destination="UNKNOWN",
                creation_task_id="UNKNOWN",
                status=ModelEndpointStatus.UPDATE_PENDING,
                owner=owner,
                public_inference=public_inference,
            )
        )
        creation_task_id = self.model_endpoint_infra_gateway.create_model_endpoint_infra(
            model_endpoint_record=model_endpoint_record,
            min_workers=min_workers,
            max_workers=max_workers,
            per_worker=per_worker,
            cpus=cpus,
            gpus=gpus,
            memory=memory,
            gpu_type=gpu_type,
            storage=storage,
            optimize_costs=optimize_costs,
            aws_role=aws_role,
            results_s3_bucket=results_s3_bucket,
            child_fn_info=child_fn_info,
            post_inference_hooks=post_inference_hooks,
            labels=labels,
            prewarm=prewarm,
            high_priority=high_priority,
            default_callback_url=default_callback_url,
            default_callback_auth=default_callback_auth,
        )
        await self.model_endpoint_record_repository.update_model_endpoint_record(
            model_endpoint_id=model_endpoint_record.id,
            creation_task_id=creation_task_id,
        )
        model_endpoint_record.creation_task_id = creation_task_id
        return model_endpoint_record

    async def get_model_endpoint(self, model_endpoint_id: str) -> Optional[ModelEndpoint]:
        model_endpoint_record = (
            await self.model_endpoint_record_repository.get_model_endpoint_record(
                model_endpoint_id=model_endpoint_id
            )
        )
        if model_endpoint_record is None:
            return None

        # TODO we might encounter weird cache staleness issues, as the cache is potentially up
        #   to a minute behind
        model_endpoint_infra_state = await self._get_model_endpoint_infra_state(
            record=model_endpoint_record, use_cache=True
        )
        return ModelEndpoint(record=model_endpoint_record, infra_state=model_endpoint_infra_state)

    async def get_model_endpoints_schema(self, owner: str) -> ModelEndpointsSchema:
        model_endpoint_records = (
            await self.model_endpoint_record_repository.list_model_endpoint_records(
                owner=owner, name=None, order_by=None
            )
        )
        return self.model_endpoints_schema_gateway.get_model_endpoints_schema(
            model_endpoint_records
        )

    async def get_model_endpoint_record(
        self, model_endpoint_id: str
    ) -> Optional[ModelEndpointRecord]:
        return await self.model_endpoint_record_repository.get_model_endpoint_record(
            model_endpoint_id=model_endpoint_id
        )

    async def list_model_endpoints(
        self,
        owner: Optional[str],
        name: Optional[str],
        order_by: Optional[ModelEndpointOrderBy],
    ) -> List[ModelEndpoint]:
        # Will read from cache at first
        records = await self.model_endpoint_record_repository.list_model_endpoint_records(
            owner=owner, name=name, order_by=order_by
        )
        endpoints: List[ModelEndpoint] = []
        for record in records:
            infra_state = await self._get_model_endpoint_infra_state(record=record, use_cache=True)
            endpoints.append(ModelEndpoint(record=record, infra_state=infra_state))
        return endpoints

    async def update_model_endpoint(
        self,
        *,
        model_endpoint_id: str,
        model_bundle_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        post_inference_hooks: Optional[Any] = None,
        cpus: Optional[CpuSpecificationType] = None,
        gpus: Optional[int] = None,
        memory: Optional[StorageSpecificationType] = None,
        gpu_type: Optional[GpuType] = None,
        storage: Optional[StorageSpecificationType] = None,
        optimize_costs: Optional[bool] = None,
        min_workers: Optional[int] = None,
        max_workers: Optional[int] = None,
        per_worker: Optional[int] = None,
        labels: Optional[Dict[str, str]] = None,
        prewarm: Optional[bool] = None,
        high_priority: Optional[bool] = None,
        default_callback_url: Optional[str] = None,
        default_callback_auth: Optional[CallbackAuth] = None,
        public_inference: Optional[bool] = None,
    ) -> ModelEndpointRecord:
        record = await self.model_endpoint_record_repository.get_model_endpoint_record(
            model_endpoint_id=model_endpoint_id
        )
        if record is None:
            raise ObjectNotFoundException

        async with self.model_endpoint_record_repository.get_lock_context(record) as lock:
            name = record.name
            created_by = record.created_by
            if not lock.lock_acquired():
                logger.warning(f"Lock was not successfully acquired by endpoint '{name}'")
                # TODO: we should raise an exception here when locking is fixed.
                # raise ExistingEndpointOperationInProgressException(
                #     f"Existing operation on endpoint {name} in progress, try again later"
                # )
            logger.info(f"Endpoint update acquired lock for {created_by}, {name}")
            if record.status in {
                ModelEndpointStatus.UPDATE_IN_PROGRESS,
                ModelEndpointStatus.UPDATE_PENDING,
            }:
                # Resource update is proceeding, abort to prevent a race
                logger.warning(f"Existing endpoint update on '{name}' in progress")
                # raise ExistingEndpointOperationInProgressException(
                #     f"Resource update on endpoint {name} in progress, try again later"
                # )

            record = await self.model_endpoint_record_repository.update_model_endpoint_record(
                model_endpoint_id=model_endpoint_id,
                model_bundle_id=model_bundle_id,
                metadata=metadata,
                status=ModelEndpointStatus.UPDATE_PENDING,
                public_inference=public_inference,
            )
            if record is None:  # pragma: no cover
                raise ObjectNotFoundException
            creation_task_id = await self.model_endpoint_infra_gateway.update_model_endpoint_infra(
                model_endpoint_record=record,
                min_workers=min_workers,
                max_workers=max_workers,
                per_worker=per_worker,
                cpus=cpus,
                gpus=gpus,
                memory=memory,
                gpu_type=gpu_type,
                storage=storage,
                optimize_costs=optimize_costs,
                post_inference_hooks=post_inference_hooks,
                labels=labels,
                prewarm=prewarm,
                high_priority=high_priority,
                default_callback_url=default_callback_url,
                default_callback_auth=default_callback_auth,
            )
            await self.model_endpoint_record_repository.update_model_endpoint_record(
                model_endpoint_id=model_endpoint_id,
                creation_task_id=creation_task_id,
            )

        record = await self.model_endpoint_record_repository.get_model_endpoint_record(
            model_endpoint_id=model_endpoint_id
        )
        if record is None:  # pragma: no cover
            raise ObjectNotFoundException
        return record

    async def delete_model_endpoint(self, model_endpoint_id: str) -> None:
        record = await self.model_endpoint_record_repository.get_model_endpoint_record(
            model_endpoint_id=model_endpoint_id
        )
        if record is None:
            raise ObjectNotFoundException

        async with self.model_endpoint_record_repository.get_lock_context(record) as lock:
            name = record.name
            created_by = record.created_by
            if not lock.lock_acquired() or record.status == ModelEndpointStatus.UPDATE_IN_PROGRESS:
                logger.warning(f"Existing operation on endpoint {name} in progress.")
            else:
                logger.info(f"Endpoint delete acquired lock for {created_by}, {name}")

            await self.model_endpoint_record_repository.update_model_endpoint_record(
                model_endpoint_id=model_endpoint_id,
                status=ModelEndpointStatus.DELETE_IN_PROGRESS,
            )

            infra_deleted = await self.model_endpoint_infra_gateway.delete_model_endpoint_infra(
                model_endpoint_record=record
            )
            if not infra_deleted:
                await self.model_endpoint_record_repository.update_model_endpoint_record(
                    model_endpoint_id=model_endpoint_id,
                    status=ModelEndpointStatus.UPDATE_FAILED,
                )
                raise EndpointDeleteFailedException

            logger.info(f"Deleting endpoint {name} for user {created_by} from db")

            await self.model_endpoint_record_repository.delete_model_endpoint_record(
                model_endpoint_id=model_endpoint_id
            )

        logger.info(f"Endpoint delete released lock for {created_by}, {name}")
