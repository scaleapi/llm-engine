from contextlib import AsyncExitStack
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

from cachetools import TTLCache
from model_engine_server.common import dict_not_none
from model_engine_server.common.dtos.model_endpoints import ModelEndpointOrderBy
from model_engine_server.core.loggers import logger_name, make_logger
from model_engine_server.db.endpoint_row_lock import AdvisoryLockContextManager, get_lock_key
from model_engine_server.db.models import Endpoint as OrmModelEndpoint
from model_engine_server.domain.entities import ModelEndpointRecord
from model_engine_server.domain.gateways import MonitoringMetricsGateway
from model_engine_server.infra.repositories.db_model_bundle_repository import (
    translate_model_bundle_orm_to_model_bundle,
)
from model_engine_server.infra.repositories.db_repository_mixin import (
    DbRepositoryMixin,
    raise_if_read_only,
)
from model_engine_server.infra.repositories.model_endpoint_record_repository import (
    ModelEndpointRecordRepository,
)
from sqlalchemy import or_, text
from sqlalchemy.ext.asyncio import AsyncSession

logger = make_logger(logger_name())

CACHE_SIZE = 512
CACHE_TTL_SECONDS = 15.0  # Kubernetes caching is 15 seconds as well

# The cache is not thread-safe, but for now that is not an issue because we
# use only 1 thread per process in production.
cache: TTLCache = TTLCache(maxsize=CACHE_SIZE, ttl=CACHE_TTL_SECONDS)


def translate_model_endpoint_orm_to_model_endpoint_record(
    model_endpoint_orm: OrmModelEndpoint,
) -> ModelEndpointRecord:
    current_model_bundle = translate_model_bundle_orm_to_model_bundle(
        model_endpoint_orm.current_bundle
    )
    return ModelEndpointRecord(
        id=model_endpoint_orm.id,
        name=model_endpoint_orm.name,
        created_by=model_endpoint_orm.created_by,
        owner=model_endpoint_orm.owner,
        created_at=model_endpoint_orm.created_at,
        last_updated_at=model_endpoint_orm.last_updated_at,
        metadata=model_endpoint_orm.endpoint_metadata,
        creation_task_id=model_endpoint_orm.creation_task_id,
        endpoint_type=model_endpoint_orm.endpoint_type,
        destination=model_endpoint_orm.destination,
        status=model_endpoint_orm.endpoint_status,
        current_model_bundle=current_model_bundle,
        public_inference=model_endpoint_orm.public_inference,
    )


class DbModelEndpointRecordRepository(ModelEndpointRecordRepository, DbRepositoryMixin):
    """
    Implementation of a ModelEndpointRecordRepository that is backed by a relational database.
    """

    def __init__(
        self,
        monitoring_metrics_gateway: MonitoringMetricsGateway,
        session: Callable[[], AsyncSession],
        read_only: bool,
    ):
        super().__init__(session=session, read_only=read_only)
        self.monitoring_metrics_gateway = monitoring_metrics_gateway

    class DbLockContext(ModelEndpointRecordRepository.LockContext):
        """
        Implementation of a LockContext that is backed by a relational database.
        """

        def __init__(self, lock_id: int, session: Callable[[], AsyncSession]):
            self._session = session
            self._exit_stack = AsyncExitStack()
            self._lock_id = lock_id
            self._lock_context_manager: Optional[AdvisoryLockContextManager] = None

        async def __aenter__(self):
            lock_session = await self._exit_stack.enter_async_context(self._session())
            self._lock_context_manager = await self._exit_stack.enter_async_context(
                AdvisoryLockContextManager(lock_session, self._lock_id)
            )
            return self

        async def __aexit__(self, exc_type, exc_val, exc_tb):
            await self._exit_stack.aclose()

        def lock_acquired(self) -> bool:
            if self._lock_context_manager is None:
                return False
            return self._lock_context_manager.lock_acquired()

    @raise_if_read_only
    def get_lock_context(
        self, model_endpoint_record: ModelEndpointRecord
    ) -> ModelEndpointRecordRepository.LockContext:
        lock_id = get_lock_key(
            user_id=model_endpoint_record.created_by,
            endpoint_name=model_endpoint_record.name,
        )
        return self.DbLockContext(lock_id=lock_id, session=self.session)

    @raise_if_read_only
    async def create_model_endpoint_record(
        self,
        *,
        name: str,
        created_by: str,
        model_bundle_id: str,
        metadata: Optional[Dict[str, Any]],
        endpoint_type: str,
        destination: str,
        creation_task_id: str,
        status: str,
        owner: str,
        public_inference: Optional[bool] = False,
        shadow_endpoints_ids: Optional[List[str]] = None,  # list of ids
    ) -> ModelEndpointRecord:
        model_endpoint_record = OrmModelEndpoint(
            name=name,
            created_by=created_by,
            current_bundle_id=model_bundle_id,
            endpoint_metadata=metadata,
            endpoint_type=endpoint_type,
            destination=destination,
            creation_task_id=creation_task_id,
            endpoint_status=status,
            owner=owner,
            public_inference=public_inference,
            shadow_endpoints_ids=shadow_endpoints_ids,
        )
        async with self.session() as session:
            await OrmModelEndpoint.create(session, model_endpoint_record)

            # HACK: Force a select_by_id to load the current_model_bundle relationship into the current session.
            # Otherwise, we'll get an error like:
            # sqlalchemy.orm.exc.DetachedInstanceError: Parent instance <Endpoint at 0x7fb1cb40d9a0>
            #   is not bound to a Session; lazy load operation of attribute 'current_bundle' cannot proceed
            # This is because there is no bound session to this ORM base model, and thus lazy
            # loading cannot occur without a session to execute the SELECT query.
            # See: https://docs.sqlalchemy.org/en/14/orm/loading_relationships.html
            model_endpoint_record = await OrmModelEndpoint.select_by_id(
                session=session,
                endpoint_id=model_endpoint_record.id,
            )
        return translate_model_endpoint_orm_to_model_endpoint_record(model_endpoint_record)

    async def list_model_endpoint_records(
        self,
        owner: Optional[str],
        name: Optional[str],
        order_by: Optional[ModelEndpointOrderBy],
    ) -> List[ModelEndpointRecord]:
        model_endpoints = cache.get((owner, name, order_by))
        if model_endpoints is None:
            self.monitoring_metrics_gateway.emit_database_cache_miss_metric()
            filters = []
            if owner:
                filters.append(OrmModelEndpoint.owner == owner)
            if name:
                filters.append(OrmModelEndpoint.name == name)

            async with self.session() as session:
                model_endpoints_orm = await OrmModelEndpoint._select_all_by_filters(
                    session=session, filters=filters
                )

            model_endpoints = [
                translate_model_endpoint_orm_to_model_endpoint_record(m)
                for m in model_endpoints_orm
            ]

            # TODO(phil): we could use an ORDER_BY operation in the DB instead.
            if order_by == ModelEndpointOrderBy.NEWEST:
                model_endpoints.sort(key=lambda x: x.created_at, reverse=True)
            elif order_by == ModelEndpointOrderBy.OLDEST:
                model_endpoints.sort(key=lambda x: x.created_at, reverse=False)
            cache[(owner, name, order_by)] = model_endpoints
        else:
            self.monitoring_metrics_gateway.emit_database_cache_hit_metric()

        return model_endpoints

    async def list_llm_model_endpoint_records(
        self,
        owner: Optional[str],
        name: Optional[str],
        order_by: Optional[ModelEndpointOrderBy],
    ) -> List[ModelEndpointRecord]:
        model_endpoints = cache.get(("llm", owner, name, order_by))
        if model_endpoints is None:
            self.monitoring_metrics_gateway.emit_database_cache_miss_metric()
            filters: List[Any] = []
            filters.append(text("endpoint_metadata ? '_llm'"))
            if name:
                filters.append(OrmModelEndpoint.name == name)
            ownership_filters = []
            if owner:
                ownership_filters.append(OrmModelEndpoint.owner == owner)
            filters.append(
                or_(*ownership_filters, OrmModelEndpoint.public_inference == True)  # noqa: E712
            )

            async with self.session() as session:
                model_endpoints_orm = await OrmModelEndpoint._select_all_by_filters(
                    session=session, filters=filters
                )

            model_endpoints = [
                translate_model_endpoint_orm_to_model_endpoint_record(m)
                for m in model_endpoints_orm
            ]

            # TODO(phil): we could use an ORDER_BY operation in the DB instead.
            if order_by == ModelEndpointOrderBy.NEWEST:
                model_endpoints.sort(key=lambda x: x.created_at, reverse=True)
            elif order_by == ModelEndpointOrderBy.OLDEST:
                model_endpoints.sort(key=lambda x: x.created_at, reverse=False)
            cache[("llm", owner, name, order_by)] = model_endpoints
        else:
            self.monitoring_metrics_gateway.emit_database_cache_hit_metric()

        return model_endpoints

    async def get_model_endpoint_record(
        self, model_endpoint_id: str
    ) -> Optional[ModelEndpointRecord]:
        model_endpoint = cache.get(model_endpoint_id)
        if model_endpoint is None:
            self.monitoring_metrics_gateway.emit_database_cache_miss_metric()
            async with self.session() as session:
                model_endpoint_orm = await OrmModelEndpoint.select_by_id(
                    session=session, endpoint_id=model_endpoint_id
                )
            if not model_endpoint_orm:
                return None

            model_endpoint = translate_model_endpoint_orm_to_model_endpoint_record(
                model_endpoint_orm
            )
            cache[model_endpoint_id] = model_endpoint
        else:
            self.monitoring_metrics_gateway.emit_database_cache_hit_metric()

        return model_endpoint

    async def get_llm_model_endpoint_record(
        self, model_endpoint_name: str
    ) -> Optional[ModelEndpointRecord]:
        model_endpoint = cache.get(("llm", model_endpoint_name))
        if model_endpoint is None:
            self.monitoring_metrics_gateway.emit_database_cache_miss_metric()
            async with self.session() as session:
                model_endpoints_orm = await OrmModelEndpoint._select_all_by_filters(
                    session=session,
                    filters=[
                        text("endpoint_metadata ? '_llm'"),
                        OrmModelEndpoint.name == model_endpoint_name,
                    ],
                )
            if len(model_endpoints_orm) == 0:
                return None
            if len(model_endpoints_orm) > 1:
                raise Exception(
                    f"Found multiple LLM endpoints with name {model_endpoint_name}. This should not happen."
                )

            model_endpoint = translate_model_endpoint_orm_to_model_endpoint_record(
                model_endpoints_orm[0]
            )
            cache[("llm", model_endpoint_name)] = model_endpoint
        else:
            self.monitoring_metrics_gateway.emit_database_cache_hit_metric()

        return model_endpoint

    @raise_if_read_only
    async def delete_model_endpoint_record(self, model_endpoint_id: str) -> bool:
        async with self.session() as session:
            model_endpoint_orm = await OrmModelEndpoint.select_by_id(
                session=session, endpoint_id=model_endpoint_id
            )
            if not model_endpoint_orm:
                return False

            await OrmModelEndpoint.delete(session=session, endpoint=model_endpoint_orm)

        if model_endpoint_id in cache:
            del cache[model_endpoint_id]

        return True

    @raise_if_read_only
    async def update_model_endpoint_record(
        self,
        *,
        model_endpoint_id: str,
        model_bundle_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        creation_task_id: Optional[str] = None,
        destination: Optional[str] = None,
        status: Optional[str] = None,
        public_inference: Optional[bool] = None,
        shadow_endpoints_ids: Optional[List[str]] = None,
    ) -> Optional[ModelEndpointRecord]:
        async with self.session() as session:
            model_endpoint_orm = await OrmModelEndpoint.select_by_id(
                session=session, endpoint_id=model_endpoint_id
            )
            if model_endpoint_orm is None:
                return None

        # Start a new session so that the current_bundle field is not cached in case the endpoint's
        # bundle is updated.
        async with self.session() as session:
            update_kwargs: Dict[str, Any] = dict_not_none(
                current_bundle_id=model_bundle_id,
                endpoint_metadata=metadata,
                creation_task_id=creation_task_id,
                destination=destination,
                endpoint_status=status,
                last_updated_at=datetime.utcnow(),
                public_inference=public_inference,
                shadow_endpoints_ids=shadow_endpoints_ids,
            )
            await OrmModelEndpoint.update_by_name_owner(
                session=session,
                name=model_endpoint_orm.name,
                owner=model_endpoint_orm.owner,
                kwargs=update_kwargs,
            )
            updated_model_endpoint_orm = await OrmModelEndpoint.select_by_id(
                session=session, endpoint_id=model_endpoint_id
            )

        model_endpoint = translate_model_endpoint_orm_to_model_endpoint_record(
            updated_model_endpoint_orm
        )
        cache[model_endpoint_id] = model_endpoint
        return model_endpoint
