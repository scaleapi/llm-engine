from collections import OrderedDict, defaultdict
from dataclasses import dataclass
from datetime import datetime
from typing import (
    IO,
    Any,
    AsyncIterable,
    DefaultDict,
    Dict,
    Iterator,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
)
from unittest import mock
from unittest.mock import mock_open
from uuid import uuid4

import pytest
from model_engine_server.api.dependencies import ExternalInterfaces
from model_engine_server.common.constants import DEFAULT_CELERY_TASK_NAME
from model_engine_server.common.dtos.batch_jobs import CreateDockerImageBatchJobResourceRequests
from model_engine_server.common.dtos.docker_repository import BuildImageRequest, BuildImageResponse
from model_engine_server.common.dtos.endpoint_builder import BuildEndpointRequest
from model_engine_server.common.dtos.model_bundles import ModelBundleOrderBy
from model_engine_server.common.dtos.model_endpoints import (
    BrokerType,
    CpuSpecificationType,
    GpuType,
    ModelEndpointOrderBy,
    StorageSpecificationType,
)
from model_engine_server.common.dtos.resource_manager import CreateOrUpdateResourcesRequest
from model_engine_server.common.dtos.tasks import (
    CreateAsyncTaskV1Response,
    EndpointPredictV1Request,
    GetAsyncTaskV1Response,
    SyncEndpointPredictV1Request,
    SyncEndpointPredictV1Response,
    TaskStatus,
)
from model_engine_server.common.settings import generate_destination
from model_engine_server.core.fake_notification_gateway import FakeNotificationGateway
from model_engine_server.core.tracing.live_tracing_gateway import LiveTracingGateway
from model_engine_server.db.endpoint_row_lock import get_lock_key
from model_engine_server.db.models import BatchJob as OrmBatchJob
from model_engine_server.db.models import Endpoint as OrmModelEndpoint
from model_engine_server.domain.entities import (
    BatchJob,
    BatchJobProgress,
    BatchJobRecord,
    BatchJobSerializationFormat,
    BatchJobStatus,
    CallbackAuth,
    CallbackBasicAuth,
    CloudpickleArtifactFlavor,
    CustomFramework,
    FileMetadata,
    FineTuneHparamValueType,
    LLMFineTuneEvent,
    ModelBundle,
    ModelBundleEnvironmentParams,
    ModelBundleFlavors,
    ModelBundleFrameworkType,
    ModelBundlePackagingType,
    ModelEndpoint,
    ModelEndpointConfig,
    ModelEndpointDeploymentState,
    ModelEndpointInfraState,
    ModelEndpointRecord,
    ModelEndpointResourceState,
    ModelEndpointsSchema,
    ModelEndpointStatus,
    ModelEndpointType,
    ModelEndpointUserConfigState,
    PytorchFramework,
    RunnableImageFlavor,
    StreamingEnhancedRunnableImageFlavor,
    TensorflowFramework,
    Trigger,
    TritonEnhancedRunnableImageFlavor,
    ZipArtifactFlavor,
)
from model_engine_server.domain.entities.batch_job_entity import DockerImageBatchJob
from model_engine_server.domain.entities.docker_image_batch_job_bundle_entity import (
    DockerImageBatchJobBundle,
)
from model_engine_server.domain.entities.llm_fine_tune_entity import LLMFineTuneTemplate
from model_engine_server.domain.exceptions import (
    EndpointResourceInfraException,
    ObjectNotFoundException,
)
from model_engine_server.domain.gateways import (
    AsyncModelEndpointInferenceGateway,
    CronJobGateway,
    DockerImageBatchJobGateway,
    FileStorageGateway,
    InferenceAutoscalingMetricsGateway,
    LLMArtifactGateway,
    StreamingModelEndpointInferenceGateway,
    SyncModelEndpointInferenceGateway,
    TaskQueueGateway,
)
from model_engine_server.domain.repositories import (
    DockerImageBatchJobBundleRepository,
    DockerRepository,
    LLMFineTuneEventsRepository,
    ModelBundleRepository,
    TokenizerRepository,
    TriggerRepository,
)
from model_engine_server.domain.services import (
    LLMFineTuningService,
    LLMModelEndpointService,
    ModelEndpointService,
)
from model_engine_server.inference.domain.gateways.streaming_storage_gateway import (
    StreamingStorageGateway,
)
from model_engine_server.infra.gateways import (
    BatchJobOrchestrationGateway,
    LiveBatchJobProgressGateway,
    LiveModelEndpointsSchemaGateway,
    ModelEndpointInfraGateway,
)
from model_engine_server.infra.gateways.fake_model_primitive_gateway import (
    FakeModelPrimitiveGateway,
)
from model_engine_server.infra.gateways.fake_monitoring_metrics_gateway import (
    FakeMonitoringMetricsGateway,
)
from model_engine_server.infra.gateways.filesystem_gateway import FilesystemGateway
from model_engine_server.infra.gateways.resources.endpoint_resource_gateway import (
    EndpointResourceGateway,
    EndpointResourceGatewayCreateOrUpdateResourcesResponse,
    QueueInfo,
)
from model_engine_server.infra.gateways.resources.image_cache_gateway import (
    CachedImages,
    ImageCacheGateway,
)
from model_engine_server.infra.repositories import (
    BatchJobRecordRepository,
    FeatureFlagRepository,
    LLMFineTuneRepository,
    ModelEndpointCacheRepository,
    ModelEndpointRecordRepository,
)
from model_engine_server.infra.repositories.db_model_bundle_repository import (
    translate_kwargs_to_model_bundle_orm,
    translate_model_bundle_orm_to_model_bundle,
)
from model_engine_server.infra.services import LiveBatchJobService, LiveModelEndpointService
from model_engine_server.infra.services.fake_llm_batch_completions_service import (
    FakeLLMBatchCompletionsService,
)
from model_engine_server.infra.services.image_cache_service import ImageCacheService
from model_engine_server.infra.services.live_llm_batch_completions_service import (
    LiveLLMBatchCompletionsService,
)
from model_engine_server.infra.services.live_llm_model_endpoint_service import (
    LiveLLMModelEndpointService,
)
from transformers import AutoTokenizer


def _translate_fake_model_endpoint_orm_to_model_endpoint_record(
    model_endpoint_orm: OrmModelEndpoint, current_model_bundle: ModelBundle
) -> ModelEndpointRecord:
    # Needed since the orm model has a column `endpoint_metadata` that turns into `metadata` in our model
    return ModelEndpointRecord(
        id=model_endpoint_orm.id,
        name=model_endpoint_orm.name,
        created_by=model_endpoint_orm.created_by,
        owner=model_endpoint_orm.owner,
        created_at=model_endpoint_orm.created_at,
        last_updated_at=model_endpoint_orm.last_updated_at,
        metadata=model_endpoint_orm.endpoint_metadata,
        creation_task_id=model_endpoint_orm.creation_task_id,
        endpoint_type=ModelEndpointType(model_endpoint_orm.endpoint_type),
        destination="test_destination",
        status=ModelEndpointStatus(model_endpoint_orm.endpoint_status),
        current_model_bundle=current_model_bundle,
    )


def _translate_fake_batch_job_orm_to_batch_job_record(
    batch_job_orm: OrmBatchJob, model_bundle: ModelBundle
) -> BatchJobRecord:
    return BatchJobRecord(
        id=batch_job_orm.id,
        created_at=batch_job_orm.created_at,
        completed_at=batch_job_orm.completed_at,
        status=BatchJobStatus(batch_job_orm.batch_job_status),
        created_by=batch_job_orm.created_by,
        owner=batch_job_orm.owner,
        model_bundle=model_bundle,
        model_endpoint_id=batch_job_orm.model_endpoint_id,
        task_ids_location=batch_job_orm.task_ids_location,
        result_location=batch_job_orm.result_location,
    )


class FakeModelBundleRepository(ModelBundleRepository):
    def __init__(self, contents: Optional[Dict[str, ModelBundle]] = None):
        if contents:
            self.db = contents
            self.unique_owner_name_versions = set()
            for model_bundle in self.db.values():
                self.unique_owner_name_versions.add((model_bundle.owner, model_bundle.name))
        else:
            self.db = {}
            self.unique_owner_name_versions = set()

    def add_model_bundle(self, model_bundle: ModelBundle):
        self.db[model_bundle.id] = model_bundle

    async def create_model_bundle(
        self,
        *,
        name: str,
        created_by: str,
        owner: str,
        model_artifact_ids: List[str],
        schema_location: Optional[str],
        metadata: Dict[str, Any],
        flavor: ModelBundleFlavors,
        # LEGACY FIELDS
        location: str,
        requirements: List[str],
        env_params: Dict[str, Any],
        packaging_type: ModelBundlePackagingType,
        app_config: Optional[Dict[str, Any]],
    ) -> ModelBundle:
        orm_model_bundle = translate_kwargs_to_model_bundle_orm(
            name=name,
            created_by=created_by,
            owner=owner,
            model_artifact_ids=model_artifact_ids,
            schema_location=schema_location,
            metadata=metadata,
            flavor=flavor,
            # LEGACY FIELDS
            location=location,
            requirements=requirements,
            env_params=env_params,
            packaging_type=packaging_type,
            app_config=app_config,
        )
        orm_model_bundle.created_at = datetime.now()
        model_bundle = translate_model_bundle_orm_to_model_bundle(orm_model_bundle)
        self.db[model_bundle.id] = model_bundle
        return self.db[model_bundle.id]

    async def list_model_bundles(
        self, owner: str, name: Optional[str], order_by: Optional[ModelBundleOrderBy]
    ) -> Sequence[ModelBundle]:
        model_bundles = [
            mb for mb in self.db.values() if mb.owner == owner and (not name or mb.name == name)
        ]

        if order_by == ModelBundleOrderBy.NEWEST:
            model_bundles.sort(key=lambda x: x.created_at, reverse=True)
        elif order_by == ModelBundleOrderBy.OLDEST:
            model_bundles.sort(key=lambda x: x.created_at, reverse=False)

        return model_bundles

    async def get_latest_model_bundle_by_name(self, owner: str, name: str) -> Optional[ModelBundle]:
        model_bundles = await self.list_model_bundles(owner, name, ModelBundleOrderBy.NEWEST)
        if not model_bundles:
            return None
        return model_bundles[0]

    async def get_model_bundle(self, model_bundle_id: str) -> Optional[ModelBundle]:
        return self.db.get(model_bundle_id)


class FakeBatchJobRecordRepository(BatchJobRecordRepository):
    db: Dict[str, BatchJobRecord]
    model_bundle_repository: ModelBundleRepository

    def __init__(
        self,
        contents: Optional[Dict[str, BatchJobRecord]] = None,
        model_bundle_repository: Optional[ModelBundleRepository] = None,
    ):
        if contents:
            self.db = contents
        else:
            self.db = {}

        if model_bundle_repository is None:
            model_bundle_repository = FakeModelBundleRepository()
        self.model_bundle_repository = model_bundle_repository

    def add_batch_job_record(self, batch_job_record: BatchJobRecord):
        self.db[batch_job_record.id] = batch_job_record

    async def create_batch_job_record(
        self,
        *,
        status: BatchJobStatus,
        created_by: str,
        owner: str,
        model_bundle_id: str,
    ) -> BatchJobRecord:
        orm_batch_job = OrmBatchJob(
            batch_job_status=status,
            created_by=created_by,
            owner=owner,
            model_bundle_id=model_bundle_id,
        )
        orm_batch_job.created_at = datetime.now()
        model_bundle = await self.model_bundle_repository.get_model_bundle(model_bundle_id)
        assert model_bundle is not None
        batch_job = _translate_fake_batch_job_orm_to_batch_job_record(
            orm_batch_job, model_bundle=model_bundle
        )
        self.db[batch_job.id] = batch_job
        return batch_job

    @staticmethod
    def update_batch_job_record_in_place(
        batch_job_record: BatchJobRecord,
        **kwargs,
    ):
        if kwargs["status"] is not None:
            batch_job_record.status = kwargs["status"]
        if kwargs["model_endpoint_id"] is not None:
            batch_job_record.model_endpoint_id = kwargs["model_endpoint_id"]
        if kwargs["task_ids_location"] is not None:
            batch_job_record.task_ids_location = kwargs["task_ids_location"]
        if kwargs["result_location"] is not None:
            batch_job_record.result_location = kwargs["result_location"]

    async def update_batch_job_record(
        self,
        *,
        batch_job_id: str,
        status: Optional[BatchJobStatus] = None,
        model_endpoint_id: Optional[str] = None,
        task_ids_location: Optional[str] = None,
        result_location: Optional[str] = None,
        completed_at: Optional[datetime] = None,
    ) -> Optional[BatchJobRecord]:
        batch_job_record = await self.get_batch_job_record(batch_job_id=batch_job_id)
        self.update_batch_job_record_in_place(**locals())
        return batch_job_record

    async def get_batch_job_record(self, batch_job_id: str) -> Optional[BatchJobRecord]:
        return self.db.get(batch_job_id)

    async def list_batch_job_records(self, owner: Optional[str]) -> List[BatchJobRecord]:
        def filter_fn(m: BatchJobRecord) -> bool:
            return not owner or m.owner == owner

        batch_jobs = list(filter(filter_fn, self.db.values()))
        return batch_jobs

    async def unset_model_endpoint_id(self, batch_job_id: str) -> Optional[BatchJobRecord]:
        batch_job_record = await self.get_batch_job_record(batch_job_id)
        if batch_job_record:
            batch_job_record.model_endpoint_id = None
        return batch_job_record


class FakeModelEndpointRecordRepository(ModelEndpointRecordRepository):
    db: Dict[str, ModelEndpointRecord]
    model_bundle_repository: ModelBundleRepository

    def __init__(
        self,
        contents: Optional[Dict[str, ModelEndpointRecord]] = None,
        model_bundle_repository: Optional[ModelBundleRepository] = None,
    ):
        if contents:
            self.db = contents
            self.unique_owner_name_versions = set()
            for model_endpoint in self.db.values():
                self.unique_owner_name_versions.add((model_endpoint.owner, model_endpoint.name))
        else:
            self.db = {}
            self.unique_owner_name_versions = set()

        if model_bundle_repository is None:
            model_bundle_repository = FakeModelBundleRepository()
        self.model_bundle_repository = model_bundle_repository

        self._lock_db: DefaultDict[int, bool] = defaultdict(bool)

    class FakeLockContext(ModelEndpointRecordRepository.LockContext):
        def __init__(self, lock_id: int, lock_db: DefaultDict[int, bool]):
            self._lock_id = lock_id
            self._lock_db = lock_db
            self._lock_acquired = False

        async def __aenter__(self):
            if not self._lock_db[self._lock_id]:
                self._lock_db[self._lock_id] = True
                self._lock_acquired = True
            return self

        async def __aexit__(self, exc_type, exc_val, exc_tb):
            if self._lock_acquired:
                self._lock_db[self._lock_id] = False
            self._lock_acquired = False

        def lock_acquired(self) -> bool:
            return self._lock_acquired

    def force_lock_model_endpoint(self, model_endpoint_record: ModelEndpointRecord):
        lock_id = get_lock_key(
            user_id=model_endpoint_record.created_by,
            endpoint_name=model_endpoint_record.name,
        )
        self._lock_db[lock_id] = True

    def force_unlock_model_endpoint(self, model_endpoint_record: ModelEndpointRecord):
        lock_id = get_lock_key(
            user_id=model_endpoint_record.created_by,
            endpoint_name=model_endpoint_record.name,
        )
        self._lock_db[lock_id] = False

    def get_lock_context(
        self, model_endpoint_record: ModelEndpointRecord
    ) -> ModelEndpointRecordRepository.LockContext:
        lock_id = get_lock_key(
            user_id=model_endpoint_record.created_by,
            endpoint_name=model_endpoint_record.name,
        )
        return self.FakeLockContext(lock_id=lock_id, lock_db=self._lock_db)

    def add_model_endpoint_record(self, model_endpoint_record: ModelEndpointRecord):
        self.db[model_endpoint_record.id] = model_endpoint_record

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
    ) -> ModelEndpointRecord:
        orm_model_endpoint = OrmModelEndpoint(
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
        )
        orm_model_endpoint.created_at = datetime.now()
        orm_model_endpoint.last_updated_at = datetime.now()
        model_bundle = await self.model_bundle_repository.get_model_bundle(model_bundle_id)
        assert model_bundle is not None
        model_endpoint = _translate_fake_model_endpoint_orm_to_model_endpoint_record(
            orm_model_endpoint, current_model_bundle=model_bundle
        )
        self.db[model_endpoint.id] = model_endpoint
        return model_endpoint

    @staticmethod
    def update_model_endpoint_record_in_place(
        model_endpoint_record: ModelEndpointRecord,
        **kwargs,
    ):
        if kwargs["current_model_bundle"] is not None:
            model_endpoint_record.current_model_bundle = kwargs["current_model_bundle"]
        if kwargs["metadata"] is not None:
            model_endpoint_record.metadata = kwargs["metadata"]
        if kwargs["creation_task_id"] is not None:
            model_endpoint_record.creation_task_id = kwargs["creation_task_id"]
        if kwargs["status"] is not None:
            model_endpoint_record.status = kwargs["status"]

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
    ) -> Optional[ModelEndpointRecord]:
        model_endpoint_record = await self.get_model_endpoint_record(
            model_endpoint_id=model_endpoint_id
        )
        current_model_bundle = None
        if model_bundle_id is not None:
            current_model_bundle = await self.model_bundle_repository.get_model_bundle(
                model_bundle_id
            )
        self.update_model_endpoint_record_in_place(**locals())
        return model_endpoint_record

    async def get_model_endpoint_record(
        self, model_endpoint_id: str
    ) -> Optional[ModelEndpointRecord]:
        return self.db.get(model_endpoint_id)

    async def get_llm_model_endpoint_record(
        self, model_endpoint_name: str
    ) -> Optional[ModelEndpointRecord]:
        def filter_fn(m: ModelEndpointRecord) -> bool:
            return "_llm" in m.metadata and m.name == model_endpoint_name

        model_endpoints = list(filter(filter_fn, self.db.values()))
        assert len(model_endpoints) <= 1

        if len(model_endpoints) == 0:
            return None
        else:
            return model_endpoints[0]

    async def list_model_endpoint_records(
        self,
        owner: Optional[str],
        name: Optional[str],
        order_by: Optional[ModelEndpointOrderBy],
    ) -> List[ModelEndpointRecord]:
        def filter_fn(m: ModelEndpointRecord) -> bool:
            return (not owner or m.owner == owner) and (not name or m.name == name)

        model_endpoints = list(filter(filter_fn, self.db.values()))

        if order_by == ModelEndpointOrderBy.NEWEST:
            model_endpoints.sort(key=lambda x: x.created_at, reverse=True)
        elif order_by == ModelEndpointOrderBy.OLDEST:
            model_endpoints.sort(key=lambda x: x.created_at, reverse=False)

        return model_endpoints

    async def list_llm_model_endpoint_records(
        self,
        owner: Optional[str],
        name: Optional[str],
        order_by: Optional[ModelEndpointOrderBy],
    ) -> List[ModelEndpointRecord]:
        def filter_fn(m: ModelEndpointRecord) -> bool:
            return ("_llm" in m.metadata) and (
                ((not owner or m.owner == owner) and (not name or m.name == name))
                or (m.public_inference is True)
            )

        model_endpoints = list(filter(filter_fn, self.db.values()))

        if order_by == ModelEndpointOrderBy.NEWEST:
            model_endpoints.sort(key=lambda x: x.created_at, reverse=True)
        elif order_by == ModelEndpointOrderBy.OLDEST:
            model_endpoints.sort(key=lambda x: x.created_at, reverse=False)

        return model_endpoints

    async def delete_model_endpoint_record(self, model_endpoint_id: str) -> bool:
        if model_endpoint_id not in self.db:
            return False
        del self.db[model_endpoint_id]
        return True


class FakeDockerImageBatchJobBundleRepository(DockerImageBatchJobBundleRepository):
    def __init__(self, contents: Optional[Dict[str, DockerImageBatchJobBundle]] = None):
        if contents:
            self.db = contents
        else:
            self.db = {}
        self.next_id = 0

    def _get_new_id(self):
        current_ids = {bun.id for bun in self.db.values()}
        while str(self.next_id) in current_ids:
            self.next_id += 1
        return str(self.next_id)

    def add_docker_image_batch_job_bundle(self, batch_bundle: DockerImageBatchJobBundle):
        new_id = batch_bundle.id
        if new_id in {bun.id for bun in self.db.values()}:
            raise ValueError(f"Error in test set up, batch bundle with {new_id} already present")
        self.db[new_id] = batch_bundle

    async def create_docker_image_batch_job_bundle(
        self,
        *,
        name: str,
        created_by: str,
        owner: str,
        image_repository: str,
        image_tag: str,
        command: List[str],
        env: Dict[str, str],
        mount_location: Optional[str],
        cpus: Optional[str],
        memory: Optional[str],
        storage: Optional[str],
        gpus: Optional[int],
        gpu_type: Optional[GpuType],
        public: Optional[bool],
    ) -> DockerImageBatchJobBundle:
        bun_id = self._get_new_id()
        batch_bundle = DockerImageBatchJobBundle(
            id=bun_id,
            created_at=datetime.now(),
            name=name,
            created_by=created_by,
            owner=owner,
            image_repository=image_repository,
            image_tag=image_tag,
            command=command,
            env=env,
            mount_location=mount_location,
            cpus=cpus,
            memory=memory,
            storage=storage,
            gpus=gpus,
            gpu_type=gpu_type,
            public=public,
        )
        self.db[bun_id] = batch_bundle
        return batch_bundle

    async def list_docker_image_batch_job_bundles(
        self, owner: str, name: Optional[str], order_by: Optional[ModelBundleOrderBy]
    ) -> Sequence[DockerImageBatchJobBundle]:
        def filter_fn(dibun: DockerImageBatchJobBundle):
            return (dibun.owner == owner) and (name is None or dibun.name == name)

        buns = [dibun for dibun in self.db.values() if filter_fn(dibun)]
        if order_by == ModelBundleOrderBy.NEWEST:
            buns.sort(key=lambda x: x.created_at, reverse=True)
        elif order_by == ModelBundleOrderBy.OLDEST:
            buns.sort(key=lambda x: x.created_at, reverse=False)

        return buns

    async def get_docker_image_batch_job_bundle(
        self, docker_image_batch_job_bundle_id: str
    ) -> Optional[DockerImageBatchJobBundle]:
        return self.db.get(docker_image_batch_job_bundle_id)

    async def get_latest_docker_image_batch_job_bundle(
        self, owner: str, name: str
    ) -> Optional[DockerImageBatchJobBundle]:
        def filter_fn(dibun: DockerImageBatchJobBundle):
            return (dibun.owner == owner) and (dibun.name == name)

        buns = [dibun for dibun in self.db.values() if filter_fn(dibun)]
        if len(buns) == 0:
            return None
        return max(buns, key=lambda bun: bun.created_at)


class FakeDockerRepository(DockerRepository):
    def __init__(self, image_always_exists: bool, raises_error: bool):
        self.image_always_exists = image_always_exists
        self.raises_error = raises_error

    def image_exists(
        self, image_tag: str, repository_name: str, aws_profile: Optional[str] = None
    ) -> bool:
        return self.image_always_exists

    def get_image_url(self, image_tag: str, repository_name: str) -> str:
        return f"{repository_name}:{image_tag}"

    def build_image(self, image_params: BuildImageRequest) -> BuildImageResponse:
        if self.raises_error:
            raise Exception("I hope you're handling this!")
        return BuildImageResponse(status=True, logs="", job_name="test-job-name")

    def get_latest_image_tag(self, repository_name: str) -> str:
        return "fake_docker_repository_latest_image_tag"


class FakeModelEndpointCacheRepository(ModelEndpointCacheRepository):
    def __init__(self):
        self.db = {}

    async def write_endpoint_info(
        self,
        endpoint_id: str,
        endpoint_info: ModelEndpointInfraState,
        ttl_seconds: float,
    ):
        self.db[endpoint_id] = endpoint_info

    async def read_endpoint_info(
        self, endpoint_id: str, deployment_name: str
    ) -> Optional[ModelEndpointInfraState]:
        return self.db.get(endpoint_id, None)

    def force_expire_key(self, endpoint_id: str):
        """
        Use to simulate key expiring
        """
        if endpoint_id in self.db:
            del self.db[endpoint_id]


class FakeFeatureFlagRepository(FeatureFlagRepository):
    def __init__(self):
        self.db = {}

    async def write_feature_flag_bool(
        self,
        key: str,
        value: bool,
    ):
        self.db[key] = value

    async def read_feature_flag_bool(
        self,
        key: str,
    ) -> Optional[bool]:
        return self.db.get(key, None)


class FakeLLMFineTuneRepository(LLMFineTuneRepository):
    def __init__(self, db: Optional[Dict[Tuple[str, str], LLMFineTuneTemplate]] = None):
        self.db = db
        if self.db is None:
            self.db = {}

    async def get_job_template_for_model(
        self, model_name: str, fine_tuning_method: str
    ) -> Optional[LLMFineTuneTemplate]:
        return self.db.get((model_name, fine_tuning_method), None)

    async def write_job_template_for_model(
        self,
        model_name: str,
        fine_tuning_method: str,
        job_template: LLMFineTuneTemplate,
    ):
        self.db[(model_name, fine_tuning_method)] = job_template


class FakeLLMFineTuneEventsRepository(LLMFineTuneEventsRepository):
    def __init__(self):
        self.initialized_events = []
        self.all_events_list = [LLMFineTuneEvent(timestamp=1, message="message", level="info")]

    async def get_fine_tune_events(self, user_id: str, model_endpoint_name: str):
        if (user_id, model_endpoint_name) in self.initialized_events:
            return self.all_events_list
        raise ObjectNotFoundException

    async def initialize_events(self, user_id: str, model_endpoint_name: str):
        self.initialized_events.append((user_id, model_endpoint_name))


class FakeLLMArtifactGateway(LLMArtifactGateway):
    def __init__(self):
        self.existing_models = []
        self.s3_bucket = {
            "fake-checkpoint": [
                "model-fake.bin, model-fake2.bin",
                "model-fake.safetensors",
            ],
            "llama-7b/tokenizer.json": ["llama-7b/tokenizer.json"],
            "llama-7b/tokenizer_config.json": ["llama-7b/tokenizer_config.json"],
            "llama-7b/special_tokens_map.json": ["llama-7b/special_tokens_map.json"],
            "llama-2-7b": ["model-fake.safetensors"],
            "mpt-7b": ["model-fake.safetensors"],
            "llama-3-70b": ["model-fake.safetensors"],
            "llama-3-1-405b-instruct": ["model-fake.safetensors"],
        }
        self.urls = {"filename": "https://test-bucket.s3.amazonaws.com/llm/llm-1.0.0.tar.gz"}
        self.model_config = {
            "_name_or_path": "meta-llama/Llama-2-7b-hf",
            "architectures": ["LlamaForCausalLM"],
            "bos_token_id": 1,
            "eos_token_id": 2,
            "hidden_act": "silu",
            "hidden_size": 4096,
            "initializer_range": 0.02,
            "intermediate_size": 11008,
            "max_position_embeddings": 4096,
            "model_type": "llama",
            "num_attention_heads": 32,
            "num_hidden_layers": 32,
            "num_key_value_heads": 32,
            "pretraining_tp": 1,
            "rms_norm_eps": 1e-05,
            "rope_scaling": None,
            "tie_word_embeddings": False,
            "torch_dtype": "float16",
            "transformers_version": "4.31.0.dev0",
            "use_cache": True,
            "vocab_size": 32000,
        }
        self.tokenizer_config = {
            "add_bos_token": True,
            "add_eos_token": False,
            "add_prefix_space": None,
            "added_tokens_decoder": {
                "0": {
                    "content": "<unk>",
                    "lstrip": False,
                    "normalized": False,
                    "rstrip": False,
                    "single_word": False,
                    "special": True,
                },
                "1": {
                    "content": "<s>",
                    "lstrip": False,
                    "normalized": False,
                    "rstrip": False,
                    "single_word": False,
                    "special": True,
                },
                "2": {
                    "content": "</s>",
                    "lstrip": False,
                    "normalized": False,
                    "rstrip": False,
                    "single_word": False,
                    "special": True,
                },
            },
            "additional_special_tokens": [],
            "bos_token": "<s>",
            "chat_template": "{%- if messages[0]['role'] == 'system' %}\n    {%- set system_message = messages[0]['content'] %}\n    {%- set loop_messages = messages[1:] %}\n{%- else %}\n    {%- set loop_messages = messages %}\n{%- endif %}\n\n{{- bos_token }}\n{%- for message in loop_messages %}\n    {%- if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}\n        {{- raise_exception('After the optional system message, conversation roles must alternate user/assistant/user/assistant/...') }}\n    {%- endif %}\n    {%- if message['role'] == 'user' %}\n        {%- if loop.first and system_message is defined %}\n            {{- ' [INST] ' + system_message + '\\n\\n' + message['content'] + ' [/INST]' }}\n        {%- else %}\n            {{- ' [INST] ' + message['content'] + ' [/INST]' }}\n        {%- endif %}\n    {%- elif message['role'] == 'assistant' %}\n        {{- ' ' + message['content'] + eos_token}}\n    {%- else %}\n        {{- raise_exception('Only user and assistant roles are supported, with the exception of an initial optional system message!') }}\n    {%- endif %}\n{%- endfor %}\n",
            "clean_up_tokenization_spaces": False,
            "eos_token": "</s>",
            "legacy": False,
            "model_max_length": 1000000000000000019884624838656,
            "pad_token": None,
            "sp_model_kwargs": {},
            "spaces_between_special_tokens": False,
            "tokenizer_class": "LlamaTokenizer",
            "unk_token": "<unk>",
            "use_default_system_prompt": False,
        }

    def _add_model(self, owner: str, model_name: str):
        self.existing_models.append((owner, model_name))

    def list_files(self, path: str, **kwargs) -> List[str]:
        path = path.lstrip("s3://")
        if path in self.s3_bucket:
            return self.s3_bucket[path]

    def download_files(self, path: str, target_path: str, overwrite=False, **kwargs) -> List[str]:
        path = path.lstrip("s3://")
        if path in self.s3_bucket:
            return self.s3_bucket[path]

    def get_model_weights_urls(self, owner: str, model_name: str):
        if (owner, model_name) in self.existing_models:
            return self.urls
        raise ObjectNotFoundException

    def get_model_config(self, path: str, **kwargs) -> Dict[str, Any]:
        return self.model_config


class FakeTriggerRepository(TriggerRepository):  # pragma: no cover
    def __init__(self, contents: Optional[Dict[str, Trigger]] = None):
        self.db = {} if contents is None else contents
        self.next_id = 0

    def _get_new_id(self):
        new_id = f"trig_{self.next_id}"
        self.next_id += 1
        return new_id

    async def create_trigger(
        self,
        *,
        name: str,
        created_by: str,
        owner: str,
        cron_schedule: str,
        docker_image_batch_job_bundle_id: str,
        default_job_config: Optional[Dict[str, Any]],
        default_job_metadata: Optional[Dict[str, str]],
    ) -> Trigger:
        trigger_id = self._get_new_id()
        trigger = Trigger(
            id=trigger_id,
            name=name,
            owner=owner,
            created_by=created_by,
            created_at=datetime.now(),
            cron_schedule=cron_schedule,
            docker_image_batch_job_bundle_id=docker_image_batch_job_bundle_id,
            default_job_config=default_job_config,
            default_job_metadata=default_job_metadata,
        )
        self.db[trigger_id] = trigger
        return trigger

    async def list_triggers(
        self,
        owner: str,
    ) -> Sequence[Trigger]:
        def filter_fn(trig: Trigger) -> bool:
            return trig.owner == owner

        return list(filter(filter_fn, self.db.values()))

    async def get_trigger(
        self,
        trigger_id: str,
    ) -> Optional[Trigger]:
        return self.db.get(trigger_id)

    async def update_trigger(
        self,
        trigger_id: str,
        cron_schedule: str,
    ) -> bool:
        if trigger_id not in self.db:
            return False

        self.db[trigger_id].cron_schedule = cron_schedule
        return True

    async def delete_trigger(
        self,
        trigger_id: str,
    ) -> bool:
        if trigger_id not in self.db:
            return False

        del self.db[trigger_id]
        return True


class FakeImageCacheGateway(ImageCacheGateway):
    def __init__(self):
        self.cached_images = CachedImages(
            cpu=[], a10=[], a100=[], t4=[], h100=[], h100_1g20gb=[], h100_3g40gb=[]
        )

    async def create_or_update_image_cache(self, cached_images: CachedImages) -> None:
        self.cached_images = cached_images


class FakeBatchJobOrchestrationGateway(BatchJobOrchestrationGateway):
    def __init__(self):
        self.db = {}

    async def create_batch_job_orchestrator(
        self,
        job_id: str,
        resource_group_name: str,
        owner: str,
        input_path: str,
        serialization_format: BatchJobSerializationFormat,
        labels: Dict[str, str],
        timeout_seconds: float,
    ) -> None:
        self.db[resource_group_name] = {
            "id": job_id,
            "resource_group_name": resource_group_name,
            "owner": owner,
            "input_path": input_path,
            "serialization_format": serialization_format,
            "labels": labels,
        }

    async def delete_batch_job_orchestrator(self, resource_group_name: str) -> bool:
        del self.db[resource_group_name]
        return True


class FakeFilesystemGateway(FilesystemGateway):
    def __init__(self, read_data: Optional[str] = None):
        self.read_data = read_data or ""
        self.mock_open = mock_open(read_data=self.read_data)

    def open(self, uri: str, mode: str = "rt", **kwargs) -> IO:
        self.mock_open = mock_open(read_data=self.read_data)
        return self.mock_open(uri, mode=mode, **kwargs)

    def generate_signed_url(self, uri: str, expiration: int = 3600, **kwargs) -> str:
        return uri


class FakeTaskQueueGateway(TaskQueueGateway):
    def __init__(self):
        self.queue = OrderedDict()
        self.completed = {}

    def send_task(
        self,
        task_name: str,
        queue_name: str,
        args: Optional[List[Any]] = None,
        kwargs: Optional[Dict[str, Any]] = None,
        expires: Optional[int] = None,
    ) -> CreateAsyncTaskV1Response:
        task = dict(
            task_name=task_name,
            queue_name=queue_name,
            args=args,
            kwargs=kwargs,
            expires=expires,
        )
        task_id = str(uuid4())[:8]
        self.queue[task_id] = task
        return CreateAsyncTaskV1Response(task_id=task_id)

    def do_task(self, result: Any = 42, status: str = "success"):
        task_id, task = self.queue.popitem(last=False)
        self.completed[task_id] = result

    def get_task_args(self, task_id: str):
        # For testing only
        return self.queue[task_id]

    def get_task(self, task_id: str) -> GetAsyncTaskV1Response:
        result = None
        status_code = None
        if task_id in self.queue:
            status = TaskStatus.PENDING
        elif task_id in self.completed:
            status = TaskStatus.SUCCESS
            result = self.completed[task_id]
            status_code = 200
        else:
            status = TaskStatus.UNDEFINED
        return GetAsyncTaskV1Response(
            task_id=task_id, status=status, result=result, traceback=None, status_code=status_code
        )

    def clear_queue(self, queue_name: str) -> bool:
        queue = {k: v for k, v in self.queue.items() if v["queue_name"] != queue_name}
        self.queue = queue
        return True


class FakeModelEndpointInfraGateway(ModelEndpointInfraGateway):
    db: Dict[str, ModelEndpointInfraState]
    in_flight_infra: Dict[str, ModelEndpointInfraState]
    model_endpoint_record_repository: ModelEndpointRecordRepository

    def __init__(
        self,
        contents: Optional[Dict[str, ModelEndpointInfraState]] = None,
        model_endpoint_record_repository: Optional[ModelEndpointRecordRepository] = None,
    ):
        self.db = contents if contents else {}
        self.in_flight_infra = {}

        # Store the endpoint record repository so that we can update the status of the endpoint
        # upon completion of creating / updating infra resources.
        if model_endpoint_record_repository is None:
            model_endpoint_record_repository = FakeModelEndpointRecordRepository()
        self.model_endpoint_record_repository = model_endpoint_record_repository

    @staticmethod
    def _get_deployment_name(user_id: str, model_endpoint_name: str) -> str:
        return f"{user_id}-{model_endpoint_name}"

    def create_model_endpoint_infra(
        self,
        *,
        model_endpoint_record: ModelEndpointRecord,
        min_workers: int,
        max_workers: int,
        per_worker: int,
        concurrent_requests_per_worker: int,
        cpus: CpuSpecificationType,
        gpus: int,
        memory: StorageSpecificationType,
        gpu_type: Optional[GpuType],
        storage: StorageSpecificationType,
        nodes_per_worker: int,
        optimize_costs: bool,
        aws_role: str,
        results_s3_bucket: str,
        child_fn_info: Optional[Dict[str, Any]],
        post_inference_hooks: Optional[List[str]],
        labels: Dict[str, str],
        prewarm: Optional[bool],
        high_priority: Optional[bool],
        billing_tags: Optional[Dict[str, Any]] = None,
        default_callback_url: Optional[str],
        default_callback_auth: Optional[CallbackAuth],
    ) -> str:
        deployment_name = self._get_deployment_name(
            model_endpoint_record.created_by, model_endpoint_record.name
        )
        in_flight = ModelEndpointInfraState(
            deployment_name=deployment_name,
            aws_role=aws_role,
            results_s3_bucket=results_s3_bucket,
            child_fn_info=child_fn_info,
            labels=labels,
            prewarm=prewarm,
            high_priority=high_priority,
            deployment_state=ModelEndpointDeploymentState(
                min_workers=min_workers,
                max_workers=max_workers,
                per_worker=per_worker,
                concurrent_requests_per_worker=concurrent_requests_per_worker,
            ),
            resource_state=ModelEndpointResourceState(
                cpus=cpus,
                gpus=gpus,
                gpu_type=gpu_type,
                memory=memory,
                storage=storage,
                nodes_per_worker=nodes_per_worker,
                optimize_costs=optimize_costs,
            ),
            user_config_state=ModelEndpointUserConfigState(
                app_config=model_endpoint_record.current_model_bundle.app_config,
                endpoint_config=ModelEndpointConfig(
                    bundle_name=model_endpoint_record.current_model_bundle.name,
                    endpoint_name=model_endpoint_record.name,
                    post_inference_hooks=post_inference_hooks,
                    default_callback_url=default_callback_url,
                    default_callback_auth=default_callback_auth,
                ),
            ),
            image="000000000000.dkr.ecr.us-west-2.amazonaws.com/non-existent-repo:fake-tag",
        )
        self.in_flight_infra[in_flight.deployment_name] = in_flight
        return "test_creation_task_id"

    @staticmethod
    def update_model_endpoint_infra_in_place(
        *,
        model_endpoint_infra: ModelEndpointInfraState,
        **kwargs,
    ):
        endpoint_config = model_endpoint_infra.user_config_state.endpoint_config
        if kwargs["min_workers"] is not None:
            model_endpoint_infra.deployment_state.min_workers = kwargs["min_workers"]
        if kwargs["max_workers"] is not None:
            model_endpoint_infra.deployment_state.max_workers = kwargs["max_workers"]
        if kwargs["per_worker"] is not None:
            model_endpoint_infra.deployment_state.per_worker = kwargs["per_worker"]
        if kwargs["concurrent_requests_per_worker"] is not None:
            model_endpoint_infra.deployment_state.concurrent_requests_per_worker = kwargs[
                "concurrent_requests_per_worker"
            ]
        if kwargs["cpus"] is not None:
            model_endpoint_infra.resource_state.cpus = kwargs["cpus"]
        if kwargs["gpus"] is not None:
            model_endpoint_infra.resource_state.gpus = kwargs["gpus"]
        if kwargs["memory"] is not None:
            model_endpoint_infra.resource_state.memory = kwargs["memory"]
        if kwargs["gpu_type"] is not None:
            model_endpoint_infra.resource_state.gpu_type = kwargs["gpu_type"]
        if kwargs["storage"] is not None:
            model_endpoint_infra.resource_state.storage = kwargs["storage"]
        if kwargs["child_fn_info"] is not None:
            model_endpoint_infra.child_fn_info = kwargs["child_fn_info"]
        if kwargs["post_inference_hooks"] is not None:
            assert endpoint_config is not None
            endpoint_config.post_inference_hooks = kwargs["post_inference_hooks"]
        if kwargs["labels"] is not None:
            model_endpoint_infra.labels = kwargs["labels"]
        if kwargs["prewarm"] is not None:
            model_endpoint_infra.prewarm = kwargs["prewarm"]
        if kwargs["high_priority"] is not None:
            model_endpoint_infra.high_priority = kwargs["high_priority"]
        if kwargs["default_callback_url"] is not None:
            assert endpoint_config is not None
            endpoint_config.default_callback_url = kwargs["default_callback_url"]

    async def update_model_endpoint_infra(
        self,
        *,
        model_endpoint_record: ModelEndpointRecord,
        min_workers: Optional[int] = None,
        max_workers: Optional[int] = None,
        per_worker: Optional[int] = None,
        concurrent_requests_per_worker: Optional[int] = None,
        cpus: Optional[CpuSpecificationType] = None,
        gpus: Optional[int] = None,
        memory: Optional[StorageSpecificationType] = None,
        gpu_type: Optional[GpuType] = None,
        storage: Optional[StorageSpecificationType] = None,
        optimize_costs: Optional[bool] = None,
        child_fn_info: Optional[Dict[str, Any]] = None,
        post_inference_hooks: Optional[List[str]] = None,
        labels: Optional[Dict[str, str]] = None,
        prewarm: Optional[bool] = None,
        high_priority: Optional[bool] = None,
        billing_tags: Optional[Dict[str, Any]] = None,
        default_callback_url: Optional[str] = None,
        default_callback_auth: Optional[CallbackAuth] = None,
    ) -> str:
        model_endpoint_infra = await self.get_model_endpoint_infra(
            model_endpoint_record=model_endpoint_record
        )
        assert model_endpoint_infra is not None
        model_endpoint_infra = model_endpoint_infra.copy()
        self.update_model_endpoint_infra_in_place(**locals())
        self.in_flight_infra[model_endpoint_infra.deployment_name] = model_endpoint_infra
        return "test_creation_task_id"

    async def get_model_endpoint_infra(
        self, model_endpoint_record: ModelEndpointRecord
    ) -> Optional[ModelEndpointInfraState]:
        deployment_name = self._get_deployment_name(
            model_endpoint_record.created_by, model_endpoint_record.name
        )
        return self.db.get(deployment_name)

    async def promote_in_flight_infra(self, owner: str, model_endpoint_name: str):
        deployment_name = self._get_deployment_name(owner, model_endpoint_name)
        self.db[deployment_name] = self.in_flight_infra[deployment_name]

        model_endpoint_records = (
            await self.model_endpoint_record_repository.list_model_endpoint_records(
                owner=owner,
                name=model_endpoint_name,
                order_by=None,
            )
        )
        assert len(model_endpoint_records) == 1
        model_endpoint_records[0].status = ModelEndpointStatus.READY
        del self.in_flight_infra[deployment_name]

    async def delete_model_endpoint_infra(self, model_endpoint_record: ModelEndpointRecord) -> bool:
        deployment_name = self._get_deployment_name(
            model_endpoint_record.created_by, model_endpoint_record.name
        )
        if deployment_name not in self.db:
            return False
        del self.db[deployment_name]
        if deployment_name in self.in_flight_infra:
            del self.in_flight_infra[deployment_name]
        return True

    async def restart_model_endpoint_infra(
        self, model_endpoint_record: ModelEndpointRecord
    ) -> None:
        # Always succeeds
        pass


class FakeEndpointResourceGateway(EndpointResourceGateway[QueueInfo]):
    def __init__(self):
        self.db: Dict[str, ModelEndpointInfraState] = {}  # type: ignore

    def add_resource(self, endpoint_id: str, infra_state: ModelEndpointInfraState):
        infra_state.labels.update({"user_id": "user_id", "endpoint_name": "endpoint_name"})
        self.db[endpoint_id] = infra_state

    async def create_queue(
        self,
        endpoint_record: ModelEndpointRecord,
        labels: Dict[str, str],
    ) -> QueueInfo:
        """Creates a new, unique queue name.
        Used by this endpoint resource gateway to create new resources.
        """
        return QueueInfo(queue_name="foobar", queue_url=None)

    async def create_or_update_resources(
        self, request: CreateOrUpdateResourcesRequest
    ) -> EndpointResourceGatewayCreateOrUpdateResourcesResponse:
        build_endpoint_request = request.build_endpoint_request
        endpoint_id = build_endpoint_request.model_endpoint_record.id
        model_endpoint_record = build_endpoint_request.model_endpoint_record
        q = await self.create_queue(model_endpoint_record, build_endpoint_request.labels)
        infra_state = ModelEndpointInfraState(
            deployment_name=build_endpoint_request.deployment_name,
            aws_role=build_endpoint_request.aws_role,
            results_s3_bucket=build_endpoint_request.results_s3_bucket,
            child_fn_info=build_endpoint_request.child_fn_info,
            post_inference_hooks=build_endpoint_request.post_inference_hooks,
            labels=build_endpoint_request.labels,
            prewarm=build_endpoint_request.prewarm,
            high_priority=build_endpoint_request.high_priority,
            deployment_state=ModelEndpointDeploymentState(
                min_workers=build_endpoint_request.min_workers,
                max_workers=build_endpoint_request.max_workers,
                per_worker=build_endpoint_request.per_worker,
                concurrent_requests_per_worker=build_endpoint_request.concurrent_requests_per_worker,
            ),
            resource_state=ModelEndpointResourceState(
                cpus=build_endpoint_request.cpus,
                gpus=build_endpoint_request.gpus,
                gpu_type=build_endpoint_request.gpu_type,
                memory=build_endpoint_request.memory,
                storage=build_endpoint_request.storage,
                nodes_per_worker=build_endpoint_request.nodes_per_worker,
                optimize_costs=build_endpoint_request.optimize_costs,
            ),
            user_config_state=ModelEndpointUserConfigState(
                app_config=model_endpoint_record.current_model_bundle.app_config,
                endpoint_config=ModelEndpointConfig(
                    bundle_name=model_endpoint_record.current_model_bundle.name,
                    endpoint_name=model_endpoint_record.name,
                    post_inference_hooks=build_endpoint_request.post_inference_hooks,
                ),
            ),
            image=request.image,
        )
        # self.db[build_endpoint_request.deployment_name] = infra_state
        self.db[endpoint_id] = infra_state
        return EndpointResourceGatewayCreateOrUpdateResourcesResponse(destination=q.queue_name)

    async def get_resources(
        self, endpoint_id: str, deployment_name: str, endpoint_type: ModelEndpointType
    ) -> ModelEndpointInfraState:
        if endpoint_id not in self.db:
            raise EndpointResourceInfraException
        return self.db[endpoint_id]

    async def get_all_resources(
        self,
    ) -> Dict[str, Tuple[bool, ModelEndpointInfraState]]:
        result: Dict[str, Tuple[bool, ModelEndpointInfraState]] = {}
        for key, value in self.db.items():
            result[key] = (True, value)
        return result

    async def delete_resources(
        self, endpoint_id: str, deployment_name: str, endpoint_type: ModelEndpointType
    ) -> bool:
        if endpoint_id not in self.db:
            return False
        del self.db[endpoint_id]
        return True

    async def restart_deployment(self, deployment_name: str) -> None:
        # Always succeeds
        pass


class FakeDockerImageBatchJobGateway(DockerImageBatchJobGateway):
    def __init__(self, contents=None):
        self.db: Dict[str, DockerImageBatchJob] = {} if contents is None else contents
        self.id = 0

    async def create_docker_image_batch_job(
        self,
        *,
        created_by: str,
        owner: str,
        job_config: Optional[Dict[str, Any]],
        env: Optional[Dict[str, str]],
        command: List[str],
        repo: str,
        tag: str,
        resource_requests: CreateDockerImageBatchJobResourceRequests,
        labels: Dict[str, str],
        mount_location: Optional[str],
        annotations: Optional[Dict[str, str]] = None,
        override_job_max_runtime_s: Optional[int] = None,
        num_workers: Optional[int] = 1,
    ) -> str:
        job_id = f"ft-{self.id}"
        self.id += 1

        self.db[job_id] = DockerImageBatchJob(
            id=job_id,
            created_by=created_by,
            owner=owner,
            created_at=datetime.now(),
            completed_at=None,
            status=BatchJobStatus.RUNNING,
            annotations=annotations,
            override_job_max_runtime_s=override_job_max_runtime_s,
            num_workers=num_workers,
        )

        return job_id

    async def get_docker_image_batch_job(self, batch_job_id: str) -> Optional[DockerImageBatchJob]:
        return self.db.get(batch_job_id)

    async def list_docker_image_batch_jobs(self, owner: str) -> List[DockerImageBatchJob]:
        return [job for job in self.db.values() if job["owner"] == owner]

    async def update_docker_image_batch_job(self, batch_job_id: str, cancel: bool) -> bool:
        if batch_job_id not in self.db:
            return False

        if cancel:
            del self.db[batch_job_id]

        return cancel


class FakeCronJobGateway(CronJobGateway):
    def __init__(self, contents=None):
        self.db = contents or {}
        self.suspended_cronjobs: Set[str] = set()
        self.id = 0

    async def create_cronjob(
        self,
        *,
        request_host: str,
        trigger_id: str,
        created_by: str,
        owner: str,
        cron_schedule: str,
        docker_image_batch_job_bundle_id: str,
        default_job_config: Optional[Dict[str, Any]],
        default_job_metadata: Dict[str, str],
    ) -> None:
        cron_job_id = f"cronjob-{trigger_id}"
        self.id += 1

        self.db[cron_job_id] = Trigger(
            id=cron_job_id,
            name=cron_job_id,
            owner=owner,
            created_by=created_by,
            created_at=datetime.now(),
            cron_schedule=cron_schedule,
            docker_image_batch_job_bundle_id=docker_image_batch_job_bundle_id,
            default_job_config=default_job_config,
            default_job_metadata=default_job_metadata,
        )

    async def list_jobs(
        self,
        *,
        owner: str,
        trigger_id: Optional[str],
    ) -> List[DockerImageBatchJob]:
        return []

    async def update_cronjob(
        self,
        *,
        trigger_id: str,
        cron_schedule: Optional[str],
        suspend: Optional[bool],
    ) -> None:
        cron_job_id = f"cronjob-{trigger_id}"
        if cron_job_id not in self.db:
            return

        if cron_schedule is not None:
            self.db[cron_job_id].cron_schedule = cron_schedule
        if suspend is not None:
            if suspend:
                self.suspended_cronjobs.add(cron_job_id)
            else:
                self.suspended_cronjobs.discard(cron_job_id)

    async def delete_cronjob(
        self,
        *,
        trigger_id: str,
    ) -> None:
        cron_job_id = f"cronjob-{trigger_id}"
        self.db.pop(cron_job_id, None)
        self.suspended_cronjobs.discard(cron_job_id)


class FakeLLMFineTuningService(LLMFineTuningService):
    def __init__(self, contents=None):
        self.db: Dict[str, DockerImageBatchJob] = {} if contents is None else contents
        self.id = 0

    async def create_fine_tune(
        self,
        created_by: str,
        owner: str,
        model: str,
        training_file: str,
        validation_file: Optional[str],
        fine_tuning_method: str,
        hyperparameters: Dict[str, FineTuneHparamValueType],
        fine_tuned_model: str,
        wandb_config: Optional[Dict[str, Any]],
    ) -> str:
        job_id = f"ft-{self.id}"
        self.id += 1

        now = datetime.now()

        self.db[job_id] = DockerImageBatchJob(
            id=job_id,
            created_by=created_by,
            owner=owner,
            created_at=now,
            completed_at=None,
            status=BatchJobStatus.RUNNING,
            annotations={
                "fine_tuned_model": fine_tuned_model,
            },
        )

        return job_id

    async def get_fine_tune(self, owner: str, fine_tune_id: str) -> Optional[DockerImageBatchJob]:
        di_batch_job = self.db.get(fine_tune_id)
        if di_batch_job is None or di_batch_job.owner != owner:
            return None
        return di_batch_job

    async def list_fine_tunes(self, owner: str) -> List[DockerImageBatchJob]:
        return [job for job in self.db.values() if job.owner == owner]

    async def cancel_fine_tune(self, owner: str, fine_tune_id: str) -> bool:
        if fine_tune_id not in self.db or self.db.get(fine_tune_id).owner != owner:
            return False

        del self.db[fine_tune_id]
        return True

    async def get_fine_tune_model_name_from_id(
        self, owner: str, fine_tune_id: str
    ) -> Optional[str]:
        fine_tune = self.db.get(fine_tune_id, None)
        if fine_tune is not None and fine_tune.owner == owner:
            return fine_tune.annotations["fine_tuned_model"]
        return None


class FakeStreamingModelEndpointInferenceGateway(StreamingModelEndpointInferenceGateway):
    def __init__(self):
        self.responses = [
            SyncEndpointPredictV1Response(
                status=TaskStatus.SUCCESS,
                result=None,
                traceback=None,
                status_code=200,
            )
        ]

    async def streaming_predict(
        self,
        topic: str,
        predict_request: EndpointPredictV1Request,
        manually_resolve_dns: bool = False,
        endpoint_name: Optional[str] = None,
    ) -> AsyncIterable[SyncEndpointPredictV1Response]:
        """
        Runs a prediction request and returns a response.
        """
        for response in self.responses:
            yield response


class FakeSyncModelEndpointInferenceGateway(SyncModelEndpointInferenceGateway):
    def __init__(self, fake_sync_inference_content=None):
        if not fake_sync_inference_content:
            self.response = SyncEndpointPredictV1Response(
                status=TaskStatus.SUCCESS,
                result=None,
                traceback=None,
                status_code=200,
            )
        else:
            self.response = fake_sync_inference_content

    async def predict(
        self,
        topic: str,
        predict_request: EndpointPredictV1Request,
        manually_resolve_dns: bool = False,
        endpoint_name: Optional[str] = None,
    ) -> SyncEndpointPredictV1Response:
        """
        Runs a prediction request and returns a response.
        """
        return self.response


class FakeFileStorageGateway(FileStorageGateway):
    def __init__(self, contents=None):
        self.db: Dict[str, FileMetadata] = {} if contents is None else contents
        self.id = 0
        self.content = "Test content"

    async def get_url_from_id(self, owner: str, file_id: str) -> Optional[str]:
        return "dummy URL"

    async def upload_file(self, owner: str, filename: str, content: bytes) -> str:
        file_id = f"file-{self.id}"
        self.id += 1

        self.db[file_id] = FileMetadata(
            id=file_id,
            filename=f"{file_id}_name",
            size=len(self.content),
            owner=owner,
            updated_at=datetime.now(),
        )

        return file_id

    async def get_file(self, owner: str, file_id: str) -> Optional[FileMetadata]:
        file = self.db.get(file_id)
        if file is None or file.owner != owner:
            return None
        return file

    async def list_files(self, owner: str) -> List[FileMetadata]:
        return [file for file in self.db.values() if file.owner == owner]

    async def delete_file(self, owner: str, file_id: str) -> bool:
        if file_id not in self.db or self.db.get(file_id).owner != owner:
            return False

        del self.db[file_id]
        return True

    async def get_file_content(self, owner: str, file_id: str) -> Optional[str]:
        file = self.db.get(file_id)
        if file is None or file.owner != owner:
            return None
        return self.content


@dataclass
class FakeAsyncTask:
    topic: str
    predict_request: EndpointPredictV1Request
    task_timeout_seconds: int
    task_name: str


class FakeAsyncModelEndpointInferenceGateway(AsyncModelEndpointInferenceGateway):
    def __init__(self):
        self.tasks = []

    def create_task(
        self,
        topic: str,
        predict_request: EndpointPredictV1Request,
        task_timeout_seconds: int,
        *,
        task_name: str = DEFAULT_CELERY_TASK_NAME,
    ) -> CreateAsyncTaskV1Response:
        self.tasks.append(
            FakeAsyncTask(
                topic=topic,
                predict_request=predict_request,
                task_timeout_seconds=task_timeout_seconds,
                task_name=task_name,
            )
        )
        return CreateAsyncTaskV1Response(
            task_id="test_task_id"
        )  # can return distinct task ids if we need

    def get_task(self, task_id: str) -> GetAsyncTaskV1Response:
        return GetAsyncTaskV1Response(
            task_id=task_id,
            status=TaskStatus.SUCCESS,
            result=None,
            traceback=None,
            status_code=200,
        )

    def get_last_request(self):
        #  For validating service inputs are correct
        assert len(self.tasks) > 0, "No async tasks have been created"
        return self.tasks[-1]


class FakeInferenceAutoscalingMetricsGateway(InferenceAutoscalingMetricsGateway):
    async def emit_inference_autoscaling_metric(self, endpoint_id: str):
        pass

    async def emit_prewarm_metric(self, endpoint_id: str):
        pass

    async def create_or_update_resources(self, endpoint_id: str):
        pass

    async def delete_resources(self, endpoint_id: str):
        pass


class FakeStreamingStorageGateway(StreamingStorageGateway):
    def put_record(self, stream_name: str, record: Dict[str, Any]):
        pass


class FakeModelEndpointService(ModelEndpointService):
    db: Dict[str, ModelEndpoint]

    def __init__(
        self,
        contents: Optional[Dict[str, ModelEndpoint]] = None,
        model_bundle_repository: Optional[ModelBundleRepository] = None,
        async_model_endpoint_inference_gateway: Optional[AsyncModelEndpointInferenceGateway] = None,
        streaming_model_endpoint_inference_gateway: Optional[
            StreamingModelEndpointInferenceGateway
        ] = None,
        sync_model_endpoint_inference_gateway: Optional[SyncModelEndpointInferenceGateway] = None,
        inference_autoscaling_metrics_gateway: Optional[InferenceAutoscalingMetricsGateway] = None,
        can_scale_http_endpoint_from_zero_flag: bool = True,
    ):
        if contents:
            self.db = contents
            self.unique_owner_name_versions = set()
            for model_endpoint in self.db.values():
                self.unique_owner_name_versions.add(
                    (model_endpoint.record.owner, model_endpoint.record.name)
                )
        else:
            self.db = {}
            self.unique_owner_name_versions = set()

        if model_bundle_repository is None:
            model_bundle_repository = FakeModelBundleRepository()
        self.model_bundle_repository = model_bundle_repository

        if async_model_endpoint_inference_gateway is None:
            async_model_endpoint_inference_gateway = FakeAsyncModelEndpointInferenceGateway()
        self.async_model_endpoint_inference_gateway = async_model_endpoint_inference_gateway

        if streaming_model_endpoint_inference_gateway is None:
            streaming_model_endpoint_inference_gateway = (
                FakeStreamingModelEndpointInferenceGateway()
            )
        self.streaming_model_endpoint_inference_gateway = streaming_model_endpoint_inference_gateway

        if sync_model_endpoint_inference_gateway is None:
            sync_model_endpoint_inference_gateway = FakeSyncModelEndpointInferenceGateway()
        self.sync_model_endpoint_inference_gateway = sync_model_endpoint_inference_gateway

        if inference_autoscaling_metrics_gateway is None:
            inference_autoscaling_metrics_gateway = FakeInferenceAutoscalingMetricsGateway()
        self.inference_autoscaling_metrics_gateway = inference_autoscaling_metrics_gateway

        self.model_endpoints_schema_gateway = LiveModelEndpointsSchemaGateway(
            filesystem_gateway=FakeFilesystemGateway()
        )

        self.can_scale_http_endpoint_from_zero_flag = can_scale_http_endpoint_from_zero_flag

    def get_async_model_endpoint_inference_gateway(
        self,
    ) -> AsyncModelEndpointInferenceGateway:
        return self.async_model_endpoint_inference_gateway

    def get_streaming_model_endpoint_inference_gateway(
        self,
    ) -> StreamingModelEndpointInferenceGateway:
        return self.streaming_model_endpoint_inference_gateway

    def get_sync_model_endpoint_inference_gateway(
        self,
    ) -> SyncModelEndpointInferenceGateway:
        return self.sync_model_endpoint_inference_gateway

    def get_inference_autoscaling_metrics_gateway(
        self,
    ) -> InferenceAutoscalingMetricsGateway:
        return self.inference_autoscaling_metrics_gateway

    def add_model_endpoint(self, model_endpoint: ModelEndpoint):
        self.db[model_endpoint.record.id] = model_endpoint

    async def create_model_endpoint(
        self,
        *,
        name: str,
        created_by: str,
        model_bundle_id: str,
        endpoint_type: ModelEndpointType,
        metadata: Dict[str, Any],
        post_inference_hooks: Optional[List[str]],
        child_fn_info: Optional[Dict[str, Any]],
        cpus: CpuSpecificationType,
        gpus: int,
        memory: StorageSpecificationType,
        gpu_type: Optional[GpuType],
        storage: StorageSpecificationType,
        nodes_per_worker: int,
        optimize_costs: bool,
        min_workers: int,
        max_workers: int,
        per_worker: int,
        concurrent_requests_per_worker: int,
        labels: Dict[str, str],
        aws_role: str,
        results_s3_bucket: str,
        prewarm: Optional[bool],
        high_priority: Optional[bool],
        billing_tags: Optional[Dict[str, Any]] = None,
        owner: str,
        default_callback_url: Optional[str] = None,
        default_callback_auth: Optional[CallbackAuth] = None,
        public_inference: Optional[bool] = None,
    ) -> ModelEndpointRecord:
        destination = generate_destination(
            user_id=created_by,
            endpoint_name=name,
            endpoint_type=endpoint_type,
        )
        current_model_bundle = await self.model_bundle_repository.get_model_bundle(model_bundle_id)
        assert current_model_bundle is not None
        model_endpoint = ModelEndpoint(
            record=ModelEndpointRecord(
                id=str(uuid4())[:8],
                name=name,
                created_by=created_by,
                created_at=datetime.now(),
                last_updated_at=datetime.now(),
                metadata=metadata,
                creation_task_id="test_creation_task_id",
                endpoint_type=endpoint_type,
                destination=destination,
                status=ModelEndpointStatus.UPDATE_IN_PROGRESS,
                current_model_bundle=current_model_bundle,
                owner=owner,
                public_inference=public_inference,
            ),
            infra_state=ModelEndpointInfraState(
                deployment_name=name,
                aws_role=aws_role,
                results_s3_bucket=results_s3_bucket,
                child_fn_info=child_fn_info,
                post_inference_hooks=post_inference_hooks,
                labels=labels,
                prewarm=prewarm,
                high_priority=high_priority,
                deployment_state=ModelEndpointDeploymentState(
                    min_workers=min_workers,
                    max_workers=max_workers,
                    per_worker=per_worker,
                    concurrent_requests_per_worker=concurrent_requests_per_worker,
                ),
                resource_state=ModelEndpointResourceState(
                    cpus=cpus,
                    gpus=gpus,
                    memory=memory,
                    gpu_type=gpu_type,
                    storage=storage,
                    nodes_per_worker=nodes_per_worker,
                    optimize_costs=optimize_costs,
                ),
                user_config_state=ModelEndpointUserConfigState(
                    app_config=current_model_bundle.app_config,
                    endpoint_config=ModelEndpointConfig(
                        bundle_name=current_model_bundle.name,
                        endpoint_name=name,
                        post_inference_hooks=post_inference_hooks,
                        billing_tags=billing_tags,
                        user_id=created_by,
                        billing_queue="some:arn:for:something",
                        default_callback_url=default_callback_url,
                        default_callback_auth=default_callback_auth,
                    ),
                ),
                image="000000000000.dkr.ecr.us-west-2.amazonaws.com/non-existent-repo:fake-tag",
            ),
        )
        self.db[model_endpoint.record.id] = model_endpoint
        return model_endpoint.record

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
        concurrent_requests_per_worker: Optional[int] = None,
        labels: Optional[Dict[str, str]] = None,
        results_s3_bucket: Optional[str] = None,
        prewarm: Optional[bool] = None,
        high_priority: Optional[bool] = None,
        billing_tags: Optional[Dict[str, Any]] = None,
        default_callback_url: Optional[str] = None,
        default_callback_auth: Optional[CallbackAuth] = None,
        public_inference: Optional[bool] = None,
    ) -> ModelEndpointRecord:
        model_endpoint = await self.get_model_endpoint(model_endpoint_id=model_endpoint_id)
        if model_endpoint is None:
            raise ObjectNotFoundException
        current_model_bundle = None
        if model_bundle_id is not None:
            current_model_bundle = await self.model_bundle_repository.get_model_bundle(
                model_bundle_id
            )
            if current_model_bundle is None:
                raise ObjectNotFoundException
        destination = generate_destination(
            user_id=model_endpoint.record.created_by,
            endpoint_name=model_endpoint.record.name,
            endpoint_type=model_endpoint.record.endpoint_type,
        )
        creation_task_id = "test_creation_task_id"
        status = ModelEndpointStatus.UPDATE_IN_PROGRESS
        FakeModelEndpointRecordRepository.update_model_endpoint_record_in_place(
            model_endpoint_record=model_endpoint.record, **locals()
        )
        assert model_endpoint.infra_state is not None
        FakeModelEndpointInfraGateway.update_model_endpoint_infra_in_place(
            model_endpoint_infra=model_endpoint.infra_state,
            child_fn_info=None,
            **locals(),
        )
        return model_endpoint.record

    async def get_model_endpoint(self, model_endpoint_id: str) -> Optional[ModelEndpoint]:
        return self.db.get(model_endpoint_id)

    async def get_model_endpoints_schema(self, owner: str) -> ModelEndpointsSchema:
        endpoints = await self.list_model_endpoints(owner=owner, name=None, order_by=None)
        records = [endpoint.record for endpoint in endpoints]
        return self.model_endpoints_schema_gateway.get_model_endpoints_schema(
            model_endpoint_records=records
        )

    async def get_model_endpoint_record(
        self, model_endpoint_id: str
    ) -> Optional[ModelEndpointRecord]:
        if model_endpoint_id not in self.db:
            return None
        return self.db[model_endpoint_id].record

    @staticmethod
    def _filter_by_name_owner(
        record: ModelEndpointRecord, owner: Optional[str], name: Optional[str]
    ):
        return (not owner or record.owner == owner) and (not name or record.name == name)

    async def list_model_endpoints(
        self,
        owner: Optional[str],
        name: Optional[str],
        order_by: Optional[ModelEndpointOrderBy],
    ) -> List[ModelEndpoint]:
        def filter_fn(m: ModelEndpoint) -> bool:
            return self._filter_by_name_owner(m.record, owner, name)

        model_endpoints = list(filter(filter_fn, self.db.values()))
        if order_by == ModelEndpointOrderBy.NEWEST:
            model_endpoints.sort(key=lambda x: x.record.created_at, reverse=True)
        elif order_by == ModelEndpointOrderBy.OLDEST:
            model_endpoints.sort(key=lambda x: x.record.created_at, reverse=False)

        return model_endpoints

    async def delete_model_endpoint(self, model_endpoint_id: str) -> None:
        if model_endpoint_id not in self.db:
            raise ObjectNotFoundException
        del self.db[model_endpoint_id]

    async def restart_model_endpoint(self, model_endpoint_id: str) -> None:
        # Always succeeds
        pass

    def set_can_scale_http_endpoint_from_zero_flag(self, flag: bool):
        self.can_scale_http_endpoint_from_zero_flag = flag

    def can_scale_http_endpoint_from_zero(self) -> bool:
        return self.can_scale_http_endpoint_from_zero_flag


class FakeTokenizerRepository(TokenizerRepository):
    def load_tokenizer(self, model_name: str) -> AutoTokenizer:
        return AutoTokenizer.from_pretrained(model_name)


class FakeLLMModelEndpointService(LLMModelEndpointService):
    db: Dict[str, ModelEndpoint]

    def __init__(
        self,
        contents: Optional[Dict[str, ModelEndpoint]] = None,
    ):
        if contents:
            self.db = contents
            self.unique_owner_name_versions = set()
            for model_endpoint in self.db.values():
                self.unique_owner_name_versions.add(
                    (model_endpoint.record.owner, model_endpoint.record.name)
                )
        else:
            self.db = {}
            self.unique_owner_name_versions = set()

    def add_model_endpoint(self, model_endpoint: ModelEndpoint):
        self.db[model_endpoint.record.id] = model_endpoint

    @staticmethod
    def _filter_by_name_owner(
        record: ModelEndpointRecord, owner: Optional[str], name: Optional[str]
    ):
        return (
            (not owner or record.owner == owner or record.public_inference is True)
            and (not name or record.name == name)
            and "_llm" in record.metadata
        )

    async def list_llm_model_endpoints(
        self,
        owner: Optional[str],
        name: Optional[str],
        order_by: Optional[ModelEndpointOrderBy],
    ) -> List[ModelEndpoint]:
        def filter_fn(m: ModelEndpoint) -> bool:
            return self._filter_by_name_owner(m.record, owner, name)

        model_endpoints = list(filter(filter_fn, self.db.values()))
        if order_by == ModelEndpointOrderBy.NEWEST:
            model_endpoints.sort(key=lambda x: x.record.created_at, reverse=True)
        elif order_by == ModelEndpointOrderBy.OLDEST:
            model_endpoints.sort(key=lambda x: x.record.created_at, reverse=False)

        return model_endpoints

    async def get_llm_model_endpoint(self, endpoint_name: str) -> List[ModelEndpoint]:
        def filter_fn(m: ModelEndpoint) -> bool:
            return self._filter_by_name_owner(m.record, None, endpoint_name)

        model_endpoints = list(filter(filter_fn, self.db.values()))
        assert len(model_endpoints) <= 1
        if (len(model_endpoints)) == 0:
            return None
        else:
            return model_endpoints[0]

    async def delete_model_endpoint(self, model_endpoint_id: str) -> None:
        if model_endpoint_id not in self.db:
            raise ObjectNotFoundException
        del self.db[model_endpoint_id]


@pytest.fixture
def fake_model_bundle_repository() -> FakeModelBundleRepository:
    repo = FakeModelBundleRepository({})
    return repo


@pytest.fixture
def fake_docker_repository_image_always_exists() -> FakeDockerRepository:
    repo = FakeDockerRepository(image_always_exists=True, raises_error=False)
    return repo


@pytest.fixture
def fake_docker_repository_image_never_exists() -> FakeDockerRepository:
    repo = FakeDockerRepository(image_always_exists=False, raises_error=False)
    return repo


@pytest.fixture
def fake_docker_repository_image_never_exists_and_builds_dont_work() -> FakeDockerRepository:
    repo = FakeDockerRepository(image_always_exists=False, raises_error=True)
    return repo


@pytest.fixture
def fake_model_endpoint_cache_repository() -> FakeModelEndpointCacheRepository:
    repo = FakeModelEndpointCacheRepository()
    return repo


@pytest.fixture
def fake_feature_flag_repository() -> FakeFeatureFlagRepository:
    repo = FakeFeatureFlagRepository()
    return repo


@pytest.fixture
def fake_batch_job_record_repository() -> FakeBatchJobRecordRepository:
    repo = FakeBatchJobRecordRepository()
    return repo


@pytest.fixture
def fake_model_endpoint_record_repository() -> FakeModelEndpointRecordRepository:
    repo = FakeModelEndpointRecordRepository()
    return repo


@pytest.fixture
def fake_docker_image_batch_job_bundle_repository() -> FakeDockerImageBatchJobBundleRepository:
    repo = FakeDockerImageBatchJobBundleRepository()
    return repo


@pytest.fixture
def fake_llm_fine_tune_repository() -> FakeLLMFineTuneRepository:
    repo = FakeLLMFineTuneRepository()
    return repo


@pytest.fixture
def fake_llm_fine_tuning_events_repository() -> FakeLLMFineTuneEventsRepository:
    repo = FakeLLMFineTuneEventsRepository()
    return repo


def fake_trigger_repository() -> FakeTriggerRepository:
    repo = FakeTriggerRepository()
    return repo


@pytest.fixture
def fake_image_cache_gateway() -> FakeImageCacheGateway:
    gateway = FakeImageCacheGateway()
    return gateway


@pytest.fixture
def fake_model_endpoint_infra_gateway() -> FakeModelEndpointInfraGateway:
    gateway = FakeModelEndpointInfraGateway()
    return gateway


@pytest.fixture
def fake_batch_job_orchestration_gateway() -> FakeBatchJobOrchestrationGateway:
    gateway = FakeBatchJobOrchestrationGateway()
    return gateway


@pytest.fixture
def fake_docker_image_batch_job_gateway() -> FakeDockerImageBatchJobGateway:
    gateway = FakeDockerImageBatchJobGateway()
    return gateway


@pytest.fixture
def fake_llm_batch_completions_service() -> FakeLLMBatchCompletionsService:
    service = FakeLLMBatchCompletionsService()
    return service


@pytest.fixture
def fake_monitoring_metrics_gateway() -> FakeMonitoringMetricsGateway:
    gateway = FakeMonitoringMetricsGateway()
    return gateway


@pytest.fixture
def fake_task_queue_gateway() -> FakeTaskQueueGateway:
    gateway = FakeTaskQueueGateway()
    return gateway


@pytest.fixture
def fake_resource_gateway() -> FakeEndpointResourceGateway:
    gateway = FakeEndpointResourceGateway()
    return gateway


@pytest.fixture
def fake_filesystem_gateway() -> FakeFilesystemGateway:
    gateway = FakeFilesystemGateway()
    return gateway


@pytest.fixture
def fake_notification_gateway() -> FakeNotificationGateway:
    gateway = FakeNotificationGateway()
    return gateway


@pytest.fixture
def fake_model_primitive_gateway() -> FakeModelPrimitiveGateway:
    gateway = FakeModelPrimitiveGateway()
    return gateway


@pytest.fixture
def fake_async_model_endpoint_inference_gateway() -> FakeAsyncModelEndpointInferenceGateway:
    gateway = FakeAsyncModelEndpointInferenceGateway()
    return gateway


@pytest.fixture
def fake_streaming_model_endpoint_inference_gateway() -> FakeStreamingModelEndpointInferenceGateway:
    gateway = FakeStreamingModelEndpointInferenceGateway()
    return gateway


@pytest.fixture
def fake_sync_model_endpoint_inference_gateway() -> FakeSyncModelEndpointInferenceGateway:
    gateway = FakeSyncModelEndpointInferenceGateway()
    return gateway


@pytest.fixture
def fake_inference_autoscaling_metrics_gateway() -> FakeInferenceAutoscalingMetricsGateway:
    gateway = FakeInferenceAutoscalingMetricsGateway()
    return gateway


@pytest.fixture
def fake_file_storage_gateway() -> FakeFileStorageGateway:
    gateway = FakeFileStorageGateway()
    return gateway


@pytest.fixture
def fake_llm_artifact_gateway() -> FakeLLMArtifactGateway:
    gateway = FakeLLMArtifactGateway()
    return gateway


def fake_cron_job_gateway() -> FakeCronJobGateway:
    gateway = FakeCronJobGateway()
    return gateway


@pytest.fixture
def fake_model_endpoint_service() -> FakeModelEndpointService:
    service = FakeModelEndpointService()
    return service


@pytest.fixture
def fake_llm_model_endpoint_service() -> FakeLLMModelEndpointService:
    service = FakeLLMModelEndpointService()
    return service


@pytest.fixture
def fake_llm_fine_tuning_service() -> FakeLLMFineTuningService:
    service = FakeLLMFineTuningService()
    return service


@pytest.fixture
def fake_image_cache_service(
    fake_image_cache_gateway,
    fake_model_endpoint_record_repository,
    fake_docker_repository_image_always_exists,
) -> ImageCacheService:
    return ImageCacheService(
        model_endpoint_record_repository=fake_model_endpoint_record_repository,
        image_cache_gateway=fake_image_cache_gateway,
        docker_repository=fake_docker_repository_image_always_exists,
    )


@pytest.fixture
def fake_tokenizer_repository() -> TokenizerRepository:
    return FakeTokenizerRepository()


@pytest.fixture
def fake_streaming_storage_gateway() -> StreamingStorageGateway:
    gateway = FakeStreamingStorageGateway()
    return gateway


@pytest.fixture
def get_repositories_generator_wrapper():
    def get_repositories_generator(
        fake_docker_repository_image_always_exists: bool,
        fake_model_bundle_repository_contents,
        fake_model_endpoint_record_repository_contents,
        fake_model_endpoint_infra_gateway_contents,
        fake_batch_job_record_repository_contents,
        fake_batch_job_progress_gateway_contents,
        fake_cron_job_gateway_contents,
        fake_docker_image_batch_job_bundle_repository_contents,
        fake_docker_image_batch_job_gateway_contents,
        fake_llm_fine_tuning_service_contents,
        fake_file_storage_gateway_contents,
        fake_trigger_repository_contents,
        fake_file_system_gateway_contents,
        fake_sync_inference_content,
    ):
        def get_test_repositories() -> Iterator[ExternalInterfaces]:
            fake_file_system_gateway = FakeFilesystemGateway()
            fake_model_bundle_repository = FakeModelBundleRepository(
                contents=fake_model_bundle_repository_contents
            )
            fake_monitoring_metrics_gateway = FakeMonitoringMetricsGateway()
            fake_model_endpoint_record_repository = FakeModelEndpointRecordRepository(
                contents=fake_model_endpoint_record_repository_contents,
                model_bundle_repository=fake_model_bundle_repository,
            )
            fake_model_endpoint_infra_gateway = FakeModelEndpointInfraGateway(
                contents=fake_model_endpoint_infra_gateway_contents,
                model_endpoint_record_repository=fake_model_endpoint_record_repository,
            )
            fake_model_endpoint_cache_repository = FakeModelEndpointCacheRepository()
            async_model_endpoint_inference_gateway = FakeAsyncModelEndpointInferenceGateway()
            streaming_model_endpoint_inference_gateway = (
                FakeStreamingModelEndpointInferenceGateway()
            )
            sync_model_endpoint_inference_gateway = FakeSyncModelEndpointInferenceGateway(
                fake_sync_inference_content
            )
            inference_autoscaling_metrics_gateway = FakeInferenceAutoscalingMetricsGateway()
            model_endpoints_schema_gateway = LiveModelEndpointsSchemaGateway(
                filesystem_gateway=FakeFilesystemGateway(),
            )
            fake_model_endpoint_service = LiveModelEndpointService(
                model_endpoint_record_repository=fake_model_endpoint_record_repository,
                model_endpoint_infra_gateway=fake_model_endpoint_infra_gateway,
                model_endpoint_cache_repository=fake_model_endpoint_cache_repository,
                async_model_endpoint_inference_gateway=async_model_endpoint_inference_gateway,
                streaming_model_endpoint_inference_gateway=streaming_model_endpoint_inference_gateway,
                sync_model_endpoint_inference_gateway=sync_model_endpoint_inference_gateway,
                inference_autoscaling_metrics_gateway=inference_autoscaling_metrics_gateway,
                model_endpoints_schema_gateway=model_endpoints_schema_gateway,
                can_scale_http_endpoint_from_zero_flag=True,  # reasonable default, gets overridden in individual tests if needed
            )
            fake_batch_job_service = LiveBatchJobService(
                batch_job_record_repository=FakeBatchJobRecordRepository(
                    contents=fake_batch_job_record_repository_contents,
                    model_bundle_repository=fake_model_bundle_repository,
                ),
                model_endpoint_service=fake_model_endpoint_service,
                batch_job_orchestration_gateway=FakeBatchJobOrchestrationGateway(),
                batch_job_progress_gateway=LiveBatchJobProgressGateway(
                    filesystem_gateway=FakeFilesystemGateway(
                        read_data=fake_batch_job_progress_gateway_contents
                    ),
                ),
            )
            fake_docker_image_batch_job_bundle_repository = FakeDockerImageBatchJobBundleRepository(
                contents=fake_docker_image_batch_job_bundle_repository_contents
            )
            fake_trigger_repository = FakeTriggerRepository(
                contents=fake_trigger_repository_contents
            )
            fake_docker_image_batch_job_gateway = FakeDockerImageBatchJobGateway(
                fake_docker_image_batch_job_gateway_contents
            )
            fake_llm_artifact_gateway = FakeLLMArtifactGateway()
            fake_cron_job_gateway = FakeCronJobGateway(fake_cron_job_gateway_contents)
            fake_llm_model_endpoint_service = LiveLLMModelEndpointService(
                model_endpoint_record_repository=fake_model_endpoint_record_repository,
                model_endpoint_service=fake_model_endpoint_service,
            )
            fake_llm_batch_completions_service = LiveLLMBatchCompletionsService(
                docker_image_batch_job_gateway=fake_docker_image_batch_job_gateway
            )
            fake_llm_fine_tuning_service = FakeLLMFineTuningService(
                fake_llm_fine_tuning_service_contents
            )
            fake_llm_fine_tuning_events_repository = FakeLLMFineTuneEventsRepository()
            fake_file_storage_gateway = FakeFileStorageGateway(fake_file_storage_gateway_contents)
            fake_tokenizer_repository = FakeTokenizerRepository()
            fake_streaming_storage_gateway = FakeStreamingStorageGateway()

            repositories = ExternalInterfaces(
                docker_repository=FakeDockerRepository(
                    fake_docker_repository_image_always_exists, False
                ),
                model_bundle_repository=fake_model_bundle_repository,
                model_endpoint_service=fake_model_endpoint_service,
                llm_model_endpoint_service=fake_llm_model_endpoint_service,
                llm_batch_completions_service=fake_llm_batch_completions_service,
                batch_job_service=fake_batch_job_service,
                resource_gateway=FakeEndpointResourceGateway(),
                endpoint_creation_task_queue_gateway=FakeTaskQueueGateway(),
                inference_task_queue_gateway=FakeTaskQueueGateway(),
                model_endpoint_infra_gateway=fake_model_endpoint_infra_gateway,
                model_primitive_gateway=FakeModelPrimitiveGateway(),
                docker_image_batch_job_bundle_repository=fake_docker_image_batch_job_bundle_repository,
                docker_image_batch_job_gateway=fake_docker_image_batch_job_gateway,
                llm_fine_tuning_service=fake_llm_fine_tuning_service,
                llm_fine_tune_events_repository=fake_llm_fine_tuning_events_repository,
                file_storage_gateway=fake_file_storage_gateway,
                trigger_repository=fake_trigger_repository,
                cron_job_gateway=fake_cron_job_gateway,
                filesystem_gateway=fake_file_system_gateway,
                llm_artifact_gateway=fake_llm_artifact_gateway,
                monitoring_metrics_gateway=fake_monitoring_metrics_gateway,
                tokenizer_repository=fake_tokenizer_repository,
                streaming_storage_gateway=fake_streaming_storage_gateway,
                tracing_gateway=LiveTracingGateway(),
            )
            try:
                yield repositories
            finally:
                pass

        return get_test_repositories

    return get_repositories_generator


@pytest.fixture
def test_api_key() -> str:
    # On team test_user_id
    return "test_user_id"


@pytest.fixture
def test_api_key_2() -> str:
    # On team test_user_id
    return "test_user_id_2"


@pytest.fixture
def test_api_key_user_on_other_team() -> str:
    # On team test_team
    return "test_user_id_on_other_team"


@pytest.fixture
def test_api_key_user_on_other_team_2() -> str:
    # On team test_team
    return "test_user_id_on_other_team_2"


@pytest.fixture
def test_api_key_team() -> str:
    # See fixture test_api_key_user_on_other_team
    return "test_team"


@pytest.fixture
def model_bundle_1(test_api_key: str) -> ModelBundle:
    model_bundle = ModelBundle(
        id="test_model_bundle_id_1",
        name="test_model_bundle_name_1",
        created_by=test_api_key,
        owner=test_api_key,
        created_at=datetime(2022, 1, 1),
        model_artifact_ids=["test_model_artifact_id"],
        metadata={},
        flavor=CloudpickleArtifactFlavor(
            flavor="cloudpickle_artifact",
            framework=PytorchFramework(
                framework_type="pytorch",
                pytorch_image_tag="test_tag",
            ),
            requirements=["numpy==0.0.0"],
            location="test_location",
            app_config=None,
            load_predict_fn="test_load_predict_fn",
            load_model_fn="test_load_model_fn",
        ),
        # LEGACY FIELDS
        location="test_location",
        requirements=["numpy==0.0.0"],
        env_params=ModelBundleEnvironmentParams(
            framework_type=ModelBundleFrameworkType.PYTORCH,
            pytorch_image_tag="test_tag",
        ),
        packaging_type=ModelBundlePackagingType.CLOUDPICKLE,
        app_config=None,
    )
    return model_bundle


@pytest.fixture
def model_bundle_2(test_api_key: str) -> ModelBundle:
    model_bundle = ModelBundle(
        id="test_model_bundle_id_2",
        name="test_model_bundle_name_2",
        created_by=test_api_key,
        owner=test_api_key,
        created_at=datetime(2022, 1, 2),
        model_artifact_ids=["test_model_artifact_id"],
        metadata={},
        flavor=CloudpickleArtifactFlavor(
            flavor="cloudpickle_artifact",
            framework=TensorflowFramework(
                framework_type="tensorflow",
                tensorflow_version="0.0.0",
            ),
            requirements=["numpy==0.0.0"],
            location="test_location",
            app_config=None,
            load_predict_fn="test_load_predict_fn",
            load_model_fn="test_load_model_fn",
        ),
        # LEGACY FIELDS
        location="test_location",
        requirements=["numpy==0.0.0"],
        env_params=ModelBundleEnvironmentParams(
            framework_type=ModelBundleFrameworkType.TENSORFLOW,
            tensorflow_version="0.0.0",
        ),
        packaging_type=ModelBundlePackagingType.CLOUDPICKLE,
        app_config=None,
    )
    return model_bundle


@pytest.fixture
def model_bundle_3(test_api_key: str) -> ModelBundle:
    model_bundle = ModelBundle(
        id="test_model_bundle_id_3",
        name="test_model_bundle_name_3",
        created_by=test_api_key,
        owner=test_api_key,
        created_at=datetime(2022, 1, 2),
        model_artifact_ids=["test_model_artifact_id"],
        metadata={
            "load_predict_fn_module_path": "test_load_predict_fn_module_path",
            "load_model_fn_module_path": "test_load_model_fn_module_path",
        },
        flavor=ZipArtifactFlavor(
            flavor="zip_artifact",
            framework=CustomFramework(
                framework_type=ModelBundleFrameworkType.CUSTOM,
                image_repository="test_repo",
                image_tag="test_tag",
            ),
            requirements=["numpy==0.0.0"],
            location="test_location",
            app_config=None,
            load_predict_fn_module_path="test_load_predict_fn_module_path",
            load_model_fn_module_path="test_load_model_fn_module_path",
        ),
        # LEGACY FIELDS
        location="test_location",
        requirements=["numpy==0.0.0"],
        env_params=ModelBundleEnvironmentParams(
            framework_type=ModelBundleFrameworkType.CUSTOM,
            ecr_repo="test_repo",
            image_tag="test_tag",
        ),
        packaging_type=ModelBundlePackagingType.ZIP,
        app_config=None,
    )
    return model_bundle


@pytest.fixture
def model_bundle_4(test_api_key: str) -> ModelBundle:
    model_bundle = ModelBundle(
        id="test_model_bundle_id_4",
        name="test_model_bundle_name_4",
        created_by=test_api_key,
        owner=test_api_key,
        created_at=datetime(2022, 1, 3),
        model_artifact_ids=["test_model_artifact_id_4"],
        metadata={
            "test_key_2": "test_value_2",
        },
        flavor=RunnableImageFlavor(
            flavor="runnable_image",
            repository="test_repo",
            tag="test_tag",
            command=["test_command"],
            predict_route="/test_predict_route",
            healthcheck_route="/test_healthcheck_route",
            env={"test_key": "test_value"},
            protocol="http",
            readiness_initial_delay_seconds=30,
        ),
        # LEGACY FIELDS
        location="test_location",
        requirements=["numpy==0.0.0"],
        env_params=ModelBundleEnvironmentParams(
            framework_type=ModelBundleFrameworkType.CUSTOM,
            ecr_repo="test_repo",
            image_tag="test_tag",
        ),
        packaging_type=ModelBundlePackagingType.LIRA,
        app_config=None,
    )
    return model_bundle


@pytest.fixture
def model_bundle_5(test_api_key: str) -> ModelBundle:
    model_bundle = ModelBundle(
        id="test_model_bundle_id_5",
        name="test_model_bundle_name_5",
        created_by=test_api_key,
        owner=test_api_key,
        created_at=datetime(2022, 1, 3),
        model_artifact_ids=["test_model_artifact_id_5"],
        metadata={
            "test_key_2": "test_value_2",
        },
        flavor=StreamingEnhancedRunnableImageFlavor(
            flavor="streaming_enhanced_runnable_image",
            repository="test_repo",
            tag="test_tag",
            command=["test_command"],
            env={"test_key": "test_value"},
            protocol="http",
            readiness_initial_delay_seconds=30,
            streaming_command=["test_streaming_command"],
            streaming_predict_route="/test_streaming_predict_route",
        ),
        # LEGACY FIELDS
        location="test_location",
        requirements=["numpy==0.0.0"],
        env_params=ModelBundleEnvironmentParams(
            framework_type=ModelBundleFrameworkType.CUSTOM,
            ecr_repo="test_repo",
            image_tag="test_tag",
        ),
        packaging_type=ModelBundlePackagingType.LIRA,
        app_config=None,
    )
    return model_bundle


@pytest.fixture
def model_bundle_6(test_api_key: str) -> ModelBundle:
    model_bundle = ModelBundle(
        id="test_model_bundle_id_6",
        name="test_model_bundle_name_6",
        created_by=test_api_key,
        owner=test_api_key,
        created_at=datetime(2022, 1, 3),
        model_artifact_ids=["test_model_artifact_id_6"],
        metadata={
            "test_key_2": "test_value_2",
        },
        flavor=TritonEnhancedRunnableImageFlavor(
            flavor="triton_enhanced_runnable_image",
            repository="test_repo",
            tag="test_tag",
            command=["test_command"],
            env={"test_key": "test_value"},
            protocol="http",
            readiness_initial_delay_seconds=30,
            triton_model_repository="test_triton_model_repository",
            triton_model_replicas=None,
            triton_num_cpu=1,
            triton_commit_tag="test_commit_tag",
            triton_storage="1G",
            triton_memory="1G",
        ),
        # LEGACY FIELDS
        location="test_location",
        requirements=["numpy==0.0.0"],
        env_params=ModelBundleEnvironmentParams(
            framework_type=ModelBundleFrameworkType.CUSTOM,
            ecr_repo="test_repo",
            image_tag="test_tag",
        ),
        packaging_type=ModelBundlePackagingType.LIRA,
        app_config=None,
    )
    return model_bundle


@pytest.fixture
def model_bundle_triton_enhanced_runnable_image_0_cpu_None_memory_storage(
    test_api_key: str,
) -> ModelBundle:
    model_bundle = ModelBundle(
        id="test_model_bundle_id_triton_enhanced_runnable_image_0_cpu_None_memory_storage",
        name="test_model_bundle_name_triton_enhanced_runnable_image_0_cpu_None_memory_storage",
        created_by=test_api_key,
        owner=test_api_key,
        created_at=datetime(2022, 1, 3),
        model_artifact_ids=[
            "test_model_artifact_id_triton_enhanced_runnable_image_0_cpu_None_memory_storage"
        ],
        metadata={
            "test_key_2": "test_value_2",
        },
        flavor=TritonEnhancedRunnableImageFlavor(
            flavor="triton_enhanced_runnable_image",
            repository="test_repo",
            tag="test_tag",
            command=["test_command"],
            env={"test_key": "test_value"},
            protocol="http",
            readiness_initial_delay_seconds=30,
            triton_model_repository="test_triton_model_repository",
            triton_model_replicas=None,
            triton_num_cpu=0,
            triton_commit_tag="test_commit_tag",
            triton_storage=None,
            triton_memory=None,
        ),
        # LEGACY FIELDS
        location="test_location",
        requirements=["numpy==0.0.0"],
        env_params=ModelBundleEnvironmentParams(
            framework_type=ModelBundleFrameworkType.CUSTOM,
            ecr_repo="test_repo",
            image_tag="test_tag",
        ),
        packaging_type=ModelBundlePackagingType.LIRA,
        app_config=None,
    )
    return model_bundle


@pytest.fixture
def model_endpoint_1(test_api_key: str, model_bundle_1: ModelBundle) -> ModelEndpoint:
    model_endpoint = ModelEndpoint(
        record=ModelEndpointRecord(
            id="test_model_endpoint_id_1",
            name="test_model_endpoint_name_1",
            created_by=test_api_key,
            created_at=datetime(2022, 1, 3),
            last_updated_at=datetime(2022, 1, 3),
            metadata={},
            creation_task_id="test_creation_task_id",
            endpoint_type=ModelEndpointType.ASYNC,
            destination="test_destination",
            status=ModelEndpointStatus.READY,
            current_model_bundle=model_bundle_1,
            owner=test_api_key,
        ),
        infra_state=ModelEndpointInfraState(
            deployment_name=f"{test_api_key}-test_model_endpoint_name_1",
            aws_role="default",
            results_s3_bucket="test_s3_bucket",
            child_fn_info=None,
            labels=dict(team="test_team", product="test_product"),
            prewarm=True,
            high_priority=False,
            deployment_state=ModelEndpointDeploymentState(
                min_workers=1,
                max_workers=3,
                per_worker=2,
                concurrent_requests_per_worker=1,
                available_workers=0,
                unavailable_workers=2,
            ),
            resource_state=ModelEndpointResourceState(
                cpus=1,
                gpus=1,
                memory="1G",
                gpu_type=GpuType.NVIDIA_TESLA_T4,
                storage="10G",
                nodes_per_worker=1,
                optimize_costs=True,
            ),
            user_config_state=ModelEndpointUserConfigState(
                app_config=model_bundle_1.app_config,
                endpoint_config=ModelEndpointConfig(
                    bundle_name=model_bundle_1.name,
                    endpoint_name="test_model_endpoint_name_1",
                    post_inference_hooks=None,
                    billing_tags={
                        "idempotencyKeyPrefix": "value1",
                        "product": "value2",
                        "type": "hi",
                        "subType": "hi",
                        "tags": {"nested_tag_1": "nested_value_1"},
                        "payee": "hi",
                        "payor": "hi",
                        "reference": {"referenceType": "hi", "referenceId": "hi"},
                    },
                ),
            ),
            image="000000000000.dkr.ecr.us-west-2.amazonaws.com/non-existent-repo:fake-tag",
        ),
    )
    return model_endpoint


@pytest.fixture
def model_endpoint_2(test_api_key: str, model_bundle_1: ModelBundle) -> ModelEndpoint:
    model_endpoint = ModelEndpoint(
        record=ModelEndpointRecord(
            id="test_model_endpoint_id_2",
            name="test_model_endpoint_name_2",
            created_by=test_api_key,
            created_at=datetime(2022, 1, 4),
            last_updated_at=datetime(2022, 1, 4),
            metadata={},
            creation_task_id="test_creation_task_id",
            endpoint_type=ModelEndpointType.SYNC,
            destination="test_destination",
            status=ModelEndpointStatus.READY,
            current_model_bundle=model_bundle_1,
            owner=test_api_key,
        ),
        infra_state=ModelEndpointInfraState(
            deployment_name=f"{test_api_key}-test_model_endpoint_name_2",
            aws_role="default",
            results_s3_bucket="test_s3_bucket",
            child_fn_info=None,
            post_inference_hooks=None,
            labels=dict(team="test_team", product="test_product"),
            prewarm=False,
            high_priority=True,
            deployment_state=ModelEndpointDeploymentState(
                min_workers=1,
                max_workers=3,
                per_worker=2,
                concurrent_requests_per_worker=1,
                available_workers=0,
                unavailable_workers=1,
            ),
            resource_state=ModelEndpointResourceState(
                cpus=1,
                gpus=1,
                memory="1G",
                gpu_type=GpuType.NVIDIA_TESLA_T4,
                storage="10G",
                nodes_per_worker=1,
                optimize_costs=False,
            ),
            user_config_state=ModelEndpointUserConfigState(
                app_config=model_bundle_1.app_config,
                endpoint_config=ModelEndpointConfig(
                    bundle_name=model_bundle_1.name,
                    endpoint_name="test_model_endpoint_name_2",
                    post_inference_hooks=None,
                ),
            ),
            image="000000000000.dkr.ecr.us-west-2.amazonaws.com/my-repo:abcdefg222",
        ),
    )
    return model_endpoint


@pytest.fixture
def model_endpoint_3(test_api_key: str, model_bundle_1: ModelBundle) -> ModelEndpoint:
    model_endpoint = ModelEndpoint(
        record=ModelEndpointRecord(
            id="test_model_endpoint_id_3",
            name="test_model_endpoint_name_3",
            created_by=test_api_key,
            created_at=datetime(2022, 1, 4),
            last_updated_at=datetime.min,
            metadata={},
            creation_task_id="test_creation_task_id",
            endpoint_type=ModelEndpointType.SYNC,
            destination="test_destination",
            status=ModelEndpointStatus.READY,
            current_model_bundle=model_bundle_1,
            owner=test_api_key,
        ),
        infra_state=ModelEndpointInfraState(
            deployment_name=f"{test_api_key}-test_model_endpoint_name_3",
            aws_role="default",
            results_s3_bucket="test_s3_bucket",
            child_fn_info=None,
            post_inference_hooks=None,
            labels=dict(team="test_team", product="test_product"),
            prewarm=False,
            high_priority=True,
            deployment_state=ModelEndpointDeploymentState(
                min_workers=1,
                max_workers=3,
                per_worker=2,
                concurrent_requests_per_worker=1,
                available_workers=0,
                unavailable_workers=1,
            ),
            resource_state=ModelEndpointResourceState(
                cpus=1,
                gpus=1,
                memory="1G",
                gpu_type=GpuType.NVIDIA_TESLA_T4,
                storage="10G",
                nodes_per_worker=1,
                optimize_costs=False,
            ),
            user_config_state=ModelEndpointUserConfigState(
                app_config=model_bundle_1.app_config,
                endpoint_config=ModelEndpointConfig(
                    bundle_name=model_bundle_1.name,
                    endpoint_name="test_model_endpoint_name_3",
                    post_inference_hooks=None,
                ),
            ),
            image="000000000000.dkr.ecr.us-west-2.amazonaws.com/my-repo:abcdefg111111111",
        ),
    )
    return model_endpoint


@pytest.fixture
def model_endpoint_4(test_api_key: str, model_bundle_1: ModelBundle) -> ModelEndpoint:
    model_endpoint = ModelEndpoint(
        record=ModelEndpointRecord(
            id="test_model_endpoint_id_4",
            name="test_model_endpoint_name_4",
            created_by=test_api_key,
            created_at=datetime(2022, 1, 4),
            last_updated_at=datetime(2022, 1, 4),
            metadata={},
            creation_task_id="test_creation_task_id",
            endpoint_type=ModelEndpointType.SYNC,
            destination="test_destination",
            status=ModelEndpointStatus.READY,
            current_model_bundle=model_bundle_1,
            owner=test_api_key,
        ),
        infra_state=ModelEndpointInfraState(
            deployment_name=f"{test_api_key}-test_model_endpoint_name_4",
            aws_role="default",
            results_s3_bucket="test_s3_bucket",
            child_fn_info=None,
            post_inference_hooks=None,
            labels=dict(team="test_team", product="test_product"),
            prewarm=False,
            high_priority=True,
            deployment_state=ModelEndpointDeploymentState(
                min_workers=1,
                max_workers=3,
                per_worker=2,
                concurrent_requests_per_worker=1,
                available_workers=1,
                unavailable_workers=1,
            ),
            resource_state=ModelEndpointResourceState(
                cpus=1,
                gpus=1,
                memory="1G",
                gpu_type=GpuType.NVIDIA_TESLA_T4,
                storage="10G",
                nodes_per_worker=1,
                optimize_costs=False,
            ),
            user_config_state=ModelEndpointUserConfigState(
                app_config=model_bundle_1.app_config,
                endpoint_config=ModelEndpointConfig(
                    bundle_name=model_bundle_1.name,
                    endpoint_name="test_model_endpoint_name_4",
                    post_inference_hooks=None,
                ),
            ),
            image="000000000000.dkr.ecr.us-west-2.amazonaws.com/my-repo:abcdefg00000",
        ),
    )
    return model_endpoint


@pytest.fixture
def model_endpoint_public(test_api_key: str, model_bundle_1: ModelBundle) -> ModelEndpoint:
    model_endpoint = ModelEndpoint(
        record=ModelEndpointRecord(
            id="test_model_endpoint_id_1",
            name="test_model_endpoint_name_1",
            created_by=test_api_key,
            created_at=datetime(2022, 1, 3),
            last_updated_at=datetime(2022, 1, 3),
            metadata={},
            creation_task_id="test_creation_task_id",
            endpoint_type=ModelEndpointType.ASYNC,
            destination="test_destination",
            status=ModelEndpointStatus.READY,
            current_model_bundle=model_bundle_1,
            owner=test_api_key,
            public_inference=True,
        ),
        infra_state=ModelEndpointInfraState(
            deployment_name=f"{test_api_key}-test_model_endpoint_name_1",
            aws_role="default",
            results_s3_bucket="test_s3_bucket",
            child_fn_info=None,
            labels=dict(team="test_team", product="test_product"),
            prewarm=True,
            high_priority=False,
            deployment_state=ModelEndpointDeploymentState(
                min_workers=1,
                max_workers=3,
                per_worker=2,
                concurrent_requests_per_worker=1,
                available_workers=0,
                unavailable_workers=2,
            ),
            resource_state=ModelEndpointResourceState(
                cpus=1,
                gpus=1,
                memory="1G",
                gpu_type=GpuType.NVIDIA_TESLA_T4,
                storage="10G",
                nodes_per_worker=1,
                optimize_costs=True,
            ),
            user_config_state=ModelEndpointUserConfigState(
                app_config=model_bundle_1.app_config,
                endpoint_config=ModelEndpointConfig(
                    bundle_name=model_bundle_1.name,
                    endpoint_name="test_model_endpoint_name_1",
                    post_inference_hooks=None,
                    billing_tags={
                        "idempotencyKeyPrefix": "value1",
                        "product": "value2",
                        "type": "hi",
                        "subType": "hi",
                        "tags": {"nested_tag_1": "nested_value_1"},
                        "payee": "hi",
                        "payor": "hi",
                        "reference": {"referenceType": "hi", "referenceId": "hi"},
                    },
                ),
            ),
            image="000000000000.dkr.ecr.us-west-2.amazonaws.com/non-existent-repo:fake-tag",
        ),
    )
    return model_endpoint


@pytest.fixture
def model_endpoint_public_sync(test_api_key: str, model_bundle_1: ModelBundle) -> ModelEndpoint:
    model_endpoint = ModelEndpoint(
        record=ModelEndpointRecord(
            id="test_model_endpoint_id_1",
            name="test_model_endpoint_name_1",
            created_by=test_api_key,
            created_at=datetime(2022, 1, 3),
            last_updated_at=datetime(2022, 1, 3),
            metadata={},
            creation_task_id="test_creation_task_id",
            endpoint_type=ModelEndpointType.SYNC,
            destination="test_destination",
            status=ModelEndpointStatus.READY,
            current_model_bundle=model_bundle_1,
            owner=test_api_key,
            public_inference=True,
        ),
        infra_state=ModelEndpointInfraState(
            deployment_name=f"{test_api_key}-test_model_endpoint_name_1",
            aws_role="default",
            results_s3_bucket="test_s3_bucket",
            child_fn_info=None,
            labels=dict(team="test_team", product="test_product"),
            prewarm=True,
            high_priority=False,
            deployment_state=ModelEndpointDeploymentState(
                min_workers=1,
                max_workers=3,
                per_worker=2,
                concurrent_requests_per_worker=1,
                available_workers=0,
                unavailable_workers=2,
            ),
            resource_state=ModelEndpointResourceState(
                cpus=1,
                gpus=1,
                memory="1G",
                gpu_type=GpuType.NVIDIA_TESLA_T4,
                storage="10G",
                nodes_per_worker=1,
                optimize_costs=True,
            ),
            user_config_state=ModelEndpointUserConfigState(
                app_config=model_bundle_1.app_config,
                endpoint_config=ModelEndpointConfig(
                    bundle_name=model_bundle_1.name,
                    endpoint_name="test_model_endpoint_name_1",
                    post_inference_hooks=None,
                    billing_tags={
                        "idempotencyKeyPrefix": "value1",
                        "product": "value2",
                        "type": "hi",
                        "subType": "hi",
                        "tags": {"nested_tag_1": "nested_value_1"},
                        "payee": "hi",
                        "payor": "hi",
                        "reference": {"referenceType": "hi", "referenceId": "hi"},
                    },
                ),
            ),
            image="000000000000.dkr.ecr.us-west-2.amazonaws.com/non-existent-repo:fake-tag",
        ),
    )
    return model_endpoint


@pytest.fixture
def model_endpoint_runnable(test_api_key: str, model_bundle_4: ModelBundle) -> ModelEndpoint:
    # model_bundle_4 is a runnable bundle
    model_endpoint = ModelEndpoint(
        record=ModelEndpointRecord(
            id="test_model_endpoint_id_runnable",
            name="test_model_endpoint_name_runnable",
            created_by=test_api_key,
            created_at=datetime(2022, 1, 4),
            last_updated_at=datetime(2022, 1, 4),
            metadata={},
            creation_task_id="test_creation_task_id",
            endpoint_type=ModelEndpointType.SYNC,
            destination="test_destination",
            status=ModelEndpointStatus.READY,
            current_model_bundle=model_bundle_4,
            owner=test_api_key,
        ),
        infra_state=ModelEndpointInfraState(
            deployment_name=f"{test_api_key}-test_model_endpoint_name_runnable",
            aws_role="default",
            results_s3_bucket="test_s3_bucket",
            child_fn_info=None,
            post_inference_hooks=None,
            labels=dict(team="test_team", product="test_product"),
            prewarm=False,
            high_priority=True,
            deployment_state=ModelEndpointDeploymentState(
                min_workers=1,
                max_workers=3,
                per_worker=2,
                concurrent_requests_per_worker=1,
                available_workers=1,
                unavailable_workers=1,
            ),
            resource_state=ModelEndpointResourceState(
                cpus=1,
                gpus=1,
                memory="1G",
                gpu_type=GpuType.NVIDIA_TESLA_T4,
                storage="10G",
                nodes_per_worker=1,
                optimize_costs=False,
            ),
            user_config_state=ModelEndpointUserConfigState(
                app_config=model_bundle_4.app_config,
                endpoint_config=ModelEndpointConfig(
                    bundle_name=model_bundle_4.name,
                    endpoint_name="test_model_endpoint_name_runnable",
                    post_inference_hooks=None,
                ),
            ),
            image="000000000000.dkr.ecr.us-west-2.amazonaws.com/hello:there",
        ),
    )
    return model_endpoint


@pytest.fixture
def model_endpoint_streaming(test_api_key: str, model_bundle_5: ModelBundle) -> ModelEndpoint:
    # model_bundle_5 is a runnable bundle
    model_endpoint = ModelEndpoint(
        record=ModelEndpointRecord(
            id="test_model_endpoint_id_streaming",
            name="test_model_endpoint_name_streaming",
            created_by=test_api_key,
            created_at=datetime(2022, 1, 4),
            last_updated_at=datetime(2022, 1, 4),
            metadata={},
            creation_task_id="test_creation_task_id",
            endpoint_type=ModelEndpointType.STREAMING,
            destination="test_destination",
            status=ModelEndpointStatus.READY,
            current_model_bundle=model_bundle_5,
            owner=test_api_key,
        ),
        infra_state=ModelEndpointInfraState(
            deployment_name=f"{test_api_key}-test_model_endpoint_name_streaming",
            aws_role="default",
            results_s3_bucket="test_s3_bucket",
            child_fn_info=None,
            post_inference_hooks=None,
            labels=dict(team="test_team", product="test_product"),
            prewarm=False,
            high_priority=True,
            deployment_state=ModelEndpointDeploymentState(
                min_workers=1,
                max_workers=3,
                per_worker=2,
                concurrent_requests_per_worker=1,
                available_workers=1,
                unavailable_workers=1,
            ),
            resource_state=ModelEndpointResourceState(
                cpus=1,
                gpus=1,
                memory="1G",
                gpu_type=GpuType.NVIDIA_TESLA_T4,
                storage="10G",
                nodes_per_worker=1,
                optimize_costs=False,
            ),
            user_config_state=ModelEndpointUserConfigState(
                app_config=model_bundle_5.app_config,
                endpoint_config=ModelEndpointConfig(
                    bundle_name=model_bundle_5.name,
                    endpoint_name="test_model_endpoint_name_streaming",
                    post_inference_hooks=None,
                ),
            ),
            image="000000000000.dkr.ecr.us-west-2.amazonaws.com/hello:there",
        ),
    )
    return model_endpoint


@pytest.fixture
def model_endpoint_multinode(test_api_key: str, model_bundle_1: ModelBundle) -> ModelEndpoint:
    model_endpoint = ModelEndpoint(
        record=ModelEndpointRecord(
            id="test_model_endpoint_id_multinode",
            name="test_model_endpoint_name_multinode",
            created_by=test_api_key,
            created_at=datetime(2022, 1, 3),
            last_updated_at=datetime(2022, 1, 3),
            metadata={},
            creation_task_id="test_creation_task_id",
            endpoint_type=ModelEndpointType.ASYNC,
            destination="test_destination",
            status=ModelEndpointStatus.READY,
            current_model_bundle=model_bundle_1,
            owner=test_api_key,
        ),
        infra_state=ModelEndpointInfraState(
            deployment_name=f"{test_api_key}-test_model_endpoint_name_multinode",
            aws_role="default",
            results_s3_bucket="test_s3_bucket",
            child_fn_info=None,
            labels=dict(team="test_team", product="test_product"),
            prewarm=True,
            high_priority=False,
            deployment_state=ModelEndpointDeploymentState(
                min_workers=1,
                max_workers=3,
                per_worker=2,
                concurrent_requests_per_worker=1,
                available_workers=0,
                unavailable_workers=2,
            ),
            resource_state=ModelEndpointResourceState(
                cpus=1,
                gpus=1,
                memory="1G",
                gpu_type=GpuType.NVIDIA_TESLA_T4,
                storage="10G",
                nodes_per_worker=2,
                optimize_costs=True,
            ),
            user_config_state=ModelEndpointUserConfigState(
                app_config=model_bundle_1.app_config,
                endpoint_config=ModelEndpointConfig(
                    bundle_name=model_bundle_1.name,
                    endpoint_name="test_model_endpoint_name_multinode",
                    post_inference_hooks=None,
                    billing_tags={
                        "idempotencyKeyPrefix": "value1",
                        "product": "value2",
                        "type": "hi",
                        "subType": "hi",
                        "tags": {"nested_tag_1": "nested_value_1"},
                        "payee": "hi",
                        "payor": "hi",
                        "reference": {"referenceType": "hi", "referenceId": "hi"},
                    },
                ),
            ),
            image="000000000000.dkr.ecr.us-west-2.amazonaws.com/non-existent-repo:fake-tag",
        ),
    )
    return model_endpoint


@pytest.fixture
def batch_job_1(model_bundle_1: ModelBundle, model_endpoint_1: ModelEndpoint) -> BatchJob:
    batch_job = BatchJob(
        record=BatchJobRecord(
            id="test_batch_job_id_1",
            created_at=datetime(2022, 1, 5),
            status=BatchJobStatus.PENDING,
            created_by=model_bundle_1.created_by,
            owner=model_bundle_1.owner,
            model_bundle=model_bundle_1,
            model_endpoint_id=model_endpoint_1.record.id,
            task_ids_location=None,
            result_location=None,
        ),
        model_endpoint=model_endpoint_1,
        progress=BatchJobProgress(
            num_tasks_pending=None,
            num_tasks_completed=None,
        ),
    )
    return batch_job


@pytest.fixture
def batch_job_from_runnable(
    model_bundle_4: ModelBundle, model_endpoint_runnable: ModelEndpoint
) -> BatchJob:
    batch_job = BatchJob(
        record=BatchJobRecord(
            id="test_batch_job_id_runnable",
            created_at=datetime(2022, 1, 6),
            status=BatchJobStatus.PENDING,
            created_by=model_bundle_4.created_by,
            owner=model_bundle_4.owner,
            model_bundle=model_bundle_4,
            model_endpoint_id=model_endpoint_runnable.record.id,
            task_ids_location=None,
            result_location=None,
        ),
        model_endpoint=model_endpoint_runnable,
        progress=BatchJobProgress(
            num_tasks_pending=None,
            num_tasks_completed=None,
        ),
    )
    return batch_job


@pytest.fixture
def docker_image_batch_job_bundle_1_v1(test_api_key: str) -> DockerImageBatchJobBundle:
    batch_bundle = DockerImageBatchJobBundle(
        id="test_docker_image_batch_job_bundle_id_11",
        created_at=datetime(2022, 1, 1),
        name="test_docker_image_batch_job_bundle_1",
        created_by=test_api_key,
        owner=test_api_key,
        image_repository="image_repository",
        image_tag="image_tag_git_sha",
        command=["python", "script.py", "--arg1"],
        env=dict(ENV1="VAL1", ENV2="VAL2"),
        mount_location="/mount/location/to/config",
        cpus="1",
        memory=None,
        storage=None,
        gpus=None,
        gpu_type=None,
        public=False,
    )
    return batch_bundle


@pytest.fixture
def docker_image_batch_job_bundle_1_v2(test_api_key: str) -> DockerImageBatchJobBundle:
    batch_bundle = DockerImageBatchJobBundle(
        id="test_docker_image_batch_job_bundle_id_12",
        created_at=datetime(2022, 1, 3),
        name="test_docker_image_batch_job_bundle_1",
        created_by=test_api_key,
        owner=test_api_key,
        image_repository="image_repository",
        image_tag="image_tag_git_sha",
        command=["python", "script.py", "--arg2"],
        env=dict(ENV1="VAL3", ENV2="VAL4"),
        mount_location="/mount/location/to/config2",
        cpus="2",
        memory=None,
        storage=None,
        gpus=None,
        gpu_type=None,
        public=True,
    )
    return batch_bundle


@pytest.fixture
def docker_image_batch_job_bundle_2_v1(test_api_key: str) -> DockerImageBatchJobBundle:
    batch_bundle = DockerImageBatchJobBundle(
        id="test_docker_image_batch_job_bundle_id_21",
        created_at=datetime(2022, 1, 2),
        name="test_docker_image_batch_job_bundle_2",
        created_by=test_api_key,
        owner=test_api_key,
        image_repository="image_repository",
        image_tag="image_tag_git_sha",
        command=["python", "script2.py", "--arg1"],
        env=dict(ENV1="VAL1", ENV2="VAL2"),
        mount_location="/mount2/location/to/config",
        cpus="3",
        memory=None,
        storage=None,
        gpus=None,
        gpu_type=None,
        public=None,
    )
    return batch_bundle


@pytest.fixture
def build_endpoint_request_async_runnable_image(
    test_api_key: str, model_bundle_4: ModelBundle
) -> BuildEndpointRequest:
    build_endpoint_request = BuildEndpointRequest(
        model_endpoint_record=ModelEndpointRecord(
            id="test_model_endpoint_id_1",
            name="test_model_endpoint_name_1",
            created_by=test_api_key,
            created_at=datetime(2022, 1, 4),
            last_updated_at=datetime(2022, 1, 4),
            metadata={},
            creation_task_id="test_creation_task_id",
            endpoint_type=ModelEndpointType.ASYNC,
            destination="test_destination",
            status=ModelEndpointStatus.READY,
            current_model_bundle=model_bundle_4,
            owner=test_api_key,
        ),
        deployment_name=f"{test_api_key}-test_model_endpoint_name_1",
        aws_role="default",
        results_s3_bucket="test_s3_bucket",
        child_fn_info=None,
        post_inference_hooks=None,
        labels=dict(team="test_team", product="test_product"),
        min_workers=1,
        max_workers=3,
        per_worker=2,
        concurrent_requests_per_worker=1,
        cpus=3,
        gpus=1,
        memory="3G",
        gpu_type=GpuType.NVIDIA_TESLA_T4,
        storage="10G",
        nodes_per_worker=1,
        optimize_costs=False,
        broker_type=BrokerType.SQS,
        default_callback_url="https://example.com",
        default_callback_auth=CallbackAuth(
            root=CallbackBasicAuth(kind="basic", username="username", password="password")
        ),
    )
    return build_endpoint_request


@pytest.fixture
def build_endpoint_request_streaming_runnable_image(
    test_api_key: str, model_bundle_5: ModelBundle
) -> BuildEndpointRequest:
    build_endpoint_request = BuildEndpointRequest(
        model_endpoint_record=ModelEndpointRecord(
            id="test_model_endpoint_id_1",
            name="test_model_endpoint_name_1",
            created_by=test_api_key,
            created_at=datetime(2022, 1, 4),
            last_updated_at=datetime(2022, 1, 4),
            metadata={},
            creation_task_id="test_creation_task_id",
            endpoint_type=ModelEndpointType.SYNC,
            destination="test_destination",
            status=ModelEndpointStatus.READY,
            current_model_bundle=model_bundle_5,
            owner=test_api_key,
        ),
        deployment_name=f"{test_api_key}-test_model_endpoint_name_1",
        aws_role="default",
        results_s3_bucket="test_s3_bucket",
        child_fn_info=None,
        post_inference_hooks=None,
        labels=dict(team="test_team", product="test_product"),
        min_workers=1,
        max_workers=3,
        per_worker=2,
        concurrent_requests_per_worker=1,
        cpus=4,
        gpus=1,
        memory="4G",
        gpu_type=GpuType.NVIDIA_TESLA_T4,
        storage="10G",
        nodes_per_worker=1,
        optimize_costs=False,
        broker_type=BrokerType.SQS,
        default_callback_url="https://example.com",
        default_callback_auth=CallbackAuth(
            root=CallbackBasicAuth(kind="basic", username="username", password="password")
        ),
    )
    return build_endpoint_request


@pytest.fixture
def build_endpoint_request_sync_runnable_image(
    test_api_key: str, model_bundle_4: ModelBundle
) -> BuildEndpointRequest:
    build_endpoint_request = BuildEndpointRequest(
        model_endpoint_record=ModelEndpointRecord(
            id="test_model_endpoint_id_1",
            name="test_model_endpoint_name_1",
            created_by=test_api_key,
            created_at=datetime(2022, 1, 4),
            last_updated_at=datetime(2022, 1, 4),
            metadata={},
            creation_task_id="test_creation_task_id",
            endpoint_type=ModelEndpointType.SYNC,
            destination="test_destination",
            status=ModelEndpointStatus.READY,
            current_model_bundle=model_bundle_4,
            owner=test_api_key,
        ),
        deployment_name=f"{test_api_key}-test_model_endpoint_name_1",
        aws_role="default",
        results_s3_bucket="test_s3_bucket",
        child_fn_info=None,
        post_inference_hooks=None,
        labels=dict(team="test_team", product="test_product"),
        min_workers=1,
        max_workers=3,
        per_worker=2,
        concurrent_requests_per_worker=1,
        cpus=3,
        gpus=1,
        memory="4G",
        gpu_type=GpuType.NVIDIA_TESLA_T4,
        storage="10G",
        nodes_per_worker=1,
        optimize_costs=False,
        broker_type=BrokerType.SQS,
        default_callback_url="https://example.com",
        default_callback_auth=CallbackAuth(
            root=CallbackBasicAuth(kind="basic", username="username", password="password")
        ),
    )
    return build_endpoint_request


@pytest.fixture
def build_endpoint_request_sync_pytorch(
    test_api_key: str, model_bundle_1: ModelBundle
) -> BuildEndpointRequest:
    build_endpoint_request = BuildEndpointRequest(
        model_endpoint_record=ModelEndpointRecord(
            id="test_model_endpoint_id_1",
            name="test_model_endpoint_name_1",
            created_by=test_api_key,
            created_at=datetime(2022, 1, 4),
            last_updated_at=datetime(2022, 1, 4),
            metadata={},
            creation_task_id="test_creation_task_id",
            endpoint_type=ModelEndpointType.SYNC,
            destination="test_destination",
            status=ModelEndpointStatus.READY,
            current_model_bundle=model_bundle_1,
            owner=test_api_key,
        ),
        deployment_name=f"{test_api_key}-test_model_endpoint_name_1",
        aws_role="default",
        results_s3_bucket="test_s3_bucket",
        child_fn_info=None,
        post_inference_hooks=None,
        labels=dict(team="test_team", product="test_product"),
        min_workers=1,
        max_workers=3,
        per_worker=2,
        concurrent_requests_per_worker=1,
        cpus=1,
        gpus=1,
        memory="1G",
        gpu_type=GpuType.NVIDIA_TESLA_T4,
        storage="10G",
        nodes_per_worker=1,
        optimize_costs=False,
        broker_type=BrokerType.SQS,
        default_callback_url="https://example.com",
        default_callback_auth=CallbackAuth(
            root=CallbackBasicAuth(kind="basic", username="username", password="password")
        ),
    )
    return build_endpoint_request


@pytest.fixture
def build_endpoint_request_async_tensorflow(
    test_api_key: str, model_bundle_2: ModelBundle
) -> BuildEndpointRequest:
    build_endpoint_request = BuildEndpointRequest(
        model_endpoint_record=ModelEndpointRecord(
            id="test_model_endpoint_id_2",
            name="test_model_endpoint_name_2",
            created_by=test_api_key,
            created_at=datetime(2022, 1, 4),
            last_updated_at=datetime(2022, 1, 4),
            metadata={},
            creation_task_id="test_creation_task_id",
            endpoint_type=ModelEndpointType.ASYNC,
            destination="test_destination",
            status=ModelEndpointStatus.READY,
            current_model_bundle=model_bundle_2,
            owner=test_api_key,
        ),
        deployment_name=f"{test_api_key}-test_model_endpoint_name_2",
        aws_role="default",
        results_s3_bucket="test_s3_bucket",
        child_fn_info=None,
        post_inference_hooks=None,
        labels=dict(team="test_team", product="test_product"),
        min_workers=1,
        max_workers=3,
        per_worker=2,
        concurrent_requests_per_worker=1,
        cpus=1,
        gpus=0,
        memory="1G",
        gpu_type=None,
        storage=None,
        nodes_per_worker=1,
        optimize_costs=False,
        default_callback_url="https://example.com/path",
        default_callback_auth=CallbackAuth(
            root=CallbackBasicAuth(kind="basic", username="username", password="password")
        ),
    )
    return build_endpoint_request


@pytest.fixture
def build_endpoint_request_async_custom(
    test_api_key: str, model_bundle_3: ModelBundle
) -> BuildEndpointRequest:
    build_endpoint_request = BuildEndpointRequest(
        model_endpoint_record=ModelEndpointRecord(
            id="test_model_endpoint_id_3",
            name="test_model_endpoint_name_3",
            created_by=test_api_key,
            created_at=datetime(2022, 1, 4),
            last_updated_at=datetime(2022, 1, 4),
            metadata={},
            creation_task_id="test_creation_task_id",
            endpoint_type=ModelEndpointType.ASYNC,
            destination="test_destination",
            status=ModelEndpointStatus.READY,
            current_model_bundle=model_bundle_3,
            owner=test_api_key,
        ),
        deployment_name=f"{test_api_key}-test_model_endpoint_name_3",
        aws_role="default",
        results_s3_bucket="test_s3_bucket",
        child_fn_info=None,
        post_inference_hooks=None,
        labels=dict(team="test_team", product="test_product"),
        min_workers=1,
        max_workers=3,
        per_worker=2,
        concurrent_requests_per_worker=1,
        cpus=1,
        gpus=0,
        memory="1G",
        gpu_type=None,
        storage=None,
        nodes_per_worker=1,
        optimize_costs=True,
        broker_type=BrokerType.SQS,
        default_callback_url=None,
        default_callback_auth=None,
    )
    return build_endpoint_request


@pytest.fixture
def build_endpoint_request_async_zipartifact_highpri(
    test_api_key: str, model_bundle_3: ModelBundle
) -> BuildEndpointRequest:
    build_endpoint_request = BuildEndpointRequest(
        model_endpoint_record=ModelEndpointRecord(
            id="test_model_endpoint_id_3",
            name="test_model_endpoint_name_3",
            created_by=test_api_key,
            created_at=datetime(2022, 1, 4),
            last_updated_at=datetime(2022, 1, 4),
            metadata={},
            creation_task_id="test_creation_task_id",
            endpoint_type=ModelEndpointType.ASYNC,
            destination="test_destination",
            status=ModelEndpointStatus.READY,
            current_model_bundle=model_bundle_3,
            owner=test_api_key,
        ),
        high_priority=True,
        deployment_name=f"{test_api_key}-test_model_endpoint_name_3",
        aws_role="default",
        results_s3_bucket="test_s3_bucket",
        child_fn_info=None,
        post_inference_hooks=None,
        labels=dict(team="test_team", product="test_product"),
        min_workers=1,
        max_workers=3,
        per_worker=2,
        concurrent_requests_per_worker=1,
        cpus=1,
        gpus=0,
        memory="1G",
        gpu_type=None,
        storage=None,
        nodes_per_worker=1,
        optimize_costs=True,
        broker_type=BrokerType.SQS,
        default_callback_url=None,
        default_callback_auth=None,
    )
    return build_endpoint_request


@pytest.fixture
def build_endpoint_request_sync_custom(
    test_api_key: str, model_bundle_3: ModelBundle
) -> BuildEndpointRequest:
    build_endpoint_request = BuildEndpointRequest(
        model_endpoint_record=ModelEndpointRecord(
            id="test_model_endpoint_id_3",
            name="test_model_endpoint_name_3",
            created_by=test_api_key,
            created_at=datetime(2022, 1, 4),
            last_updated_at=datetime(2022, 1, 4),
            metadata={},
            creation_task_id="test_creation_task_id",
            endpoint_type=ModelEndpointType.SYNC,
            destination="test_destination",
            status=ModelEndpointStatus.READY,
            current_model_bundle=model_bundle_3,
            owner=test_api_key,
        ),
        deployment_name=f"{test_api_key}-test_model_endpoint_name_3",
        aws_role="default",
        results_s3_bucket="test_s3_bucket",
        child_fn_info=None,
        post_inference_hooks=None,
        labels=dict(team="test_team", product="test_product"),
        min_workers=1,
        max_workers=3,
        per_worker=2,
        concurrent_requests_per_worker=1,
        cpus=1,
        gpus=0,
        memory="1G",
        gpu_type=None,
        storage=None,
        nodes_per_worker=1,
        optimize_costs=True,
        default_callback_url=None,
        default_callback_auth=None,
    )
    return build_endpoint_request


@pytest.fixture
def endpoint_predict_request_1() -> Tuple[EndpointPredictV1Request, Dict[str, Any]]:
    request = EndpointPredictV1Request(
        url="test_url",
        return_pickled=False,
    )
    request_dict = request.dict()
    return request, request_dict


@pytest.fixture
def endpoint_predict_request_2() -> Tuple[EndpointPredictV1Request, Dict[str, Any]]:
    request = EndpointPredictV1Request(
        args=["test_arg_1", "test_arg_2"],
        callback_url="http://test_callback_url.xyz",
        callback_auth=CallbackAuth(
            root=CallbackBasicAuth(kind="basic", username="test_username", password="test_password")
        ),
        return_pickled=True,
    )
    request_dict = request.dict()
    return request, request_dict


@pytest.fixture
def sync_endpoint_predict_request_1() -> Tuple[SyncEndpointPredictV1Request, Dict[str, Any]]:
    request = SyncEndpointPredictV1Request(
        url="test_url",
        return_pickled=False,
        timeout_seconds=10,
        num_retries=5,
    )
    request_dict = request.dict()
    return request, request_dict


@pytest.fixture
def llm_model_endpoint_async(
    test_api_key: str, model_bundle_1: ModelBundle
) -> Tuple[ModelEndpoint, Any]:
    model_endpoint = ModelEndpoint(
        record=ModelEndpointRecord(
            id="test_llm_model_endpoint_id_1",
            name="test_llm_model_endpoint_name_1",
            created_by=test_api_key,
            created_at=datetime(2022, 1, 3),
            last_updated_at=datetime(2022, 1, 3),
            metadata={
                "_llm": {
                    "model_name": "llama-7b",
                    "source": "hugging_face",
                    "inference_framework": "deepspeed",
                    "inference_framework_image_tag": "123",
                    "num_shards": 4,
                }
            },
            creation_task_id="test_creation_task_id",
            endpoint_type=ModelEndpointType.ASYNC,
            destination="test_destination",
            status=ModelEndpointStatus.READY,
            current_model_bundle=model_bundle_1,
            owner=test_api_key,
            public_inference=True,
        ),
        infra_state=ModelEndpointInfraState(
            deployment_name=f"{test_api_key}-test_llm_model_endpoint_name_1",
            aws_role="test_aws_role",
            results_s3_bucket="test_s3_bucket",
            child_fn_info=None,
            labels={},
            prewarm=True,
            high_priority=False,
            deployment_state=ModelEndpointDeploymentState(
                min_workers=1,
                max_workers=3,
                per_worker=2,
                concurrent_requests_per_worker=1,
                available_workers=1,
                unavailable_workers=1,
            ),
            resource_state=ModelEndpointResourceState(
                cpus=1,
                gpus=1,
                memory="1G",
                gpu_type=GpuType.NVIDIA_TESLA_T4,
                storage="10G",
                nodes_per_worker=1,
                optimize_costs=True,
            ),
            user_config_state=ModelEndpointUserConfigState(
                app_config=model_bundle_1.app_config,
                endpoint_config=ModelEndpointConfig(
                    bundle_name=model_bundle_1.name,
                    endpoint_name="test_llm_model_endpoint_name_1",
                    post_inference_hooks=["callback"],
                    default_callback_url="http://www.example.com",
                    default_callback_auth=CallbackAuth(
                        root=CallbackBasicAuth(
                            kind="basic",
                            username="test_username",
                            password="test_password",
                        ),
                    ),
                ),
            ),
            num_queued_items=1,
            image="test_image",
        ),
    )
    model_endpoint_json: Dict[str, Any] = {
        "id": "test_llm_model_endpoint_id_1",
        "name": "test_llm_model_endpoint_name_1",
        "model_name": "llama-7b",
        "source": "hugging_face",
        "status": "READY",
        "inference_framework": "deepspeed",
        "inference_framework_image_tag": "123",
        "num_shards": 4,
        "spec": {
            "id": "test_llm_model_endpoint_id_1",
            "name": "test_llm_model_endpoint_name_1",
            "endpoint_type": "async",
            "destination": "test_destination",
            "deployment_name": f"{test_api_key}-test_llm_model_endpoint_name_1",
            "metadata": {
                "_llm": {
                    "model_name": "llama-7b",
                    "source": "hugging_face",
                    "inference_framework": "deepspeed",
                    "inference_framework_image_tag": "123",
                    "num_shards": 4,
                }
            },
            "bundle_name": "test_model_bundle_name_1",
            "status": "READY",
            "post_inference_hooks": ["callback"],
            "default_callback_url": "http://www.example.com",
            "default_callback_auth": {
                "kind": "basic",
                "username": "test_username",
                "password": "test_password",
            },
            "labels": {},
            "aws_role": "test_aws_role",
            "results_s3_bucket": "test_s3_bucket",
            "created_by": test_api_key,
            "created_at": "2022-01-03T00:00:00",
            "last_updated_at": "2022-01-03T00:00:00",
            "deployment_state": {
                "min_workers": 1,
                "max_workers": 3,
                "per_worker": 2,
                "concurrent_requests_per_worker": 1,
                "available_workers": 1,
                "unavailable_workers": 1,
            },
            "resource_state": {
                "cpus": 1,
                "gpus": 1,
                "memory": "1G",
                "gpu_type": "nvidia-tesla-t4",
                "storage": "10G",
                "nodes_per_worker": 1,
                "optimize_costs": True,
            },
            "num_queued_items": 1,
            "public_inference": True,
        },
    }
    return model_endpoint, model_endpoint_json


@pytest.fixture
def llm_model_endpoint_sync(
    test_api_key: str, model_bundle_1: ModelBundle
) -> Tuple[ModelEndpoint, Any]:
    model_endpoint = ModelEndpoint(
        record=ModelEndpointRecord(
            id="test_llm_model_endpoint_id_2",
            name="test_llm_model_endpoint_name_1",
            created_by=test_api_key,
            created_at=datetime(2022, 1, 3),
            last_updated_at=datetime(2022, 1, 3),
            metadata={
                "_llm": {
                    "model_name": "llama-7b",
                    "source": "hugging_face",
                    "inference_framework": "vllm",
                    "inference_framework_image_tag": "123",
                    "num_shards": 4,
                }
            },
            creation_task_id="test_creation_task_id",
            endpoint_type=ModelEndpointType.SYNC,
            destination="test_destination",
            status=ModelEndpointStatus.READY,
            current_model_bundle=model_bundle_1,
            owner=test_api_key,
            public_inference=True,
        ),
        infra_state=ModelEndpointInfraState(
            deployment_name=f"{test_api_key}-test_llm_model_endpoint_name_1",
            aws_role="test_aws_role",
            results_s3_bucket="test_s3_bucket",
            child_fn_info=None,
            labels={},
            prewarm=True,
            high_priority=False,
            deployment_state=ModelEndpointDeploymentState(
                min_workers=1,
                max_workers=3,
                per_worker=2,
                concurrent_requests_per_worker=1,
                available_workers=1,
                unavailable_workers=1,
            ),
            resource_state=ModelEndpointResourceState(
                cpus=1,
                gpus=1,
                memory="1G",
                gpu_type=GpuType.NVIDIA_TESLA_T4,
                storage="10G",
                nodes_per_worker=1,
                optimize_costs=True,
            ),
            user_config_state=ModelEndpointUserConfigState(
                app_config=model_bundle_1.app_config,
                endpoint_config=ModelEndpointConfig(
                    bundle_name=model_bundle_1.name,
                    endpoint_name="test_llm_model_endpoint_name_1",
                    post_inference_hooks=["callback"],
                    default_callback_url="http://www.example.com",
                    default_callback_auth=CallbackAuth(
                        root=CallbackBasicAuth(
                            kind="basic",
                            username="test_username",
                            password="test_password",
                        ),
                    ),
                ),
            ),
            num_queued_items=1,
            image="test_image",
        ),
    )
    model_endpoint_json: Dict[str, Any] = {
        "id": "test_llm_model_endpoint_id_2",
        "name": "test_llm_model_endpoint_name_1",
        "model_name": "llama-7b",
        "source": "hugging_face",
        "status": "READY",
        "inference_framework": "vllm",
        "inference_framework_image_tag": "123",
        "num_shards": 4,
        "spec": {
            "id": "test_llm_model_endpoint_id_2",
            "name": "test_llm_model_endpoint_name_1",
            "endpoint_type": "sync",
            "destination": "test_destination",
            "deployment_name": f"{test_api_key}-test_llm_model_endpoint_name_1",
            "metadata": {
                "_llm": {
                    "model_name": "llama-7b",
                    "source": "hugging_face",
                    "inference_framework": "vllm",
                    "inference_framework_image_tag": "123",
                    "num_shards": 4,
                }
            },
            "bundle_name": "test_model_bundle_name_1",
            "status": "READY",
            "post_inference_hooks": ["callback"],
            "default_callback_url": "http://www.example.com",
            "default_callback_auth": {
                "kind": "basic",
                "username": "test_username",
                "password": "test_password",
            },
            "labels": {},
            "aws_role": "test_aws_role",
            "results_s3_bucket": "test_s3_bucket",
            "created_by": test_api_key,
            "created_at": "2022-01-03T00:00:00",
            "last_updated_at": "2022-01-03T00:00:00",
            "deployment_state": {
                "min_workers": 1,
                "max_workers": 3,
                "per_worker": 2,
                "concurrent_requests_per_worker": 1,
                "available_workers": 1,
                "unavailable_workers": 1,
            },
            "resource_state": {
                "cpus": 1,
                "gpus": 1,
                "memory": "1G",
                "gpu_type": "nvidia-tesla-t4",
                "storage": "10G",
                "nodes_per_worker": 1,
                "optimize_costs": True,
            },
            "num_queued_items": 1,
            "public_inference": True,
        },
    }
    return model_endpoint, model_endpoint_json


@pytest.fixture
def llm_model_endpoint_stream(
    test_api_key: str, model_bundle_1: ModelBundle
) -> Tuple[ModelEndpoint, Any]:
    model_endpoint = ModelEndpoint(
        record=ModelEndpointRecord(
            id="test_llm_model_endpoint_id_2",
            name="test_llm_model_endpoint_name_1",
            created_by=test_api_key,
            created_at=datetime(2022, 1, 3),
            last_updated_at=datetime(2022, 1, 3),
            metadata={
                "_llm": {
                    "model_name": "llama-7b",
                    "source": "hugging_face",
                    "inference_framework": "vllm",
                    "inference_framework_image_tag": "123",
                    "num_shards": 4,
                }
            },
            creation_task_id="test_creation_task_id",
            endpoint_type=ModelEndpointType.STREAMING,
            destination="test_destination",
            status=ModelEndpointStatus.READY,
            current_model_bundle=model_bundle_1,
            owner=test_api_key,
            public_inference=True,
        ),
        infra_state=ModelEndpointInfraState(
            deployment_name=f"{test_api_key}-test_llm_model_endpoint_name_1",
            aws_role="test_aws_role",
            results_s3_bucket="test_s3_bucket",
            child_fn_info=None,
            labels={},
            prewarm=True,
            high_priority=False,
            deployment_state=ModelEndpointDeploymentState(
                min_workers=1,
                max_workers=3,
                per_worker=2,
                concurrent_requests_per_worker=1,
                available_workers=1,
                unavailable_workers=1,
            ),
            resource_state=ModelEndpointResourceState(
                cpus=1,
                gpus=1,
                memory="1G",
                gpu_type=GpuType.NVIDIA_TESLA_T4,
                storage="10G",
                nodes_per_worker=1,
                optimize_costs=True,
            ),
            user_config_state=ModelEndpointUserConfigState(
                app_config=model_bundle_1.app_config,
                endpoint_config=ModelEndpointConfig(
                    bundle_name=model_bundle_1.name,
                    endpoint_name="test_llm_model_endpoint_name_1",
                    post_inference_hooks=["callback"],
                    default_callback_url="http://www.example.com",
                    default_callback_auth=CallbackAuth(
                        root=CallbackBasicAuth(
                            kind="basic",
                            username="test_username",
                            password="test_password",
                        ),
                    ),
                ),
            ),
            num_queued_items=1,
            image="test_image",
        ),
    )
    model_endpoint_json: Dict[str, Any] = {
        "id": "test_llm_model_endpoint_id_2",
        "name": "test_llm_model_endpoint_name_1",
        "model_name": "llama-7b",
        "source": "hugging_face",
        "status": "READY",
        "inference_framework": "vllm",
        "inference_framework_image_tag": "123",
        "num_shards": 4,
        "spec": {
            "id": "test_llm_model_endpoint_id_2",
            "name": "test_llm_model_endpoint_name_1",
            "endpoint_type": "streaming",
            "destination": "test_destination",
            "deployment_name": f"{test_api_key}-test_llm_model_endpoint_name_1",
            "metadata": {
                "_llm": {
                    "model_name": "llama-7b",
                    "source": "hugging_face",
                    "inference_framework": "vllm",
                    "inference_framework_image_tag": "123",
                    "num_shards": 4,
                }
            },
            "bundle_name": "test_model_bundle_name_1",
            "status": "READY",
            "post_inference_hooks": ["callback"],
            "default_callback_url": "http://www.example.com",
            "default_callback_auth": {
                "kind": "basic",
                "username": "test_username",
                "password": "test_password",
            },
            "labels": {},
            "aws_role": "test_aws_role",
            "results_s3_bucket": "test_s3_bucket",
            "created_by": test_api_key,
            "created_at": "2022-01-03T00:00:00",
            "last_updated_at": "2022-01-03T00:00:00",
            "deployment_state": {
                "min_workers": 1,
                "max_workers": 3,
                "per_worker": 2,
                "concurrent_requests_per_worker": 1,
                "available_workers": 1,
                "unavailable_workers": 1,
            },
            "resource_state": {
                "cpus": 1,
                "gpus": 1,
                "memory": "1G",
                "gpu_type": "nvidia-tesla-t4",
                "storage": "10G",
                "nodes_per_worker": 1,
                "optimize_costs": True,
            },
            "num_queued_items": 1,
            "public_inference": True,
        },
    }
    return model_endpoint, model_endpoint_json


@pytest.fixture
def llm_model_endpoint_sync_tgi(
    test_api_key: str, model_bundle_1: ModelBundle
) -> Tuple[ModelEndpoint, Any]:
    model_endpoint = ModelEndpoint(
        record=ModelEndpointRecord(
            id="test_llm_model_endpoint_id_2",
            name="test_llm_model_endpoint_name_1",
            created_by=test_api_key,
            created_at=datetime(2022, 1, 3),
            last_updated_at=datetime(2022, 1, 3),
            metadata={
                "_llm": {
                    "model_name": "llama-7b",
                    "source": "hugging_face",
                    "inference_framework": "text_generation_inference",
                    "inference_framework_image_tag": "0.9.4",
                    "num_shards": 4,
                }
            },
            creation_task_id="test_creation_task_id",
            endpoint_type=ModelEndpointType.SYNC,
            destination="test_destination",
            status=ModelEndpointStatus.READY,
            current_model_bundle=model_bundle_1,
            owner=test_api_key,
            public_inference=True,
        ),
        infra_state=ModelEndpointInfraState(
            deployment_name=f"{test_api_key}-test_llm_model_endpoint_name_1",
            aws_role="test_aws_role",
            results_s3_bucket="test_s3_bucket",
            child_fn_info=None,
            labels={},
            prewarm=True,
            high_priority=False,
            deployment_state=ModelEndpointDeploymentState(
                min_workers=1,
                max_workers=3,
                per_worker=2,
                concurrent_requests_per_worker=1,
                available_workers=1,
                unavailable_workers=1,
            ),
            resource_state=ModelEndpointResourceState(
                cpus=1,
                gpus=1,
                memory="1G",
                gpu_type=GpuType.NVIDIA_TESLA_T4,
                storage="10G",
                nodes_per_worker=1,
                optimize_costs=True,
            ),
            user_config_state=ModelEndpointUserConfigState(
                app_config=model_bundle_1.app_config,
                endpoint_config=ModelEndpointConfig(
                    bundle_name=model_bundle_1.name,
                    endpoint_name="test_llm_model_endpoint_name_1",
                    post_inference_hooks=["callback"],
                    default_callback_url="http://www.example.com",
                    default_callback_auth=CallbackAuth(
                        root=CallbackBasicAuth(
                            kind="basic",
                            username="test_username",
                            password="test_password",
                        ),
                    ),
                ),
            ),
            num_queued_items=1,
            image="test_image",
        ),
    )
    model_endpoint_json: Dict[str, Any] = {
        "id": "test_llm_model_endpoint_id_2",
        "name": "test_llm_model_endpoint_name_1",
        "model_name": "llama-7b",
        "source": "hugging_face",
        "status": "READY",
        "inference_framework": "text_generation_inference",
        "inference_framework_image_tag": "0.9.4",
        "num_shards": 4,
        "spec": {
            "id": "test_llm_model_endpoint_id_2",
            "name": "test_llm_model_endpoint_name_1",
            "endpoint_type": "sync",
            "destination": "test_destination",
            "deployment_name": f"{test_api_key}-test_llm_model_endpoint_name_1",
            "metadata": {
                "_llm": {
                    "model_name": "llama-7b",
                    "source": "hugging_face",
                    "inference_framework": "text_generation_inference",
                    "inference_framework_image_tag": "0.9.4",
                    "num_shards": 4,
                }
            },
            "bundle_name": "test_model_bundle_name_1",
            "status": "READY",
            "post_inference_hooks": ["callback"],
            "default_callback_url": "http://www.example.com",
            "default_callback_auth": {
                "kind": "basic",
                "username": "test_username",
                "password": "test_password",
            },
            "labels": {},
            "aws_role": "test_aws_role",
            "results_s3_bucket": "test_s3_bucket",
            "created_by": test_api_key,
            "created_at": "2022-01-03T00:00:00",
            "last_updated_at": "2022-01-03T00:00:00",
            "deployment_state": {
                "min_workers": 1,
                "max_workers": 3,
                "per_worker": 2,
                "concurrent_requests_per_worker": 1,
                "available_workers": 1,
                "unavailable_workers": 1,
            },
            "resource_state": {
                "cpus": 1,
                "gpus": 1,
                "memory": "1G",
                "gpu_type": "nvidia-tesla-t4",
                "storage": "10G",
                "nodes_per_worker": 1,
                "optimize_costs": True,
            },
            "num_queued_items": 1,
            "public_inference": True,
        },
    }
    return model_endpoint, model_endpoint_json


@pytest.fixture
def llm_model_endpoint_sync_lightllm(
    test_api_key: str, model_bundle_1: ModelBundle
) -> Tuple[ModelEndpoint, Any]:
    model_endpoint = ModelEndpoint(
        record=ModelEndpointRecord(
            id="test_llm_model_endpoint_id_2",
            name="test_llm_model_endpoint_name_1",
            created_by=test_api_key,
            created_at=datetime(2022, 1, 3),
            last_updated_at=datetime(2022, 1, 3),
            metadata={
                "_llm": {
                    "model_name": "llama-7b",
                    "source": "hugging_face",
                    "inference_framework": "lightllm",
                    "inference_framework_image_tag": "0.9.4",
                    "num_shards": 4,
                }
            },
            creation_task_id="test_creation_task_id",
            endpoint_type=ModelEndpointType.SYNC,
            destination="test_destination",
            status=ModelEndpointStatus.READY,
            current_model_bundle=model_bundle_1,
            owner=test_api_key,
            public_inference=True,
        ),
        infra_state=ModelEndpointInfraState(
            deployment_name=f"{test_api_key}-test_llm_model_endpoint_name_1",
            aws_role="test_aws_role",
            results_s3_bucket="test_s3_bucket",
            child_fn_info=None,
            labels={},
            prewarm=True,
            high_priority=False,
            deployment_state=ModelEndpointDeploymentState(
                min_workers=1,
                max_workers=3,
                per_worker=2,
                concurrent_requests_per_worker=1,
                available_workers=1,
                unavailable_workers=1,
            ),
            resource_state=ModelEndpointResourceState(
                cpus=1,
                gpus=1,
                memory="1G",
                gpu_type=GpuType.NVIDIA_TESLA_T4,
                storage="10G",
                nodes_per_worker=1,
                optimize_costs=True,
            ),
            user_config_state=ModelEndpointUserConfigState(
                app_config=model_bundle_1.app_config,
                endpoint_config=ModelEndpointConfig(
                    bundle_name=model_bundle_1.name,
                    endpoint_name="test_llm_model_endpoint_name_1",
                    post_inference_hooks=["callback"],
                    default_callback_url="http://www.example.com",
                    default_callback_auth=CallbackAuth(
                        root=CallbackBasicAuth(
                            kind="basic",
                            username="test_username",
                            password="test_password",
                        ),
                    ),
                ),
            ),
            num_queued_items=1,
            image="test_image",
        ),
    )
    model_endpoint_json: Dict[str, Any] = {
        "id": "test_llm_model_endpoint_id_2",
        "name": "test_llm_model_endpoint_name_1",
        "model_name": "llama-7b",
        "source": "hugging_face",
        "status": "READY",
        "inference_framework": "lightllm",
        "inference_framework_image_tag": "0.9.4",
        "num_shards": 4,
        "spec": {
            "id": "test_llm_model_endpoint_id_2",
            "name": "test_llm_model_endpoint_name_1",
            "endpoint_type": "sync",
            "destination": "test_destination",
            "deployment_name": f"{test_api_key}-test_llm_model_endpoint_name_1",
            "metadata": {
                "_llm": {
                    "model_name": "llama-7b",
                    "source": "hugging_face",
                    "inference_framework": "lightllm",
                    "inference_framework_image_tag": "0.9.4",
                    "num_shards": 4,
                }
            },
            "bundle_name": "test_model_bundle_name_1",
            "status": "READY",
            "post_inference_hooks": ["callback"],
            "default_callback_url": "http://www.example.com",
            "default_callback_auth": {
                "kind": "basic",
                "username": "test_username",
                "password": "test_password",
            },
            "labels": {},
            "aws_role": "test_aws_role",
            "results_s3_bucket": "test_s3_bucket",
            "created_by": test_api_key,
            "created_at": "2022-01-03T00:00:00",
            "last_updated_at": "2022-01-03T00:00:00",
            "deployment_state": {
                "min_workers": 1,
                "max_workers": 3,
                "per_worker": 2,
                "concurrent_requests_per_worker": 1,
                "available_workers": 1,
                "unavailable_workers": 1,
            },
            "resource_state": {
                "cpus": 1,
                "gpus": 1,
                "memory": "1G",
                "gpu_type": "nvidia-tesla-t4",
                "storage": "10G",
                "nodes_per_worker": 1,
                "optimize_costs": True,
            },
            "num_queued_items": 1,
            "public_inference": True,
        },
    }
    return model_endpoint, model_endpoint_json


@pytest.fixture
def llm_model_endpoint_sync_trt_llm(
    test_api_key: str, model_bundle_1: ModelBundle
) -> Tuple[ModelEndpoint, Any]:
    model_endpoint = ModelEndpoint(
        record=ModelEndpointRecord(
            id="test_llm_model_endpoint_id_2",
            name="test_llm_model_endpoint_name_1",
            created_by=test_api_key,
            created_at=datetime(2022, 1, 3),
            last_updated_at=datetime(2022, 1, 3),
            metadata={
                "_llm": {
                    "model_name": "llama-7b",
                    "source": "hugging_face",
                    "inference_framework": "tensorrt_llm",
                    "inference_framework_image_tag": "0.9.4",
                    "num_shards": 4,
                }
            },
            creation_task_id="test_creation_task_id",
            endpoint_type=ModelEndpointType.SYNC,
            destination="test_destination",
            status=ModelEndpointStatus.READY,
            current_model_bundle=model_bundle_1,
            owner=test_api_key,
            public_inference=True,
        ),
        infra_state=ModelEndpointInfraState(
            deployment_name=f"{test_api_key}-test_llm_model_endpoint_name_1",
            aws_role="test_aws_role",
            results_s3_bucket="test_s3_bucket",
            child_fn_info=None,
            labels={},
            prewarm=True,
            high_priority=False,
            deployment_state=ModelEndpointDeploymentState(
                min_workers=1,
                max_workers=3,
                per_worker=2,
                concurrent_requests_per_worker=1,
                available_workers=1,
                unavailable_workers=1,
            ),
            resource_state=ModelEndpointResourceState(
                cpus=1,
                gpus=1,
                memory="1G",
                gpu_type=GpuType.NVIDIA_TESLA_T4,
                storage="10G",
                nodes_per_worker=1,
                optimize_costs=True,
            ),
            user_config_state=ModelEndpointUserConfigState(
                app_config=model_bundle_1.app_config,
                endpoint_config=ModelEndpointConfig(
                    bundle_name=model_bundle_1.name,
                    endpoint_name="test_llm_model_endpoint_name_1",
                    post_inference_hooks=["callback"],
                    default_callback_url="http://www.example.com",
                    default_callback_auth=CallbackAuth(
                        root=CallbackBasicAuth(
                            kind="basic",
                            username="test_username",
                            password="test_password",
                        ),
                    ),
                ),
            ),
            num_queued_items=1,
            image="test_image",
        ),
    )
    model_endpoint_json: Dict[str, Any] = {
        "id": "test_llm_model_endpoint_id_2",
        "name": "test_llm_model_endpoint_name_1",
        "model_name": "llama-7b",
        "source": "hugging_face",
        "status": "READY",
        "inference_framework": "tensorrt_llm",
        "inference_framework_image_tag": "0.9.4",
        "num_shards": 4,
        "spec": {
            "id": "test_llm_model_endpoint_id_2",
            "name": "test_llm_model_endpoint_name_1",
            "endpoint_type": "sync",
            "destination": "test_destination",
            "deployment_name": f"{test_api_key}-test_llm_model_endpoint_name_1",
            "metadata": {
                "_llm": {
                    "model_name": "llama-7b",
                    "source": "hugging_face",
                    "inference_framework": "tensorrt_llm",
                    "inference_framework_image_tag": "0.9.4",
                    "num_shards": 4,
                }
            },
            "bundle_name": "test_model_bundle_name_1",
            "status": "READY",
            "post_inference_hooks": ["callback"],
            "default_callback_url": "http://www.example.com",
            "default_callback_auth": {
                "kind": "basic",
                "username": "test_username",
                "password": "test_password",
            },
            "labels": {},
            "aws_role": "test_aws_role",
            "results_s3_bucket": "test_s3_bucket",
            "created_by": test_api_key,
            "created_at": "2022-01-03T00:00:00",
            "last_updated_at": "2022-01-03T00:00:00",
            "deployment_state": {
                "min_workers": 1,
                "max_workers": 3,
                "per_worker": 2,
                "concurrent_requests_per_worker": 1,
                "available_workers": 1,
                "unavailable_workers": 1,
            },
            "resource_state": {
                "cpus": 1,
                "gpus": 1,
                "memory": "1G",
                "gpu_type": "nvidia-tesla-t4",
                "storage": "10G",
                "nodes_per_worker": 1,
                "optimize_costs": True,
            },
            "num_queued_items": 1,
            "public_inference": True,
        },
    }
    return model_endpoint, model_endpoint_json


@pytest.fixture
def llm_model_endpoint_streaming(test_api_key: str, model_bundle_5: ModelBundle) -> ModelEndpoint:
    # model_bundle_5 is a runnable bundle
    model_endpoint = ModelEndpoint(
        record=ModelEndpointRecord(
            id="test_model_endpoint_id_streaming",
            name="test_model_endpoint_name_streaming",
            created_by=test_api_key,
            created_at=datetime(2022, 1, 4),
            last_updated_at=datetime(2022, 1, 4),
            metadata={
                "_llm": {
                    "model_name": "llama-7b",
                    "source": "hugging_face",
                    "inference_framework": "deepspeed",
                    "inference_framework_image_tag": "123",
                    "num_shards": 4,
                }
            },
            creation_task_id="test_creation_task_id",
            endpoint_type=ModelEndpointType.STREAMING,
            destination="test_destination",
            status=ModelEndpointStatus.READY,
            current_model_bundle=model_bundle_5,
            owner=test_api_key,
            public_inference=True,
        ),
        infra_state=ModelEndpointInfraState(
            deployment_name=f"{test_api_key}-test_model_endpoint_name_streaming",
            aws_role="default",
            results_s3_bucket="test_s3_bucket",
            child_fn_info=None,
            post_inference_hooks=None,
            labels=dict(team="test_team", product="test_product"),
            prewarm=False,
            high_priority=True,
            deployment_state=ModelEndpointDeploymentState(
                min_workers=1,
                max_workers=3,
                per_worker=2,
                concurrent_requests_per_worker=1,
                available_workers=1,
                unavailable_workers=1,
            ),
            resource_state=ModelEndpointResourceState(
                cpus=1,
                gpus=1,
                memory="1G",
                gpu_type=GpuType.NVIDIA_TESLA_T4,
                storage="10G",
                nodes_per_worker=1,
                optimize_costs=False,
            ),
            user_config_state=ModelEndpointUserConfigState(
                app_config=model_bundle_5.app_config,
                endpoint_config=ModelEndpointConfig(
                    bundle_name=model_bundle_5.name,
                    endpoint_name="test_model_endpoint_name_streaming",
                    post_inference_hooks=None,
                ),
            ),
            image="000000000000.dkr.ecr.us-west-2.amazonaws.com/hello:there",
        ),
    )
    return model_endpoint


@pytest.fixture
def llm_model_endpoint_text_generation_inference(
    test_api_key: str, model_bundle_1: ModelBundle
) -> Tuple[ModelEndpoint, Any]:
    return ModelEndpoint(
        record=ModelEndpointRecord(
            id="test_llm_model_endpoint_id_3",
            name="test_llm_model_endpoint_name_tgi",
            created_by=test_api_key,
            created_at=datetime(2022, 1, 3),
            last_updated_at=datetime(2022, 1, 3),
            metadata={
                "_llm": {
                    "model_name": "llama-7b",
                    "source": "hugging_face",
                    "inference_framework": "text_generation_inference",
                    "inference_framework_image_tag": "0.9.4",
                    "num_shards": 4,
                }
            },
            creation_task_id="test_creation_task_id",
            endpoint_type=ModelEndpointType.STREAMING,
            destination="test_destination",
            status=ModelEndpointStatus.READY,
            current_model_bundle=model_bundle_1,
            owner=test_api_key,
            public_inference=True,
        ),
        infra_state=ModelEndpointInfraState(
            deployment_name=f"{test_api_key}-test_llm_model_endpoint_name_tgi",
            aws_role="test_aws_role",
            results_s3_bucket="test_s3_bucket",
            child_fn_info=None,
            labels={},
            prewarm=True,
            high_priority=False,
            deployment_state=ModelEndpointDeploymentState(
                min_workers=1,
                max_workers=3,
                per_worker=2,
                concurrent_requests_per_worker=1,
                available_workers=1,
                unavailable_workers=1,
            ),
            resource_state=ModelEndpointResourceState(
                cpus=1,
                gpus=1,
                memory="1G",
                gpu_type=GpuType.NVIDIA_TESLA_T4,
                storage="10G",
                nodes_per_worker=1,
                optimize_costs=True,
            ),
            user_config_state=ModelEndpointUserConfigState(
                app_config=model_bundle_1.app_config,
                endpoint_config=ModelEndpointConfig(
                    bundle_name=model_bundle_1.name,
                    endpoint_name="test_llm_model_endpoint_name_1",
                    post_inference_hooks=["callback"],
                    default_callback_url="http://www.example.com",
                    default_callback_auth=CallbackAuth(
                        root=CallbackBasicAuth(
                            kind="basic",
                            username="test_username",
                            password="test_password",
                        ),
                    ),
                ),
            ),
            num_queued_items=1,
            image="test_image",
        ),
    )


@pytest.fixture
def llm_model_endpoint_trt_llm(
    test_api_key: str, model_bundle_1: ModelBundle
) -> Tuple[ModelEndpoint, Any]:
    return ModelEndpoint(
        record=ModelEndpointRecord(
            id="test_llm_model_endpoint_id_3",
            name="test_llm_model_endpoint_name_trt_llm",
            created_by=test_api_key,
            created_at=datetime(2022, 1, 3),
            last_updated_at=datetime(2022, 1, 3),
            metadata={
                "_llm": {
                    "model_name": "llama-2-7b",
                    "source": "hugging_face",
                    "inference_framework": "tensorrt_llm",
                    "inference_framework_image_tag": "23.10",
                    "num_shards": 4,
                }
            },
            creation_task_id="test_creation_task_id",
            endpoint_type=ModelEndpointType.STREAMING,
            destination="test_destination",
            status=ModelEndpointStatus.READY,
            current_model_bundle=model_bundle_1,
            owner=test_api_key,
            public_inference=True,
        ),
        infra_state=ModelEndpointInfraState(
            deployment_name=f"{test_api_key}-test_llm_model_endpoint_name_trt_llm",
            aws_role="test_aws_role",
            results_s3_bucket="test_s3_bucket",
            child_fn_info=None,
            labels={},
            prewarm=True,
            high_priority=False,
            deployment_state=ModelEndpointDeploymentState(
                min_workers=1,
                max_workers=3,
                per_worker=2,
                concurrent_requests_per_worker=1,
                available_workers=1,
                unavailable_workers=1,
            ),
            resource_state=ModelEndpointResourceState(
                cpus=1,
                gpus=1,
                memory="1G",
                gpu_type=GpuType.NVIDIA_TESLA_T4,
                storage="10G",
                nodes_per_worker=1,
                optimize_costs=True,
            ),
            user_config_state=ModelEndpointUserConfigState(
                app_config=model_bundle_1.app_config,
                endpoint_config=ModelEndpointConfig(
                    bundle_name=model_bundle_1.name,
                    endpoint_name="test_llm_model_endpoint_name_1",
                    post_inference_hooks=["callback"],
                    default_callback_url="http://www.example.com",
                    default_callback_auth=CallbackAuth(
                        root=CallbackBasicAuth(
                            kind="basic",
                            username="test_username",
                            password="test_password",
                        ),
                    ),
                ),
            ),
            num_queued_items=1,
            image="test_image",
        ),
    )


def mocked__get_recommended_hardware_config_map():
    async def async_mock(*args, **kwargs):  # noqa
        return {
            "byGpuMemoryGb": """
    - gpu_memory_le: 20
      cpus: 5
      gpus: 1
      memory: 20Gi
      storage: 40Gi
      gpu_type: nvidia-hopper-h100-1g20gb
      nodes_per_worker: 1
    - gpu_memory_le: 40
      cpus: 10
      gpus: 1
      memory: 40Gi
      storage: 80Gi
      gpu_type: nvidia-hopper-h100-3g40gb
      nodes_per_worker: 1
    - gpu_memory_le: 80
      cpus: 20
      gpus: 1
      memory: 80Gi
      storage: 96Gi
      gpu_type: nvidia-hopper-h100
      nodes_per_worker: 1
    - gpu_memory_le: 160
      cpus: 40
      gpus: 2
      memory: 160Gi
      storage: 160Gi
      gpu_type: nvidia-hopper-h100
      nodes_per_worker: 1
    - gpu_memory_le: 320
      cpus: 80
      gpus: 4
      memory: 320Gi
      storage: 320Gi
      gpu_type: nvidia-hopper-h100
      nodes_per_worker: 1
    - gpu_memory_le: 640
      cpus: 160
      gpus: 8
      memory: 800Gi
      storage: 640Gi
      gpu_type: nvidia-hopper-h100
      nodes_per_worker: 1
    - gpu_memory_le: 1280
      cpus: 160
      gpus: 8
      memory: 800Gi
      storage: 900Gi
      gpu_type: nvidia-hopper-h100
      nodes_per_worker: 2
                """,
            "byModelName": """
    - name: llama-3-8b-instruct-262k
      cpus: 40
      gpus: 2
      memory: 160Gi
      storage: 160Gi
      gpu_type: nvidia-hopper-h100
      nodes_per_worker: 1
    - name: deepseek-coder-v2
      cpus: 160
      gpus: 8
      memory: 800Gi
      storage: 640Gi
      gpu_type: nvidia-hopper-h100
      nodes_per_worker: 1
    - name: deepseek-coder-v2-instruct
      cpus: 160
      gpus: 8
      memory: 800Gi
      storage: 640Gi
      gpu_type: nvidia-hopper-h100
      nodes_per_worker: 1
                """,
        }

    return mock.AsyncMock(side_effect=async_mock)
