from typing import Dict, List, Optional, Sequence

from pydantic.error_wrappers import ValidationError

from llm_engine_server.common import dict_not_none
from llm_engine_server.common.dtos.model_bundles import ModelBundleOrderBy
from llm_engine_server.db.models import DockerImageBatchJobBundle as OrmDockerImageBatchJobBundle
from llm_engine_server.domain.entities import GpuType
from llm_engine_server.domain.entities.docker_image_batch_job_bundle_entity import (
    DockerImageBatchJobBundle,
)
from llm_engine_server.domain.exceptions import CorruptRecordInfraStateException
from llm_engine_server.domain.repositories.docker_image_batch_job_bundle_repository import (
    DockerImageBatchJobBundleRepository,
)
from llm_engine_server.infra.repositories.db_repository_mixin import (
    DbRepositoryMixin,
    raise_if_read_only,
)


class DbDockerImageBatchJobBundleRepository(DockerImageBatchJobBundleRepository, DbRepositoryMixin):
    @raise_if_read_only
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
    ) -> DockerImageBatchJobBundle:
        docker_image_batch_job_record = translate_kwargs_to_batch_bundle_orm(
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
        )
        async with self.session() as session:
            await OrmDockerImageBatchJobBundle.create(session, docker_image_batch_job_record)
            docker_image_batch_job_record = await OrmDockerImageBatchJobBundle.select_by_id(
                session=session, batch_bundle_id=docker_image_batch_job_record.id
            )
        return translate_docker_image_batch_job_bundle_orm_to_entity(docker_image_batch_job_record)

    async def list_docker_image_batch_job_bundles(
        self, owner: str, name: Optional[str], order_by: Optional[ModelBundleOrderBy]
    ) -> Sequence[DockerImageBatchJobBundle]:
        async with self.session() as session:
            if name is not None:
                batch_bundle_records = await OrmDockerImageBatchJobBundle.select_all_by_name_owner(
                    session=session, name=name, owner=owner
                )
            else:
                batch_bundle_records = await OrmDockerImageBatchJobBundle.select_all_by_owner(
                    session=session, owner=owner
                )
        batch_bundles = [
            translate_docker_image_batch_job_bundle_orm_to_entity(bb) for bb in batch_bundle_records
        ]

        # TODO: we could use an ORDER_BY operation in the DB instead.
        if order_by == ModelBundleOrderBy.NEWEST:
            batch_bundles.sort(key=lambda x: x.created_at, reverse=True)
        elif order_by == ModelBundleOrderBy.OLDEST:
            batch_bundles.sort(key=lambda x: x.created_at, reverse=False)

        return batch_bundles

    async def get_docker_image_batch_job_bundle(
        self, docker_image_batch_job_bundle_id: str
    ) -> Optional[DockerImageBatchJobBundle]:
        async with self.session() as session:
            batch_bundle_record = await OrmDockerImageBatchJobBundle.select_by_id(
                session=session, batch_bundle_id=docker_image_batch_job_bundle_id
            )
        if not batch_bundle_record:
            return None

        return translate_docker_image_batch_job_bundle_orm_to_entity(batch_bundle_record)

    async def get_latest_docker_image_batch_job_bundle(
        self, owner: str, name: str
    ) -> Optional[DockerImageBatchJobBundle]:
        async with self.session() as session:
            batch_bundle_record = await OrmDockerImageBatchJobBundle.select_latest_by_name_owner(
                session=session, name=name, owner=owner
            )
        if not batch_bundle_record:
            return None
        return translate_docker_image_batch_job_bundle_orm_to_entity(batch_bundle_record)


def translate_docker_image_batch_job_bundle_orm_to_entity(
    batch_bundle_orm: OrmDockerImageBatchJobBundle,
) -> DockerImageBatchJobBundle:
    kwargs = dict_not_none(
        id=batch_bundle_orm.id,
        created_at=batch_bundle_orm.created_at,
        name=batch_bundle_orm.name,
        created_by=batch_bundle_orm.created_by,
        owner=batch_bundle_orm.owner,
        image_repository=batch_bundle_orm.image_repository,
        image_tag=batch_bundle_orm.image_tag,
        command=batch_bundle_orm.command,
        env=batch_bundle_orm.env,
        mount_location=batch_bundle_orm.mount_location,
        cpus=batch_bundle_orm.cpus,
        memory=batch_bundle_orm.memory,
        storage=batch_bundle_orm.storage,
        gpus=batch_bundle_orm.gpus,
        gpu_type=batch_bundle_orm.gpu_type,
    )
    try:
        return DockerImageBatchJobBundle.parse_obj(kwargs)  # auto-translates str to int/GpuType
    except ValidationError as exc:
        raise CorruptRecordInfraStateException() from exc


def translate_kwargs_to_batch_bundle_orm(
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
) -> OrmDockerImageBatchJobBundle:
    return OrmDockerImageBatchJobBundle(
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
    )
