from typing import Any, Dict, Optional, Sequence

from model_engine_server.common import dict_not_none
from model_engine_server.db.models import Trigger as OrmTrigger
from model_engine_server.domain.entities.trigger_entity import Trigger
from model_engine_server.domain.exceptions import (
    CorruptRecordInfraStateException,
    TriggerNameAlreadyExistsException,
)
from model_engine_server.domain.repositories.trigger_repository import TriggerRepository
from model_engine_server.infra.repositories.db_repository_mixin import (
    DbRepositoryMixin,
    raise_if_read_only,
)
from pydantic import ValidationError
from sqlalchemy.exc import IntegrityError


class DbTriggerRepository(TriggerRepository, DbRepositoryMixin):
    @raise_if_read_only
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
        trigger_record = translate_kwargs_to_trigger_orm(
            name=name,
            created_by=created_by,
            owner=owner,
            cron_schedule=cron_schedule,
            docker_image_batch_job_bundle_id=docker_image_batch_job_bundle_id,
            default_job_config=default_job_config,
            default_job_metadata=default_job_metadata,
        )
        try:
            async with self.session() as session:
                await OrmTrigger.create(session, trigger_record)
                trigger_record = await OrmTrigger.select_by_id(
                    session=session, trigger_id=trigger_record.id
                )
        except IntegrityError:
            raise TriggerNameAlreadyExistsException(
                f"Trigger with name {name} already exists for {owner}"
            )
        return translate_trigger_orm_to_entity(trigger_record)

    async def list_triggers(self, owner: str) -> Sequence[Trigger]:
        async with self.session() as session:
            trigger_records = await OrmTrigger.select_all_by_owner(session=session, owner=owner)
        triggers = [translate_trigger_orm_to_entity(tr) for tr in trigger_records]
        return triggers

    async def get_trigger(self, trigger_id: str) -> Optional[Trigger]:
        async with self.session() as session:
            trigger_record = await OrmTrigger.select_by_id(session=session, trigger_id=trigger_id)
        if not trigger_record:
            return None

        return translate_trigger_orm_to_entity(trigger_record)

    @raise_if_read_only
    async def update_trigger(
        self,
        trigger_id: str,
        cron_schedule: str,
    ) -> bool:
        async with self.session() as session:
            trigger = await OrmTrigger.select_by_id(session=session, trigger_id=trigger_id)
            if trigger is None:
                return False

            await OrmTrigger.update_by_id(
                session=session, trigger_id=trigger_id, kwargs=dict(cron_schedule=cron_schedule)
            )
        return True

    @raise_if_read_only
    async def delete_trigger(
        self,
        trigger_id: str,
    ) -> bool:
        async with self.session() as session:
            trigger = await OrmTrigger.select_by_id(session=session, trigger_id=trigger_id)
            if trigger is None:
                return False

            await OrmTrigger.delete_by_id(session=session, trigger_id=trigger_id)
        return True


def translate_trigger_orm_to_entity(
    trigger_orm: OrmTrigger,
) -> Trigger:
    kwargs = dict_not_none(
        id=trigger_orm.id,
        name=trigger_orm.name,
        owner=trigger_orm.owner,
        created_at=trigger_orm.created_at,
        created_by=trigger_orm.created_by,
        cron_schedule=trigger_orm.cron_schedule,
        docker_image_batch_job_bundle_id=trigger_orm.docker_image_batch_job_bundle_id,
        default_job_config=trigger_orm.default_job_config,
        default_job_metadata=trigger_orm.default_job_metadata,
    )
    try:
        return Trigger.parse_obj(kwargs)
    except ValidationError as exc:
        raise CorruptRecordInfraStateException() from exc


def translate_kwargs_to_trigger_orm(
    name: str,
    created_by: str,
    owner: str,
    cron_schedule: str,
    docker_image_batch_job_bundle_id: str,
    default_job_config: Optional[Dict[str, Any]],
    default_job_metadata: Optional[Dict[str, str]],
) -> OrmTrigger:
    return OrmTrigger(
        name=name,
        owner=owner,
        created_by=created_by,
        cron_schedule=cron_schedule,
        docker_image_batch_job_bundle_id=docker_image_batch_job_bundle_id,
        default_job_config=default_job_config,
        default_job_metadata=default_job_metadata,
    )
