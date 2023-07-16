from datetime import datetime
from typing import Any, Dict, List, Optional

from spellbook_serve.common import dict_not_none
from spellbook_serve.db.models import BatchJob as OrmBatchJob
from spellbook_serve.domain.entities import BatchJobRecord, BatchJobStatus
from spellbook_serve.infra.repositories.batch_job_record_repository import BatchJobRecordRepository
from spellbook_serve.infra.repositories.db_model_bundle_repository import (
    translate_model_bundle_orm_to_model_bundle,
)
from spellbook_serve.infra.repositories.db_repository_mixin import (
    DbRepositoryMixin,
    raise_if_read_only,
)


def translate_batch_job_orm_to_batch_job_record(batch_job_orm: OrmBatchJob) -> BatchJobRecord:
    return BatchJobRecord(
        id=batch_job_orm.id,
        created_at=batch_job_orm.created_at,
        completed_at=batch_job_orm.completed_at,
        status=batch_job_orm.batch_job_status,
        created_by=batch_job_orm.created_by,
        owner=batch_job_orm.owner,
        model_bundle=translate_model_bundle_orm_to_model_bundle(batch_job_orm.model_bundle),
        model_endpoint_id=batch_job_orm.model_endpoint_id,
        task_ids_location=batch_job_orm.task_ids_location,
        result_location=batch_job_orm.result_location,
    )


class DbBatchJobRecordRepository(BatchJobRecordRepository, DbRepositoryMixin):
    """
    Implements the BatchJobRecordRepository interface using a relational database.
    """

    @raise_if_read_only
    async def create_batch_job_record(
        self,
        *,
        status: BatchJobStatus,
        created_by: str,
        owner: str,
        model_bundle_id: str,
    ) -> BatchJobRecord:
        batch_job = OrmBatchJob(
            batch_job_status=status,
            created_by=created_by,
            owner=owner,
            model_bundle_id=model_bundle_id,
        )
        async with self.session() as session:
            await OrmBatchJob.create(session, batch_job)
            # HACK: Force a select_by_id to load the model_bundle relationship into the current
            # session. Otherwise, we'll get an error like:
            # sqlalchemy.orm.exc.DetachedInstanceError: Parent instance <BatchJob at 0x7fb1cb40d9a0>
            #   is not bound to a Session; lazy load operation of attribute 'model_bundle' cannot
            #   proceed.
            # This is because there is no bound session to this ORM base model, and thus lazy
            # loading cannot occur without a session to execute the SELECT query.
            # See: https://docs.sqlalchemy.org/en/14/orm/loading_relationships.html
            batch_job_orm = await OrmBatchJob.select_by_id(
                session=session,
                batch_job_id=batch_job.id,
            )
        return translate_batch_job_orm_to_batch_job_record(batch_job_orm)

    async def list_batch_job_records(self, owner: Optional[str]) -> List[BatchJobRecord]:
        async with self.session() as session:
            batch_jobs_orm = await OrmBatchJob.select_all_by_owner(session=session, owner=owner)
        batch_jobs = [translate_batch_job_orm_to_batch_job_record(b) for b in batch_jobs_orm]
        return batch_jobs

    @raise_if_read_only
    async def unset_model_endpoint_id(self, batch_job_id: str) -> Optional[BatchJobRecord]:
        async with self.session() as session:
            batch_job_orm = await OrmBatchJob.select_by_id(
                session=session, batch_job_id=batch_job_id
            )
            if batch_job_orm is None:
                return None

            await OrmBatchJob.update_by_id(
                session=session,
                batch_job_id=batch_job_id,
                kwargs={"model_endpoint_id": None},
            )

        async with self.session() as session:
            updated_batch_job_orm = await OrmBatchJob.select_by_id(
                session=session, batch_job_id=batch_job_id
            )

        return translate_batch_job_orm_to_batch_job_record(updated_batch_job_orm)

    @raise_if_read_only
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
        async with self.session() as session:
            batch_job_orm = await OrmBatchJob.select_by_id(
                session=session, batch_job_id=batch_job_id
            )
            if batch_job_orm is None:
                return None

            update_kwargs: Dict[str, Any] = dict_not_none(
                status=status,
                model_endpoint_id=model_endpoint_id,
                task_ids_location=task_ids_location,
                result_location=result_location,
                completed_at=completed_at,
            )
            await OrmBatchJob.update_by_id(
                session=session,
                batch_job_id=batch_job_id,
                kwargs=update_kwargs,
            )

        async with self.session() as session:
            updated_batch_job_orm = await OrmBatchJob.select_by_id(
                session=session, batch_job_id=batch_job_id
            )

        return translate_batch_job_orm_to_batch_job_record(updated_batch_job_orm)

    async def get_batch_job_record(self, batch_job_id: str) -> Optional[BatchJobRecord]:
        async with self.session() as session:
            batch_job_orm = await OrmBatchJob.select_by_id(
                session=session, batch_job_id=batch_job_id
            )
            if batch_job_orm is None:
                return None

        return translate_batch_job_orm_to_batch_job_record(batch_job_orm)
