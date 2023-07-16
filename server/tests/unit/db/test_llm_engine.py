from datetime import datetime
from typing import List

import pytest

from llm_engine_server.db.base import SessionAsync
from llm_engine_server.db.models import BatchJob, Bundle, DockerImageBatchJobBundle, Endpoint


@pytest.mark.asyncio
async def test_bundle_select(dbsession_async: SessionAsync, bundles: List[Bundle]):
    bundle_by_name_created_by = await Bundle.select_by_name_created_by(
        dbsession_async, name="test_bundle_1", created_by="test_user_1"
    )
    assert bundle_by_name_created_by is not None

    bundle_by_name_owner = await Bundle.select_by_name_owner(
        dbsession_async, name="test_bundle_1", owner="test_user_1"
    )
    assert bundle_by_name_owner is not None

    bundles_by_name_created_by = await Bundle.select_all_by_name_created_by(
        dbsession_async, name="test_bundle_1", created_by="test_user_1"
    )
    assert len(bundles_by_name_created_by) == 1

    bundles_by_name_owner = await Bundle.select_all_by_name_owner(
        dbsession_async, name="test_bundle_1", owner="test_user_1"
    )
    assert len(bundles_by_name_owner) == 1

    bundle_by_id = await Bundle.select_by_id(dbsession_async, bundle_id=bundles[0].id)
    assert bundle_by_id is not None

    bundles_by_owner = await Bundle.select_all_by_created_by(
        dbsession_async, created_by="test_user_1"
    )
    assert len(bundles_by_owner) == 2


@pytest.mark.asyncio
async def test_bundle_select_delete(dbsession_async: SessionAsync, bundles: List[Bundle]):
    bundles_by_owner = await Bundle.select_all_by_created_by(
        dbsession_async, created_by="test_user_1"
    )
    prev_num_bundles = len(bundles_by_owner)

    await Bundle.delete(dbsession_async, bundles_by_owner[0])

    # After deletion, there should now be 1 fewer bundles for this user.
    bundles_by_owner = await Bundle.select_all_by_created_by(
        dbsession_async, created_by="test_user_1"
    )
    assert len(bundles_by_owner) == prev_num_bundles - 1


@pytest.mark.asyncio
async def test_endpoint_select(
    dbsession_async: SessionAsync, bundles: List[Bundle], endpoints: List[Endpoint]
):
    endpoint_by_name_created_by = await Endpoint.select_by_name_created_by(
        dbsession_async, name="test_endpoint_1", created_by="test_user_1"
    )
    assert endpoint_by_name_created_by is not None

    endpoints_by_created_by = await Endpoint.select_all_by_created_by(
        dbsession_async, created_by="test_user_1"
    )
    assert len(endpoints_by_created_by) == 3

    endpoints_by_owner = await Endpoint.select_all_by_owner(dbsession_async, owner="test_user_1")
    assert len(endpoints_by_owner) == 3

    endpoints_by_bundle_owner = await Endpoint.select_all_by_bundle_created_by(
        dbsession_async, current_bundle_id=bundles[0].id, created_by="test_user_1"
    )
    assert len(endpoints_by_bundle_owner) == 2


@pytest.mark.asyncio
async def test_endpoint_select_delete(
    dbsession_async: SessionAsync, bundles: List[Bundle], endpoints: List[Endpoint]
):
    endpoints_by_user_id = await Endpoint.select_all_by_created_by(
        dbsession_async, created_by="test_user_1"
    )
    prev_num_endpoints = len(endpoints_by_user_id)

    await Endpoint.delete(dbsession_async, endpoints_by_user_id[0])

    # After deletion, there should now be 1 fewer endpoints for this user.
    endpoints_by_user_id = await Endpoint.select_all_by_created_by(
        dbsession_async, created_by="test_user_1"
    )
    assert len(endpoints_by_user_id) == prev_num_endpoints - 1


@pytest.mark.asyncio
async def test_batch_job_select(dbsession_async: SessionAsync, batch_jobs: List[BatchJob]):
    batch_job_by_id = await BatchJob.select_by_id(dbsession_async, batch_job_id=batch_jobs[0].id)
    assert batch_job_by_id is not None

    batch_jobs_by_owner = await BatchJob.select_all_by_owner(dbsession_async, owner="test_user_1")
    assert len(batch_jobs_by_owner) == 2

    batch_jobs_by_owner = await BatchJob.select_all_by_bundle_owner(
        dbsession_async, model_bundle_id=batch_jobs[0].model_bundle_id, owner="test_user_1"
    )
    assert len(batch_jobs_by_owner) == 2


@pytest.mark.asyncio
async def test_batch_job_update(dbsession_async: SessionAsync, batch_jobs: List[BatchJob]):
    update_kwargs = {"status": "FAILED", "completed_at": datetime.now()}
    await BatchJob.update_by_id(
        session=dbsession_async, batch_job_id=batch_jobs[0].id, kwargs=update_kwargs
    )
    batch_job = await BatchJob.select_by_id(dbsession_async, batch_job_id=batch_jobs[0].id)
    assert batch_job is not None
    assert batch_job.batch_job_status == update_kwargs["status"]
    assert batch_job.completed_at.second == update_kwargs["completed_at"].second  # type: ignore


@pytest.mark.asyncio
async def test_docker_image_batch_job_bundle_select(
    dbsession_async: SessionAsync, docker_image_batch_job_bundles: List[DockerImageBatchJobBundle]
):
    batch_job_by_id = await DockerImageBatchJobBundle.select_by_id(
        dbsession_async, batch_bundle_id=docker_image_batch_job_bundles[0].id
    )
    assert batch_job_by_id is not None

    batch_jobs_by_owner = await DockerImageBatchJobBundle.select_all_by_owner(
        dbsession_async, owner="test_user_1"
    )
    assert len(batch_jobs_by_owner) == 2

    batch_jobs_by_owner = await DockerImageBatchJobBundle.select_all_by_name_owner(
        dbsession_async, name=docker_image_batch_job_bundles[0].name, owner="test_user_1"
    )
    assert len(batch_jobs_by_owner) == 2

    batch_jobs_by_owner = await DockerImageBatchJobBundle.select_all_by_name_owner(
        dbsession_async, name=docker_image_batch_job_bundles[2].name, owner="test_user_2"
    )
    assert len(batch_jobs_by_owner) == 1

    batch_job_latest_by_name_owner = await DockerImageBatchJobBundle.select_latest_by_name_owner(
        dbsession_async, name=docker_image_batch_job_bundles[0].name, owner="test_user_1"
    )
    assert batch_job_latest_by_name_owner is not None
    assert batch_job_latest_by_name_owner.id == docker_image_batch_job_bundles[1].id
