import pytest
from model_engine_server.common.dtos.batch_jobs import (
    CreateDockerImageBatchJobBundleV1Request,
    CreateDockerImageBatchJobBundleV1Response,
)
from model_engine_server.common.dtos.model_bundles import ModelBundleOrderBy
from model_engine_server.core.auth.authentication_repository import User
from model_engine_server.core.domain_exceptions import (
    ObjectNotAuthorizedException,
    ObjectNotFoundException,
)
from model_engine_server.domain.repositories import DockerImageBatchJobBundleRepository
from model_engine_server.domain.use_cases.docker_image_batch_job_bundle_use_cases import (
    CreateDockerImageBatchJobBundleV1UseCase,
    GetDockerImageBatchJobBundleByIdV1UseCase,
    GetLatestDockerImageBatchJobBundleByNameV1UseCase,
    ListDockerImageBatchJobBundleV1UseCase,
)


@pytest.mark.asyncio
async def test_create_docker_image_batch_job_bundle_use_case(
    test_api_key: str,
    fake_docker_image_batch_job_bundle_repository: DockerImageBatchJobBundleRepository,
    create_docker_image_batch_job_bundle_request: CreateDockerImageBatchJobBundleV1Request,
):
    use_case = CreateDockerImageBatchJobBundleV1UseCase(
        docker_image_batch_job_bundle_repo=fake_docker_image_batch_job_bundle_repository
    )
    user = User(user_id=test_api_key, team_id=test_api_key, is_privileged_user=True)
    response = await use_case.execute(
        user=user, request=create_docker_image_batch_job_bundle_request
    )
    assert response.docker_image_batch_job_bundle_id
    assert isinstance(response, CreateDockerImageBatchJobBundleV1Response)


@pytest.mark.asyncio
async def test_create_list_docker_image_batch_job_bundle_use_case(
    test_api_key: str,
    fake_docker_image_batch_job_bundle_repository: DockerImageBatchJobBundleRepository,
    create_docker_image_batch_job_bundle_request: CreateDockerImageBatchJobBundleV1Request,
):
    use_case_list = ListDockerImageBatchJobBundleV1UseCase(
        docker_image_batch_job_bundle_repo=fake_docker_image_batch_job_bundle_repository
    )
    user = User(user_id=test_api_key, team_id=test_api_key, is_privileged_user=True)
    response = await use_case_list.execute(user=user, bundle_name="nonexistent", order_by=None)
    assert len(response.docker_image_batch_job_bundles) == 0

    # Create a bundle
    use_case_create = CreateDockerImageBatchJobBundleV1UseCase(
        docker_image_batch_job_bundle_repo=fake_docker_image_batch_job_bundle_repository
    )
    create_response = await use_case_create.execute(
        user=user, request=create_docker_image_batch_job_bundle_request
    )
    response_2 = await use_case_list.execute(
        user=user, bundle_name=create_docker_image_batch_job_bundle_request.name, order_by=None
    )
    assert len(response_2.docker_image_batch_job_bundles) == 1
    assert (
        response_2.docker_image_batch_job_bundles[0].id
        == create_response.docker_image_batch_job_bundle_id
    )

    # Create two more bundles, one with a different name
    await use_case_create.execute(user=user, request=create_docker_image_batch_job_bundle_request)

    new_request = create_docker_image_batch_job_bundle_request.copy()
    new_request.name = "new_name"

    await use_case_create.execute(user=user, request=new_request)
    response_3 = await use_case_list.execute(
        user=user,
        bundle_name=create_docker_image_batch_job_bundle_request.name,
        order_by=ModelBundleOrderBy.OLDEST,
    )
    assert len(response_3.docker_image_batch_job_bundles) == 2
    response_4 = await use_case_list.execute(
        user=user, bundle_name=new_request.name, order_by=ModelBundleOrderBy.OLDEST
    )
    assert len(response_4.docker_image_batch_job_bundles) == 1
    response_5 = await use_case_list.execute(
        user=user, bundle_name=None, order_by=ModelBundleOrderBy.OLDEST
    )
    assert len(response_5.docker_image_batch_job_bundles) == 3


@pytest.mark.asyncio
async def test_create_list_docker_image_batch_job_bundle_team_use_case(
    test_api_key: str,
    fake_docker_image_batch_job_bundle_repository: DockerImageBatchJobBundleRepository,
    create_docker_image_batch_job_bundle_request: CreateDockerImageBatchJobBundleV1Request,
    test_api_key_user_on_other_team: str,
    test_api_key_user_on_other_team_2: str,
    test_api_key_team: str,
):
    use_case_list = ListDockerImageBatchJobBundleV1UseCase(
        docker_image_batch_job_bundle_repo=fake_docker_image_batch_job_bundle_repository
    )
    user = User(user_id=test_api_key, team_id=test_api_key, is_privileged_user=True)
    user_other_team_1 = User(
        user_id=test_api_key_user_on_other_team, team_id=test_api_key_team, is_privileged_user=True
    )
    user_other_team_2 = User(
        user_id=test_api_key_user_on_other_team_2,
        team_id=test_api_key_team,
        is_privileged_user=True,
    )
    response = await use_case_list.execute(user=user, bundle_name="nonexistent", order_by=None)
    assert len(response.docker_image_batch_job_bundles) == 0

    # Create a bundle for a given user, check it appears in this user's list and no one else's
    use_case_create = CreateDockerImageBatchJobBundleV1UseCase(
        docker_image_batch_job_bundle_repo=fake_docker_image_batch_job_bundle_repository
    )
    await use_case_create.execute(user=user, request=create_docker_image_batch_job_bundle_request)
    response_2 = await use_case_list.execute(
        user=user, bundle_name=create_docker_image_batch_job_bundle_request.name, order_by=None
    )
    assert len(response_2.docker_image_batch_job_bundles) == 1
    response_3 = await use_case_list.execute(
        user=user_other_team_1,
        bundle_name=create_docker_image_batch_job_bundle_request.name,
        order_by=ModelBundleOrderBy.NEWEST,
    )
    assert len(response_3.docker_image_batch_job_bundles) == 0

    # Create a bundle each for the two other uesrs' teams
    await use_case_create.execute(
        user=user_other_team_1, request=create_docker_image_batch_job_bundle_request
    )
    await use_case_create.execute(
        user=user_other_team_2, request=create_docker_image_batch_job_bundle_request
    )
    response_4 = await use_case_list.execute(
        user=user,
        bundle_name=create_docker_image_batch_job_bundle_request.name,
        order_by=ModelBundleOrderBy.OLDEST,
    )
    assert len(response_4.docker_image_batch_job_bundles) == 1
    response_5 = await use_case_list.execute(
        user=user_other_team_1,
        bundle_name=create_docker_image_batch_job_bundle_request.name,
        order_by=None,
    )
    assert len(response_5.docker_image_batch_job_bundles) == 2


@pytest.mark.asyncio
async def test_create_get_docker_image_batch_job_bundle_by_id_success_use_case(
    test_api_key: str,
    fake_docker_image_batch_job_bundle_repository: DockerImageBatchJobBundleRepository,
    create_docker_image_batch_job_bundle_request: CreateDockerImageBatchJobBundleV1Request,
):
    user = User(user_id=test_api_key, team_id=test_api_key, is_privileged_user=True)
    use_case_create = CreateDockerImageBatchJobBundleV1UseCase(
        docker_image_batch_job_bundle_repo=fake_docker_image_batch_job_bundle_repository
    )
    use_case_get_by_id = GetDockerImageBatchJobBundleByIdV1UseCase(
        docker_image_batch_job_bundle_repo=fake_docker_image_batch_job_bundle_repository
    )
    response_1 = await use_case_create.execute(
        user=user, request=create_docker_image_batch_job_bundle_request
    )
    batch_bundle_id = response_1.docker_image_batch_job_bundle_id
    response_2 = await use_case_get_by_id.execute(
        user=user, docker_image_batch_job_bundle_id=batch_bundle_id
    )
    assert response_2.id == batch_bundle_id


@pytest.mark.asyncio
async def test_get_docker_image_batch_job_bundle_by_id_not_found_use_case(
    test_api_key: str,
    fake_docker_image_batch_job_bundle_repository: DockerImageBatchJobBundleRepository,
    create_docker_image_batch_job_bundle_request: CreateDockerImageBatchJobBundleV1Request,
):
    user = User(user_id=test_api_key, team_id=test_api_key, is_privileged_user=True)
    use_case_create = CreateDockerImageBatchJobBundleV1UseCase(
        docker_image_batch_job_bundle_repo=fake_docker_image_batch_job_bundle_repository
    )
    use_case_get_by_id = GetDockerImageBatchJobBundleByIdV1UseCase(
        docker_image_batch_job_bundle_repo=fake_docker_image_batch_job_bundle_repository
    )
    response_1 = await use_case_create.execute(
        user=user, request=create_docker_image_batch_job_bundle_request
    )
    batch_bundle_id = response_1.docker_image_batch_job_bundle_id
    nonexistent_id = f"not_{batch_bundle_id}"
    with pytest.raises(ObjectNotFoundException):
        await use_case_get_by_id.execute(user=user, docker_image_batch_job_bundle_id=nonexistent_id)


@pytest.mark.asyncio
async def test_create_get_docker_image_batch_job_bundle_by_id_unauthorized_use_case(
    test_api_key: str,
    fake_docker_image_batch_job_bundle_repository: DockerImageBatchJobBundleRepository,
    create_docker_image_batch_job_bundle_request: CreateDockerImageBatchJobBundleV1Request,
    test_api_key_user_on_other_team: str,
    test_api_key_team: str,
):
    user = User(user_id=test_api_key, team_id=test_api_key, is_privileged_user=True)
    user_other_team_1 = User(
        user_id=test_api_key_user_on_other_team, team_id=test_api_key_team, is_privileged_user=True
    )
    use_case_create = CreateDockerImageBatchJobBundleV1UseCase(
        docker_image_batch_job_bundle_repo=fake_docker_image_batch_job_bundle_repository
    )
    use_case_get_by_id = GetDockerImageBatchJobBundleByIdV1UseCase(
        docker_image_batch_job_bundle_repo=fake_docker_image_batch_job_bundle_repository
    )
    response_1 = await use_case_create.execute(
        user=user, request=create_docker_image_batch_job_bundle_request
    )
    batch_bundle_id = response_1.docker_image_batch_job_bundle_id
    with pytest.raises(ObjectNotAuthorizedException):
        await use_case_get_by_id.execute(
            user=user_other_team_1, docker_image_batch_job_bundle_id=batch_bundle_id
        )


@pytest.mark.asyncio
async def test_create_get_latest_docker_image_batch_job_bundle_by_name_success_use_case(
    test_api_key: str,
    fake_docker_image_batch_job_bundle_repository: DockerImageBatchJobBundleRepository,
    create_docker_image_batch_job_bundle_request: CreateDockerImageBatchJobBundleV1Request,
):
    user = User(user_id=test_api_key, team_id=test_api_key, is_privileged_user=True)
    use_case_create = CreateDockerImageBatchJobBundleV1UseCase(
        docker_image_batch_job_bundle_repo=fake_docker_image_batch_job_bundle_repository
    )
    use_case_get_by_name = GetLatestDockerImageBatchJobBundleByNameV1UseCase(
        docker_image_batch_job_bundle_repo=fake_docker_image_batch_job_bundle_repository
    )
    await use_case_create.execute(user=user, request=create_docker_image_batch_job_bundle_request)
    bun_name = create_docker_image_batch_job_bundle_request.name
    response = await use_case_get_by_name.execute(user=user, bundle_name=bun_name)
    assert response.name == bun_name


@pytest.mark.asyncio
async def test_create_get_latest_docker_image_batch_job_bundle_by_name_not_found_use_case(
    test_api_key: str,
    fake_docker_image_batch_job_bundle_repository: DockerImageBatchJobBundleRepository,
    create_docker_image_batch_job_bundle_request: CreateDockerImageBatchJobBundleV1Request,
):
    user = User(user_id=test_api_key, team_id=test_api_key, is_privileged_user=True)
    use_case_create = CreateDockerImageBatchJobBundleV1UseCase(
        docker_image_batch_job_bundle_repo=fake_docker_image_batch_job_bundle_repository
    )
    use_case_get_by_name = GetLatestDockerImageBatchJobBundleByNameV1UseCase(
        docker_image_batch_job_bundle_repo=fake_docker_image_batch_job_bundle_repository
    )
    await use_case_create.execute(user=user, request=create_docker_image_batch_job_bundle_request)
    bun_name = create_docker_image_batch_job_bundle_request.name
    other_name = f"not_{bun_name}"
    with pytest.raises(ObjectNotFoundException):
        await use_case_get_by_name.execute(user=user, bundle_name=other_name)
