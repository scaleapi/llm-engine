import pytest
from llm_engine_server.common.dtos.model_bundles import (
    CloneModelBundleV1Request,
    CreateModelBundleV1Request,
    CreateModelBundleV1Response,
    CreateModelBundleV2Request,
    CreateModelBundleV2Response,
    ListModelBundlesV1Response,
    ModelBundleOrderBy,
    ModelBundleV1Response,
)
from llm_engine_server.core.auth.authentication_repository import User
from llm_engine_server.core.domain_exceptions import (
    DockerImageNotFoundException,
    ObjectNotAuthorizedException,
    ObjectNotFoundException,
)
from llm_engine_server.domain.gateways import ModelPrimitiveGateway
from llm_engine_server.domain.repositories import DockerRepository, ModelBundleRepository
from llm_engine_server.domain.use_cases.model_bundle_use_cases import (
    CloneModelBundleV1UseCase,
    CreateModelBundleV1UseCase,
    CreateModelBundleV2UseCase,
    GetLatestModelBundleByNameV1UseCase,
    GetModelBundleByIdV1UseCase,
    ListModelBundlesV1UseCase,
)


@pytest.mark.asyncio
async def test_create_model_bundle_use_case_success(
    test_api_key: str,
    fake_model_bundle_repository: ModelBundleRepository,
    create_model_bundle_request: CreateModelBundleV1Request,
    fake_docker_repository_image_always_exists: DockerRepository,
    fake_model_primitive_gateway: ModelPrimitiveGateway,
):
    use_case = CreateModelBundleV1UseCase(
        model_bundle_repository=fake_model_bundle_repository,
        docker_repository=fake_docker_repository_image_always_exists,
        model_primitive_gateway=fake_model_primitive_gateway,
    )
    user = User(user_id=test_api_key, team_id=test_api_key, is_privileged_user=True)
    response = await use_case.execute(user=user, request=create_model_bundle_request)
    assert response.model_bundle_id
    assert isinstance(response, CreateModelBundleV1Response)


@pytest.mark.asyncio
async def test_create_model_bundle_use_case_custom_not_authorized(
    test_api_key: str,
    fake_model_bundle_repository: ModelBundleRepository,
    create_model_bundle_request: CreateModelBundleV1Request,
    fake_docker_repository_image_always_exists: DockerRepository,
    fake_model_primitive_gateway: ModelPrimitiveGateway,
):
    use_case = CreateModelBundleV1UseCase(
        model_bundle_repository=fake_model_bundle_repository,
        docker_repository=fake_docker_repository_image_always_exists,
        model_primitive_gateway=fake_model_primitive_gateway,
    )
    user = User(user_id=test_api_key, team_id=test_api_key, is_privileged_user=False)
    with pytest.raises(ObjectNotAuthorizedException):
        await use_case.execute(user=user, request=create_model_bundle_request)


@pytest.mark.asyncio
async def test_create_model_bundle_use_case_docker_not_found(
    test_api_key: str,
    fake_model_bundle_repository: ModelBundleRepository,
    create_model_bundle_request: CreateModelBundleV1Request,
    fake_docker_repository_image_never_exists: DockerRepository,
    fake_model_primitive_gateway: ModelPrimitiveGateway,
):
    use_case = CreateModelBundleV1UseCase(
        model_bundle_repository=fake_model_bundle_repository,
        docker_repository=fake_docker_repository_image_never_exists,
        model_primitive_gateway=fake_model_primitive_gateway,
    )
    user = User(user_id=test_api_key, team_id=test_api_key, is_privileged_user=True)
    with pytest.raises(DockerImageNotFoundException):
        await use_case.execute(user=user, request=create_model_bundle_request)


@pytest.mark.asyncio
async def test_create_get_model_bundle_use_case_success(
    test_api_key: str,
    fake_model_bundle_repository: ModelBundleRepository,
    create_model_bundle_request: CreateModelBundleV1Request,
    fake_docker_repository_image_always_exists: DockerRepository,
    fake_model_primitive_gateway: ModelPrimitiveGateway,
):
    use_case_1 = CreateModelBundleV1UseCase(
        model_bundle_repository=fake_model_bundle_repository,
        docker_repository=fake_docker_repository_image_always_exists,
        model_primitive_gateway=fake_model_primitive_gateway,
    )
    user = User(user_id=test_api_key, team_id=test_api_key, is_privileged_user=True)
    response_1 = await use_case_1.execute(user=user, request=create_model_bundle_request)

    use_case_2 = GetModelBundleByIdV1UseCase(model_bundle_repository=fake_model_bundle_repository)
    response_2 = await use_case_2.execute(user=user, model_bundle_id=response_1.model_bundle_id)

    assert response_2.id == response_1.model_bundle_id
    assert isinstance(response_2, ModelBundleV1Response)


@pytest.mark.asyncio
async def test_create_get_model_bundle_use_case_raises(
    test_api_key: str,
    fake_model_bundle_repository: ModelBundleRepository,
    create_model_bundle_request: CreateModelBundleV1Request,
    fake_docker_repository_image_always_exists: DockerRepository,
    fake_model_primitive_gateway: ModelPrimitiveGateway,
):
    use_case_1 = CreateModelBundleV1UseCase(
        model_bundle_repository=fake_model_bundle_repository,
        docker_repository=fake_docker_repository_image_always_exists,
        model_primitive_gateway=fake_model_primitive_gateway,
    )
    user = User(user_id=test_api_key, team_id=test_api_key, is_privileged_user=True)
    response_1 = await use_case_1.execute(user=user, request=create_model_bundle_request)

    use_case_2 = GetModelBundleByIdV1UseCase(model_bundle_repository=fake_model_bundle_repository)
    user_unauth = User(user_id="not_authorized", team_id="not_authorized", is_privileged_user=True)
    with pytest.raises(ObjectNotAuthorizedException):
        await use_case_2.execute(user=user_unauth, model_bundle_id=response_1.model_bundle_id)

    with pytest.raises(ObjectNotFoundException):
        await use_case_2.execute(user=user, model_bundle_id="invalid_model_bundle_id")


@pytest.mark.asyncio
async def test_clone_model(
    test_api_key: str,
    test_api_key_2: str,
    fake_model_bundle_repository: ModelBundleRepository,
    create_model_bundle_request: CreateModelBundleV1Request,
    fake_docker_repository_image_always_exists: DockerRepository,
    fake_model_primitive_gateway: ModelPrimitiveGateway,
):
    create_use_case = CreateModelBundleV1UseCase(
        model_bundle_repository=fake_model_bundle_repository,
        docker_repository=fake_docker_repository_image_always_exists,
        model_primitive_gateway=fake_model_primitive_gateway,
    )
    user1 = User(user_id=test_api_key, team_id=test_api_key, is_privileged_user=True)
    create_response = await create_use_case.execute(user=user1, request=create_model_bundle_request)
    original_model_bundle_id = create_response.model_bundle_id

    clone_request = CloneModelBundleV1Request(
        original_model_bundle_id=original_model_bundle_id,
        new_app_config={"foo": "bar"},
    )

    # Make sure that a user on the same team as the original bundle creator can clone the bundle.
    user2 = User(user_id=test_api_key_2, team_id=test_api_key, is_privileged_user=True)
    clone_use_case = CloneModelBundleV1UseCase(model_bundle_repository=fake_model_bundle_repository)
    clone_response = await clone_use_case.execute(user=user2, request=clone_request)

    assert clone_response.model_bundle_id != original_model_bundle_id

    original_bundle = await fake_model_bundle_repository.get_model_bundle(
        model_bundle_id=original_model_bundle_id
    )
    new_bundle = await fake_model_bundle_repository.get_model_bundle(
        model_bundle_id=clone_response.model_bundle_id
    )

    # More to make mypy happy
    assert original_bundle is not None
    assert new_bundle is not None

    # These fields should be different in the new bundle.
    assert new_bundle.app_config == {"foo": "bar"}
    assert new_bundle.created_by == user2.user_id
    expected_new_metadata = original_bundle.metadata.copy()
    expected_new_metadata["__cloned_from_bundle_id"] = original_model_bundle_id
    assert new_bundle.metadata == expected_new_metadata

    # These fields should be the same as the original bundle.
    assert new_bundle.name == original_bundle.name
    assert new_bundle.location == original_bundle.location
    assert new_bundle.requirements == original_bundle.requirements
    assert new_bundle.env_params == original_bundle.env_params
    assert new_bundle.packaging_type == original_bundle.packaging_type
    assert new_bundle.model_artifact_ids == original_bundle.model_artifact_ids
    assert new_bundle.schema_location == original_bundle.schema_location
    assert new_bundle.owner == original_bundle.owner


@pytest.mark.asyncio
async def test_clone_model_raises_unauthorized(
    test_api_key: str,
    test_api_key_2: str,
    fake_model_bundle_repository: ModelBundleRepository,
    create_model_bundle_request: CreateModelBundleV1Request,
    fake_docker_repository_image_always_exists: DockerRepository,
    fake_model_primitive_gateway: ModelPrimitiveGateway,
):
    create_use_case = CreateModelBundleV1UseCase(
        model_bundle_repository=fake_model_bundle_repository,
        docker_repository=fake_docker_repository_image_always_exists,
        model_primitive_gateway=fake_model_primitive_gateway,
    )
    user1 = User(user_id=test_api_key, team_id=test_api_key, is_privileged_user=True)
    create_response = await create_use_case.execute(user=user1, request=create_model_bundle_request)
    original_model_bundle_id = create_response.model_bundle_id

    clone_request = CloneModelBundleV1Request(
        original_model_bundle_id=original_model_bundle_id,
        new_app_config={"foo": "bar"},
    )

    # User2 is on a different team, not authorized.
    user2 = User(user_id=test_api_key_2, team_id="some other team", is_privileged_user=True)
    clone_use_case = CloneModelBundleV1UseCase(model_bundle_repository=fake_model_bundle_repository)

    with pytest.raises(ObjectNotAuthorizedException):
        await clone_use_case.execute(user=user2, request=clone_request)


async def test_clone_model_raises_not_found(
    test_api_key: str,
    test_api_key_2: str,
    fake_model_bundle_repository: ModelBundleRepository,
    create_model_bundle_request: CreateModelBundleV1Request,
    fake_docker_repository_image_always_exists: DockerRepository,
    fake_model_primitive_gateway: ModelPrimitiveGateway,
):
    create_use_case = CreateModelBundleV1UseCase(
        model_bundle_repository=fake_model_bundle_repository,
        docker_repository=fake_docker_repository_image_always_exists,
        model_primitive_gateway=fake_model_primitive_gateway,
    )
    user1 = User(user_id=test_api_key, team_id=test_api_key, is_privileged_user=True)
    await create_use_case.execute(user=user1, request=create_model_bundle_request)

    clone_request = CloneModelBundleV1Request(
        original_model_bundle_id="unknown bundle id",
        new_app_config={"foo": "bar"},
    )
    clone_use_case = CloneModelBundleV1UseCase(model_bundle_repository=fake_model_bundle_repository)

    with pytest.raises(ObjectNotFoundException):
        await clone_use_case.execute(user=user1, request=clone_request)


@pytest.mark.asyncio
async def test_create_get_latest_model_bundle_use_case_success(
    test_api_key: str,
    fake_model_bundle_repository: ModelBundleRepository,
    create_model_bundle_request: CreateModelBundleV1Request,
    fake_docker_repository_image_always_exists: DockerRepository,
    fake_model_primitive_gateway: ModelPrimitiveGateway,
):
    use_case_1 = CreateModelBundleV1UseCase(
        model_bundle_repository=fake_model_bundle_repository,
        docker_repository=fake_docker_repository_image_always_exists,
        model_primitive_gateway=fake_model_primitive_gateway,
    )
    user = User(user_id=test_api_key, team_id=test_api_key, is_privileged_user=True)
    response_1 = await use_case_1.execute(user=user, request=create_model_bundle_request)

    use_case_2 = GetLatestModelBundleByNameV1UseCase(
        model_bundle_repository=fake_model_bundle_repository
    )
    response_2 = await use_case_2.execute(user=user, model_name=create_model_bundle_request.name)

    assert response_2.id == response_1.model_bundle_id
    assert isinstance(response_2, ModelBundleV1Response)


@pytest.mark.asyncio
async def test_create_get_latest_model_bundle_use_case_raises(
    test_api_key: str,
    fake_model_bundle_repository: ModelBundleRepository,
    create_model_bundle_request: CreateModelBundleV1Request,
    fake_docker_repository_image_always_exists: DockerRepository,
    fake_model_primitive_gateway: ModelPrimitiveGateway,
):
    use_case_1 = CreateModelBundleV1UseCase(
        model_bundle_repository=fake_model_bundle_repository,
        docker_repository=fake_docker_repository_image_always_exists,
        model_primitive_gateway=fake_model_primitive_gateway,
    )
    user = User(user_id=test_api_key, team_id=test_api_key, is_privileged_user=True)
    await use_case_1.execute(user=user, request=create_model_bundle_request)

    use_case_2 = GetLatestModelBundleByNameV1UseCase(
        model_bundle_repository=fake_model_bundle_repository
    )
    user_invalid = User(user_id="invalid_user", team_id="invalid_team", is_privileged_user=True)
    with pytest.raises(ObjectNotFoundException):
        await use_case_2.execute(user=user_invalid, model_name=create_model_bundle_request.name)

    with pytest.raises(ObjectNotFoundException):
        await use_case_2.execute(user=user, model_name="invalid_model_name")


@pytest.mark.asyncio
async def test_create_list_model_bundles(
    test_api_key: str,
    fake_model_bundle_repository: ModelBundleRepository,
    create_model_bundle_request: CreateModelBundleV1Request,
    fake_docker_repository_image_always_exists: DockerRepository,
    fake_model_primitive_gateway: ModelPrimitiveGateway,
):
    # Initially there should be 0 model bundles.
    use_case_1 = ListModelBundlesV1UseCase(model_bundle_repository=fake_model_bundle_repository)
    user = User(user_id=test_api_key, team_id=test_api_key, is_privileged_user=True)
    response_1 = await use_case_1.execute(
        user=user, model_name=create_model_bundle_request.name, order_by=None
    )
    assert isinstance(response_1, ListModelBundlesV1Response)
    assert response_1.model_bundles == []

    # Create 1 model bundle.
    use_case_2 = CreateModelBundleV1UseCase(
        model_bundle_repository=fake_model_bundle_repository,
        docker_repository=fake_docker_repository_image_always_exists,
        model_primitive_gateway=fake_model_primitive_gateway,
    )
    await use_case_2.execute(user=user, request=create_model_bundle_request)

    # Now listing model bundles should yield 1 model bundle.
    response_2 = await use_case_1.execute(
        user=user,
        model_name=create_model_bundle_request.name,
        order_by=ModelBundleOrderBy.OLDEST,
    )
    assert len(response_2.model_bundles) == 1

    # Create 2 more model bundles.
    await use_case_2.execute(user=user, request=create_model_bundle_request)
    await use_case_2.execute(user=user, request=create_model_bundle_request)

    # Now listing model bundles should yield 3 model bundles.
    response_3 = await use_case_1.execute(
        user=user,
        model_name=create_model_bundle_request.name,
        order_by=ModelBundleOrderBy.NEWEST,
    )
    assert len(response_3.model_bundles) == 3


@pytest.mark.asyncio
async def test_create_list_model_bundles_team(
    fake_model_bundle_repository: ModelBundleRepository,
    create_model_bundle_request: CreateModelBundleV1Request,
    fake_docker_repository_image_always_exists: DockerRepository,
    fake_model_primitive_gateway: ModelPrimitiveGateway,
    test_api_key_user_on_other_team: str,
    test_api_key_user_on_other_team_2: str,
    test_api_key_team: str,
):
    user_1 = User(
        user_id=test_api_key_user_on_other_team, team_id=test_api_key_team, is_privileged_user=True
    )
    user_2 = User(
        user_id=test_api_key_user_on_other_team_2,
        team_id=test_api_key_team,
        is_privileged_user=True,
    )

    # Initially there should be 0 model bundles.
    use_case_1 = ListModelBundlesV1UseCase(model_bundle_repository=fake_model_bundle_repository)

    response_1 = await use_case_1.execute(
        user=user_1, model_name=create_model_bundle_request.name, order_by=None
    )
    assert isinstance(response_1, ListModelBundlesV1Response)
    assert response_1.model_bundles == []

    # Create 1 model bundle.
    use_case_2 = CreateModelBundleV1UseCase(
        model_bundle_repository=fake_model_bundle_repository,
        docker_repository=fake_docker_repository_image_always_exists,
        model_primitive_gateway=fake_model_primitive_gateway,
    )
    await use_case_2.execute(user=user_1, request=create_model_bundle_request)

    # Now listing model bundles should yield 1 model bundle.
    response_2 = await use_case_1.execute(
        user=user_1,
        model_name=create_model_bundle_request.name,
        order_by=ModelBundleOrderBy.OLDEST,
    )
    assert len(response_2.model_bundles) == 1

    response_3 = await use_case_1.execute(
        user=user_2,
        model_name=create_model_bundle_request.name,
        order_by=ModelBundleOrderBy.OLDEST,
    )
    assert len(response_3.model_bundles) == 1

    # Create 2 more model bundles.
    await use_case_2.execute(user=user_1, request=create_model_bundle_request)
    await use_case_2.execute(user=user_2, request=create_model_bundle_request)

    # Now listing model bundles should yield 3 model bundles.
    response_4 = await use_case_1.execute(
        user=user_1,
        model_name=create_model_bundle_request.name,
        order_by=ModelBundleOrderBy.NEWEST,
    )
    assert len(response_4.model_bundles) == 3
    response_5 = await use_case_1.execute(
        user=user_2,
        model_name=create_model_bundle_request.name,
        order_by=ModelBundleOrderBy.NEWEST,
    )
    assert len(response_5.model_bundles) == 3


@pytest.mark.asyncio
async def test_create_model_bundle_v2_use_case_docker_not_found(
    test_api_key: str,
    fake_model_bundle_repository: ModelBundleRepository,
    create_model_bundle_v2_request: CreateModelBundleV2Request,
    fake_docker_repository_image_never_exists: DockerRepository,
    fake_model_primitive_gateway: ModelPrimitiveGateway,
):
    use_case = CreateModelBundleV2UseCase(
        model_bundle_repository=fake_model_bundle_repository,
        docker_repository=fake_docker_repository_image_never_exists,
        model_primitive_gateway=fake_model_primitive_gateway,
    )
    user = User(user_id=test_api_key, team_id=test_api_key, is_privileged_user=True)
    with pytest.raises(DockerImageNotFoundException):
        await use_case.execute(user=user, request=create_model_bundle_v2_request)


@pytest.mark.asyncio
async def test_create_model_bundle_v2_full_url_use_case_success(
    test_api_key: str,
    fake_model_bundle_repository: ModelBundleRepository,
    create_model_bundle_v2_request: CreateModelBundleV2Request,
    fake_docker_repository_image_never_exists: DockerRepository,
    fake_model_primitive_gateway: ModelPrimitiveGateway,
):
    use_case = CreateModelBundleV2UseCase(
        model_bundle_repository=fake_model_bundle_repository,
        docker_repository=fake_docker_repository_image_never_exists,
        model_primitive_gateway=fake_model_primitive_gateway,
    )
    user = User(user_id=test_api_key, team_id=test_api_key, is_privileged_user=True)
    # will a full uri specification, image existence is not checked
    create_model_bundle_v2_request.flavor.repository = "registry.hub.docker.com/library/busybox"
    response = await use_case.execute(user=user, request=create_model_bundle_v2_request)
    assert response.model_bundle_id
    assert isinstance(response, CreateModelBundleV2Response)
