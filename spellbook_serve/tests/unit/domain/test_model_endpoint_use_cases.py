import pytest

from spellbook_serve.common.dtos.model_endpoints import (
    CreateModelEndpointV1Request,
    CreateModelEndpointV1Response,
    DeleteModelEndpointV1Response,
    GetModelEndpointV1Response,
    ListModelEndpointsV1Response,
    ModelEndpointOrderBy,
    UpdateModelEndpointV1Request,
    UpdateModelEndpointV1Response,
)
from spellbook_serve.core.auth.authentication_repository import User
from spellbook_serve.core.domain_exceptions import (
    ObjectHasInvalidValueException,
    ObjectNotAuthorizedException,
    ObjectNotFoundException,
)
from spellbook_serve.domain.entities import ModelBundle, ModelEndpoint
from spellbook_serve.domain.exceptions import (
    EndpointLabelsException,
    EndpointResourceInvalidRequestException,
)
from spellbook_serve.domain.use_cases.model_endpoint_use_cases import (
    ALLOWED_TEAMS,
    CreateModelEndpointV1UseCase,
    DeleteModelEndpointByIdV1UseCase,
    GetModelEndpointByIdV1UseCase,
    ListModelEndpointsV1UseCase,
    UpdateModelEndpointByIdV1UseCase,
)


@pytest.mark.asyncio
async def test_create_model_endpoint_use_case_success(
    fake_model_bundle_repository,
    fake_model_endpoint_service,
    model_bundle_1: ModelBundle,
    model_bundle_5: ModelBundle,
    create_model_endpoint_request_async: CreateModelEndpointV1Request,
    create_model_endpoint_request_sync: CreateModelEndpointV1Request,
    create_model_endpoint_request_streaming: CreateModelEndpointV1Request,
):
    fake_model_bundle_repository.add_model_bundle(model_bundle_1)
    fake_model_bundle_repository.add_model_bundle(model_bundle_5)
    fake_model_endpoint_service.model_bundle_repository = fake_model_bundle_repository
    use_case = CreateModelEndpointV1UseCase(
        model_bundle_repository=fake_model_bundle_repository,
        model_endpoint_service=fake_model_endpoint_service,
    )
    user_id = model_bundle_1.created_by
    user = User(user_id=user_id, team_id=user_id, is_privileged_user=True)
    response_1 = await use_case.execute(user=user, request=create_model_endpoint_request_async)
    assert response_1.endpoint_creation_task_id
    assert isinstance(response_1, CreateModelEndpointV1Response)

    response_2 = await use_case.execute(user=user, request=create_model_endpoint_request_sync)
    assert response_2.endpoint_creation_task_id
    assert isinstance(response_2, CreateModelEndpointV1Response)

    response_3 = await use_case.execute(user=user, request=create_model_endpoint_request_streaming)
    assert response_3.endpoint_creation_task_id
    assert isinstance(response_3, CreateModelEndpointV1Response)


@pytest.mark.asyncio
async def test_create_model_endpoint_use_case_raises_invalid_value_exception(
    fake_model_bundle_repository,
    fake_model_endpoint_service,
    model_bundle_1: ModelBundle,
    create_model_endpoint_request_streaming: CreateModelEndpointV1Request,
):
    fake_model_bundle_repository.add_model_bundle(model_bundle_1)
    fake_model_endpoint_service.model_bundle_repository = fake_model_bundle_repository
    use_case = CreateModelEndpointV1UseCase(
        model_bundle_repository=fake_model_bundle_repository,
        model_endpoint_service=fake_model_endpoint_service,
    )

    request = create_model_endpoint_request_streaming.copy()
    request.model_bundle_id = model_bundle_1.id
    user_id = model_bundle_1.created_by
    user = User(user_id=user_id, team_id=user_id, is_privileged_user=True)
    with pytest.raises(ObjectHasInvalidValueException):
        await use_case.execute(user=user, request=request)


@pytest.mark.asyncio
async def test_create_model_endpoint_use_case_raises_per_worker_invalid_value_exception(
    fake_model_bundle_repository,
    fake_model_endpoint_service,
    model_bundle_5: ModelBundle,
    create_model_endpoint_request_streaming: CreateModelEndpointV1Request,
):
    fake_model_bundle_repository.add_model_bundle(model_bundle_5)
    fake_model_endpoint_service.model_bundle_repository = fake_model_bundle_repository
    use_case = CreateModelEndpointV1UseCase(
        model_bundle_repository=fake_model_bundle_repository,
        model_endpoint_service=fake_model_endpoint_service,
    )

    request = create_model_endpoint_request_streaming.copy()
    request.per_worker = 2
    user_id = model_bundle_5.created_by
    user = User(user_id=user_id, team_id=user_id, is_privileged_user=True)
    with pytest.raises(ObjectHasInvalidValueException):
        await use_case.execute(user=user, request=request)


@pytest.mark.asyncio
async def test_create_model_endpoint_use_case_raises_resource_request_exception(
    fake_model_bundle_repository,
    fake_model_endpoint_service,
    model_bundle_1: ModelBundle,
    create_model_endpoint_request_async: CreateModelEndpointV1Request,
    create_model_endpoint_request_sync: CreateModelEndpointV1Request,
):
    fake_model_bundle_repository.add_model_bundle(model_bundle_1)
    fake_model_endpoint_service.model_bundle_repository = fake_model_bundle_repository
    use_case = CreateModelEndpointV1UseCase(
        model_bundle_repository=fake_model_bundle_repository,
        model_endpoint_service=fake_model_endpoint_service,
    )

    request = create_model_endpoint_request_async.copy()
    request.cpus = -1
    user_id = model_bundle_1.created_by
    user = User(user_id=user_id, team_id=user_id, is_privileged_user=True)
    with pytest.raises(EndpointResourceInvalidRequestException):
        await use_case.execute(user=user, request=request)

    request = create_model_endpoint_request_async.copy()
    request.cpus = float("inf")
    with pytest.raises(EndpointResourceInvalidRequestException):
        await use_case.execute(user=user, request=request)

    request = create_model_endpoint_request_async.copy()
    request.memory = "invalid_memory_amount"
    with pytest.raises(EndpointResourceInvalidRequestException):
        await use_case.execute(user=user, request=request)

    request = create_model_endpoint_request_async.copy()
    request.memory = float("inf")
    with pytest.raises(EndpointResourceInvalidRequestException):
        await use_case.execute(user=user, request=request)

    request = create_model_endpoint_request_async.copy()
    request.storage = "invalid_storage_amount"
    with pytest.raises(EndpointResourceInvalidRequestException):
        await use_case.execute(user=user, request=request)

    request = create_model_endpoint_request_async.copy()
    request.storage = float("inf")
    with pytest.raises(EndpointResourceInvalidRequestException):
        await use_case.execute(user=user, request=request)

    request = create_model_endpoint_request_sync.copy()
    request.min_workers = 0
    with pytest.raises(EndpointResourceInvalidRequestException):
        await use_case.execute(user=user, request=request)

    request = create_model_endpoint_request_async.copy()
    request.max_workers = 2**63
    with pytest.raises(EndpointResourceInvalidRequestException):
        await use_case.execute(user=user, request=request)

    request = create_model_endpoint_request_async.copy()
    request.gpus = 0
    with pytest.raises(EndpointResourceInvalidRequestException):
        await use_case.execute(user=user, request=request)

    request = create_model_endpoint_request_async.copy()
    request.gpu_type = None
    with pytest.raises(EndpointResourceInvalidRequestException):
        await use_case.execute(user=user, request=request)

    request = create_model_endpoint_request_async.copy()
    request.gpu_type = "invalid_gpu_type"
    with pytest.raises(EndpointResourceInvalidRequestException):
        await use_case.execute(user=user, request=request)


@pytest.mark.asyncio
async def test_create_model_endpoint_use_case_raises_endpoint_labels_exception(
    fake_model_bundle_repository,
    fake_model_endpoint_service,
    model_bundle_1: ModelBundle,
    create_model_endpoint_request_async: CreateModelEndpointV1Request,
):
    fake_model_bundle_repository.add_model_bundle(model_bundle_1)
    fake_model_endpoint_service.model_bundle_repository = fake_model_bundle_repository
    use_case = CreateModelEndpointV1UseCase(
        model_bundle_repository=fake_model_bundle_repository,
        model_endpoint_service=fake_model_endpoint_service,
    )
    user_id = model_bundle_1.created_by
    user = User(user_id=user_id, team_id=user_id, is_privileged_user=True)

    request = create_model_endpoint_request_async.copy()
    request.labels = None  # type: ignore
    with pytest.raises(EndpointLabelsException):
        await use_case.execute(user=user, request=request)

    request = create_model_endpoint_request_async.copy()
    request.labels = {}
    with pytest.raises(EndpointLabelsException):
        await use_case.execute(user=user, request=request)

    request = create_model_endpoint_request_async.copy()
    request.labels = {"team": "infra"}
    with pytest.raises(EndpointLabelsException):
        await use_case.execute(user=user, request=request)

    request = create_model_endpoint_request_async.copy()
    request.labels = {"product": "my_product"}
    with pytest.raises(EndpointLabelsException):
        await use_case.execute(user=user, request=request)

    request = create_model_endpoint_request_async.copy()
    request.labels = {
        "team": "infra",
        "product": "my_product",
        "user_id": "test_labels_user",
    }
    with pytest.raises(EndpointLabelsException):
        await use_case.execute(user=user, request=request)

    request = create_model_endpoint_request_async.copy()
    request.labels = {
        "team": "infra",
        "product": "my_product",
        "endpoint_name": "test_labels_endpoint_name",
    }
    with pytest.raises(EndpointLabelsException):
        await use_case.execute(user=user, request=request)


@pytest.mark.asyncio
async def test_create_model_endpoint_use_case_invalid_team_raises_endpoint_labels_exception(
    fake_model_bundle_repository,
    fake_model_endpoint_service,
    model_bundle_1: ModelBundle,
    create_model_endpoint_request_async: CreateModelEndpointV1Request,
):
    fake_model_bundle_repository.add_model_bundle(model_bundle_1)
    fake_model_endpoint_service.model_bundle_repository = fake_model_bundle_repository
    use_case = CreateModelEndpointV1UseCase(
        model_bundle_repository=fake_model_bundle_repository,
        model_endpoint_service=fake_model_endpoint_service,
    )
    user_id = model_bundle_1.created_by
    user = User(user_id=user_id, team_id=user_id, is_privileged_user=True)

    request = create_model_endpoint_request_async.copy()
    request.labels = {
        "team": "unknown_team",
        "product": "my_product",
    }
    with pytest.raises(EndpointLabelsException):
        await use_case.execute(user=user, request=request)

    for team in ALLOWED_TEAMS:
        # Conversely, make sure that all the ALLOWED_TEAMS are, well, allowed.
        request = create_model_endpoint_request_async.copy()
        request.labels = {
            "team": team,
            "product": "my_product",
        }
        await use_case.execute(user=user, request=request)


@pytest.mark.asyncio
async def test_create_model_endpoint_use_case_validates_post_inference_hooks(
    fake_model_bundle_repository,
    fake_model_endpoint_service,
    model_bundle_1: ModelBundle,
    create_model_endpoint_request_async: CreateModelEndpointV1Request,
):
    fake_model_bundle_repository.add_model_bundle(model_bundle_1)
    fake_model_endpoint_service.model_bundle_repository = fake_model_bundle_repository
    use_case = CreateModelEndpointV1UseCase(
        model_bundle_repository=fake_model_bundle_repository,
        model_endpoint_service=fake_model_endpoint_service,
    )
    user_id = model_bundle_1.created_by
    user = User(user_id=user_id, team_id=user_id, is_privileged_user=True)

    request = create_model_endpoint_request_async.copy()
    request.post_inference_hooks = ["invalid_hook"]
    with pytest.raises(ValueError):
        await use_case.execute(user=user, request=request)


@pytest.mark.asyncio
async def test_create_model_endpoint_use_case_model_bundle_not_authorized(
    fake_model_bundle_repository,
    fake_model_endpoint_service,
    model_bundle_1: ModelBundle,
    create_model_endpoint_request_async: CreateModelEndpointV1Request,
):
    fake_model_bundle_repository.add_model_bundle(model_bundle_1)
    use_case = CreateModelEndpointV1UseCase(
        model_bundle_repository=fake_model_bundle_repository,
        model_endpoint_service=fake_model_endpoint_service,
    )
    user_id = "invalid_user_id"
    user = User(user_id=user_id, team_id=user_id, is_privileged_user=True)
    with pytest.raises(ObjectNotAuthorizedException):
        await use_case.execute(user=user, request=create_model_endpoint_request_async)


@pytest.mark.asyncio
async def test_create_model_endpoint_use_case_model_bundle_same_team(
    fake_model_bundle_repository,
    fake_model_endpoint_service,
    model_bundle_1: ModelBundle,
    create_model_endpoint_request_async: CreateModelEndpointV1Request,
    test_api_key_user_on_other_team: str,
    test_api_key_user_on_other_team_2: str,
    test_api_key_team: str,
):
    # person 2 on team 1 can create an endpoint from a bundle from person 1 on team 1
    model_bundle_1.created_by = test_api_key_user_on_other_team
    model_bundle_1.owner = test_api_key_team
    fake_model_bundle_repository.add_model_bundle(model_bundle_1)
    fake_model_endpoint_service.model_bundle_repository = fake_model_bundle_repository
    use_case = CreateModelEndpointV1UseCase(
        model_bundle_repository=fake_model_bundle_repository,
        model_endpoint_service=fake_model_endpoint_service,
    )
    user = User(
        user_id=test_api_key_user_on_other_team_2,
        team_id=test_api_key_team,
        is_privileged_user=True,
    )
    response_1 = await use_case.execute(user=user, request=create_model_endpoint_request_async)
    assert response_1.endpoint_creation_task_id


@pytest.mark.asyncio
async def test_create_model_endpoint_use_case_model_bundle_not_found(
    test_api_key: str,
    fake_model_bundle_repository,
    fake_model_endpoint_service,
    create_model_endpoint_request_sync: CreateModelEndpointV1Request,
):
    use_case = CreateModelEndpointV1UseCase(
        model_bundle_repository=fake_model_bundle_repository,
        model_endpoint_service=fake_model_endpoint_service,
    )
    user_id = test_api_key
    user = User(user_id=user_id, team_id=user_id, is_privileged_user=True)
    with pytest.raises(ObjectNotFoundException):
        await use_case.execute(user=user, request=create_model_endpoint_request_sync)


@pytest.mark.parametrize(
    "prewarm_input, expected_result",
    [
        (False, False),
        (None, True),
        (True, True),
    ],
)
@pytest.mark.asyncio
async def test_create_model_endpoint_use_case_sets_prewarm(
    fake_model_bundle_repository,
    fake_model_endpoint_service,
    model_bundle_1: ModelBundle,
    create_model_endpoint_request_async: CreateModelEndpointV1Request,
    create_model_endpoint_request_sync: CreateModelEndpointV1Request,
    prewarm_input,
    expected_result,
):
    fake_model_bundle_repository.add_model_bundle(model_bundle_1)
    fake_model_endpoint_service.model_bundle_repository = fake_model_bundle_repository
    use_case = CreateModelEndpointV1UseCase(
        model_bundle_repository=fake_model_bundle_repository,
        model_endpoint_service=fake_model_endpoint_service,
    )
    user_id = model_bundle_1.created_by
    user = User(user_id=user_id, team_id=user_id, is_privileged_user=True)

    for endpoint_request in [
        create_model_endpoint_request_async,
        create_model_endpoint_request_sync,
    ]:
        request = endpoint_request.copy()
        request.prewarm = prewarm_input
        await use_case.execute(user=user, request=request)
        endpoints = await fake_model_endpoint_service.list_model_endpoints(
            name=request.name, owner=model_bundle_1.created_by, order_by=None
        )
        assert len(endpoints) == 1, "The test itself is probably broken"
        assert (
            endpoints[0].infra_state.prewarm == expected_result
        ), f"Fed in {prewarm_input}, expected {expected_result}"
        await fake_model_endpoint_service.delete_model_endpoint(endpoints[0].record.id)


@pytest.mark.parametrize(
    "high_priority_input, expected_result",
    [
        (False, False),
        (None, False),
        (True, True),
    ],
)
@pytest.mark.asyncio
async def test_create_model_endpoint_use_case_sets_high_priority(
    fake_model_bundle_repository,
    fake_model_endpoint_service,
    model_bundle_1: ModelBundle,
    create_model_endpoint_request_async: CreateModelEndpointV1Request,
    create_model_endpoint_request_sync: CreateModelEndpointV1Request,
    high_priority_input,
    expected_result,
):
    fake_model_bundle_repository.add_model_bundle(model_bundle_1)
    fake_model_endpoint_service.model_bundle_repository = fake_model_bundle_repository
    use_case = CreateModelEndpointV1UseCase(
        model_bundle_repository=fake_model_bundle_repository,
        model_endpoint_service=fake_model_endpoint_service,
    )
    user_id = model_bundle_1.created_by
    user = User(user_id=user_id, team_id=user_id, is_privileged_user=True)
    for endpoint_request in [
        create_model_endpoint_request_async,
        create_model_endpoint_request_sync,
    ]:
        request = endpoint_request.copy()
        request.high_priority = high_priority_input
        await use_case.execute(user=user, request=request)
        endpoints = await fake_model_endpoint_service.list_model_endpoints(
            name=request.name, owner=model_bundle_1.created_by, order_by=None
        )
        assert len(endpoints) == 1, "The test itself is probably broken"
        assert (
            endpoints[0].infra_state.high_priority == expected_result
        ), f"Fed in {high_priority_input}, expected {expected_result}"
        await fake_model_endpoint_service.delete_model_endpoint(endpoints[0].record.id)


@pytest.mark.asyncio
async def test_get_model_endpoint_use_case_success(
    test_api_key: str,
    fake_model_endpoint_service,
    model_endpoint_1: ModelEndpoint,
):
    fake_model_endpoint_service.add_model_endpoint(model_endpoint_1)
    use_case = GetModelEndpointByIdV1UseCase(model_endpoint_service=fake_model_endpoint_service)
    user = User(user_id=test_api_key, team_id=test_api_key, is_privileged_user=True)
    response = await use_case.execute(user=user, model_endpoint_id=model_endpoint_1.record.id)

    assert isinstance(response, GetModelEndpointV1Response)


@pytest.mark.asyncio
async def test_get_model_endpoint_use_case_same_team_finds_endpoint(
    test_api_key_user_on_other_team: str,
    test_api_key_user_on_other_team_2: str,
    test_api_key_team: str,
    fake_model_endpoint_service,
    model_endpoint_1: ModelEndpoint,
):
    model_endpoint_1.record.created_by = test_api_key_user_on_other_team
    model_endpoint_1.record.owner = test_api_key_team
    fake_model_endpoint_service.add_model_endpoint(model_endpoint_1)
    use_case = GetModelEndpointByIdV1UseCase(model_endpoint_service=fake_model_endpoint_service)
    user = User(
        user_id=test_api_key_user_on_other_team_2,
        team_id=test_api_key_team,
        is_privileged_user=True,
    )
    response = await use_case.execute(user, model_endpoint_id=model_endpoint_1.record.id)

    assert isinstance(response, GetModelEndpointV1Response)


@pytest.mark.asyncio
async def test_get_model_endpoint_use_case_raises_not_found(
    test_api_key: str,
    fake_model_endpoint_service,
    model_endpoint_1: ModelEndpoint,
):
    fake_model_endpoint_service.add_model_endpoint(model_endpoint_1)
    use_case = GetModelEndpointByIdV1UseCase(model_endpoint_service=fake_model_endpoint_service)
    user = User(user_id=test_api_key, team_id=test_api_key, is_privileged_user=True)
    with pytest.raises(ObjectNotFoundException):
        await use_case.execute(user=user, model_endpoint_id="invalid_model_endpoint_id")


@pytest.mark.asyncio
async def test_get_model_endpoint_use_case_raises_not_authorized(
    fake_model_endpoint_service,
    model_endpoint_1: ModelEndpoint,
):
    fake_model_endpoint_service.add_model_endpoint(model_endpoint_1)
    use_case = GetModelEndpointByIdV1UseCase(model_endpoint_service=fake_model_endpoint_service)
    user = User(user_id="invalid_user_id", team_id="invalid_team_id", is_privileged_user=True)
    with pytest.raises(ObjectNotAuthorizedException):
        await use_case.execute(user=user, model_endpoint_id=model_endpoint_1.record.id)


@pytest.mark.asyncio
async def test_list_model_endpoints(
    test_api_key: str,
    fake_model_endpoint_service,
    model_endpoint_1: ModelEndpoint,
    model_endpoint_2: ModelEndpoint,
):
    # Initially there should be 0 model endpoints.
    use_case = ListModelEndpointsV1UseCase(model_endpoint_service=fake_model_endpoint_service)
    user = User(user_id=test_api_key, team_id=test_api_key, is_privileged_user=True)
    response_1 = await use_case.execute(user=user, name=None, order_by=None)
    assert isinstance(response_1, ListModelEndpointsV1Response)
    assert response_1.model_endpoints == []

    # Add 2 model endpoints.
    fake_model_endpoint_service.add_model_endpoint(model_endpoint_1)
    fake_model_endpoint_service.add_model_endpoint(model_endpoint_2)

    # Now listing model endpoints should yield 2 model endpoint.
    response_2 = await use_case.execute(user=user, name=None, order_by=ModelEndpointOrderBy.NEWEST)
    assert len(response_2.model_endpoints) == 2
    assert response_2.model_endpoints[0].name == "test_model_endpoint_name_2"
    assert response_2.model_endpoints[1].name == "test_model_endpoint_name_1"

    response_3 = await use_case.execute(
        user=user,
        name=None,
        order_by=ModelEndpointOrderBy.ALPHABETICAL,
    )
    assert len(response_3.model_endpoints) == 2
    assert response_3.model_endpoints[0].name == "test_model_endpoint_name_1"
    assert response_3.model_endpoints[1].name == "test_model_endpoint_name_2"

    # Listing model endpoints by name should yield 1 model endpoint.
    response_4 = await use_case.execute(
        user=user,
        name=model_endpoint_1.record.name,
        order_by=ModelEndpointOrderBy.OLDEST,
    )
    assert len(response_4.model_endpoints) == 1


@pytest.mark.asyncio
async def test_list_model_endpoints_team(
    test_api_key_user_on_other_team: str,
    test_api_key_user_on_other_team_2: str,
    test_api_key_team: str,
    fake_model_endpoint_service,
    model_endpoint_1: ModelEndpoint,
    model_endpoint_2: ModelEndpoint,
):
    # Add two model endpoints
    model_endpoint_1.record.created_by = test_api_key_user_on_other_team
    model_endpoint_1.record.owner = test_api_key_team
    model_endpoint_2.record.created_by = test_api_key_user_on_other_team_2
    model_endpoint_2.record.owner = test_api_key_team
    fake_model_endpoint_service.add_model_endpoint(model_endpoint_1)
    fake_model_endpoint_service.add_model_endpoint(model_endpoint_2)

    user_1 = User(
        user_id=test_api_key_user_on_other_team,
        team_id=test_api_key_team,
        is_privileged_user=True,
    )
    user_2 = User(
        user_id=test_api_key_user_on_other_team,
        team_id=test_api_key_team,
        is_privileged_user=True,
    )
    other_user = User(user_id="invalid_user", team_id="invalid_team", is_privileged_user=True)

    # Listing model endpoints should give 2 endpoints no matter who lists them
    use_case = ListModelEndpointsV1UseCase(model_endpoint_service=fake_model_endpoint_service)
    response_1 = await use_case.execute(
        user=user_1, name=None, order_by=ModelEndpointOrderBy.NEWEST
    )
    assert len(response_1.model_endpoints) == 2
    response_2 = await use_case.execute(
        user=user_2, name=None, order_by=ModelEndpointOrderBy.OLDEST
    )
    assert len(response_2.model_endpoints) == 2

    # Other user shouldn't see endpoints
    response_3 = await use_case.execute(
        user=other_user, name=None, order_by=ModelEndpointOrderBy.NEWEST
    )
    assert len(response_3.model_endpoints) == 0

    # Listing by name should give the endpoint no matter who created it, as long as user is on same team
    response_4 = await use_case.execute(
        user=user_1,
        name=model_endpoint_1.record.name,
        order_by=ModelEndpointOrderBy.OLDEST,
    )
    assert len(response_4.model_endpoints) == 1
    response_5 = await use_case.execute(
        user=user_2,
        name=model_endpoint_1.record.name,
        order_by=ModelEndpointOrderBy.OLDEST,
    )
    assert len(response_5.model_endpoints) == 1
    response_6 = await use_case.execute(
        user=other_user,
        name=model_endpoint_1.record.name,
        order_by=ModelEndpointOrderBy.OLDEST,
    )
    assert len(response_6.model_endpoints) == 0


@pytest.mark.asyncio
async def test_update_model_endpoint_success(
    fake_model_bundle_repository,
    fake_model_endpoint_service,
    model_bundle_1: ModelBundle,
    model_bundle_2: ModelBundle,
    model_endpoint_1: ModelEndpoint,
    update_model_endpoint_request: UpdateModelEndpointV1Request,
):
    fake_model_bundle_repository.add_model_bundle(model_bundle_1)
    fake_model_bundle_repository.add_model_bundle(model_bundle_2)
    fake_model_endpoint_service.add_model_endpoint(model_endpoint_1)
    fake_model_endpoint_service.model_bundle_repository = fake_model_bundle_repository
    use_case = UpdateModelEndpointByIdV1UseCase(
        model_bundle_repository=fake_model_bundle_repository,
        model_endpoint_service=fake_model_endpoint_service,
    )
    user_id = model_endpoint_1.record.created_by
    user = User(user_id=user_id, team_id=user_id, is_privileged_user=True)
    response = await use_case.execute(
        user=user,
        model_endpoint_id=model_endpoint_1.record.id,
        request=update_model_endpoint_request,
    )
    assert response.endpoint_creation_task_id
    assert isinstance(response, UpdateModelEndpointV1Response)


@pytest.mark.asyncio
async def test_update_model_endpoint_team_success(
    fake_model_bundle_repository,
    fake_model_endpoint_service,
    model_bundle_1: ModelBundle,
    model_bundle_2: ModelBundle,
    model_endpoint_1: ModelEndpoint,
    update_model_endpoint_request: UpdateModelEndpointV1Request,
    test_api_key_user_on_other_team: str,
    test_api_key_user_on_other_team_2: str,
    test_api_key_team: str,
):
    # Someone can update a model endpoint of someone else on the same team
    model_bundle_1.created_by = test_api_key_user_on_other_team
    model_bundle_1.owner = test_api_key_team
    model_bundle_2.created_by = test_api_key_user_on_other_team_2
    model_bundle_2.owner = test_api_key_team
    model_endpoint_1.record.created_by = test_api_key_user_on_other_team
    model_endpoint_1.record.owner = test_api_key_team
    fake_model_bundle_repository.add_model_bundle(model_bundle_1)
    fake_model_bundle_repository.add_model_bundle(model_bundle_2)
    fake_model_endpoint_service.add_model_endpoint(model_endpoint_1)
    fake_model_endpoint_service.model_bundle_repository = fake_model_bundle_repository
    use_case = UpdateModelEndpointByIdV1UseCase(
        model_bundle_repository=fake_model_bundle_repository,
        model_endpoint_service=fake_model_endpoint_service,
    )
    user = User(
        user_id=test_api_key_user_on_other_team_2,
        team_id=test_api_key_team,
        is_privileged_user=True,
    )
    response = await use_case.execute(
        user=user,
        model_endpoint_id=model_endpoint_1.record.id,
        request=update_model_endpoint_request,
    )
    assert response.endpoint_creation_task_id
    assert isinstance(response, UpdateModelEndpointV1Response)


@pytest.mark.asyncio
async def test_update_model_endpoint_raises_not_found(
    fake_model_bundle_repository,
    fake_model_endpoint_service,
    model_bundle_1: ModelBundle,
    model_bundle_2: ModelBundle,
    model_endpoint_1: ModelEndpoint,
    update_model_endpoint_request: UpdateModelEndpointV1Request,
):
    fake_model_bundle_repository.add_model_bundle(model_bundle_1)
    fake_model_bundle_repository.add_model_bundle(model_bundle_2)
    fake_model_endpoint_service.add_model_endpoint(model_endpoint_1)
    fake_model_endpoint_service.model_bundle_repository = fake_model_bundle_repository
    use_case = UpdateModelEndpointByIdV1UseCase(
        model_bundle_repository=fake_model_bundle_repository,
        model_endpoint_service=fake_model_endpoint_service,
    )
    user_id = model_endpoint_1.record.created_by
    user = User(user_id=user_id, team_id=user_id, is_privileged_user=True)

    with pytest.raises(ObjectNotFoundException):
        await use_case.execute(
            user=user,
            model_endpoint_id="invalid_model_endpoint_id",
            request=update_model_endpoint_request,
        )

    update_model_endpoint_request.model_bundle_id = "invalid_model_bundle_id"
    with pytest.raises(ObjectNotFoundException):
        await use_case.execute(
            user=user,
            model_endpoint_id=model_endpoint_1.record.id,
            request=update_model_endpoint_request,
        )


@pytest.mark.asyncio
async def test_update_model_endpoint_raises_not_authorized(
    fake_model_bundle_repository,
    fake_model_endpoint_service,
    model_bundle_1: ModelBundle,
    model_bundle_2: ModelBundle,
    model_endpoint_1: ModelEndpoint,
    update_model_endpoint_request: UpdateModelEndpointV1Request,
):
    fake_model_bundle_repository.add_model_bundle(model_bundle_1)
    fake_model_bundle_repository.add_model_bundle(model_bundle_2)
    fake_model_endpoint_service.add_model_endpoint(model_endpoint_1)
    fake_model_endpoint_service.model_bundle_repository = fake_model_bundle_repository
    use_case = UpdateModelEndpointByIdV1UseCase(
        model_bundle_repository=fake_model_bundle_repository,
        model_endpoint_service=fake_model_endpoint_service,
    )

    old_user_id = model_endpoint_1.record.created_by
    model_endpoint_1.record.created_by = "test_user_id_2"
    model_endpoint_1.record.owner = "test_user_id_2"
    user1 = User(user_id=old_user_id, team_id=old_user_id, is_privileged_user=True)
    with pytest.raises(ObjectNotAuthorizedException):
        await use_case.execute(
            user=user1,
            model_endpoint_id=model_endpoint_1.record.id,
            request=update_model_endpoint_request,
        )
    model_endpoint_1.record.created_by = old_user_id

    model_bundle_2.created_by = "test_user_id_2"
    model_bundle_2.owner = "test_user_id_2"
    user_id = model_endpoint_1.record.created_by
    user2 = User(user_id=user_id, team_id=user_id, is_privileged_user=True)
    with pytest.raises(ObjectNotAuthorizedException):
        await use_case.execute(
            user=user2,
            model_endpoint_id=model_endpoint_1.record.id,
            request=update_model_endpoint_request,
        )


@pytest.mark.asyncio
async def test_update_model_endpoint_raises_endpoint_labels_exception(
    fake_model_bundle_repository,
    fake_model_endpoint_service,
    model_bundle_1: ModelBundle,
    model_bundle_2: ModelBundle,
    model_endpoint_1: ModelEndpoint,
    update_model_endpoint_request: UpdateModelEndpointV1Request,
):
    fake_model_bundle_repository.add_model_bundle(model_bundle_1)
    fake_model_bundle_repository.add_model_bundle(model_bundle_2)
    fake_model_endpoint_service.add_model_endpoint(model_endpoint_1)
    fake_model_endpoint_service.model_bundle_repository = fake_model_bundle_repository
    use_case = UpdateModelEndpointByIdV1UseCase(
        model_bundle_repository=fake_model_bundle_repository,
        model_endpoint_service=fake_model_endpoint_service,
    )

    request = update_model_endpoint_request.copy()
    request.labels = {"team": "infra"}
    user_id = model_endpoint_1.record.created_by
    user = User(user_id=user_id, team_id=user_id, is_privileged_user=True)
    with pytest.raises(EndpointLabelsException):
        await use_case.execute(
            user=user,
            model_endpoint_id=model_endpoint_1.record.id,
            request=request,
        )

    request = update_model_endpoint_request.copy()
    request.labels = {"product": "my_product"}
    with pytest.raises(EndpointLabelsException):
        await use_case.execute(
            user=user,
            model_endpoint_id=model_endpoint_1.record.id,
            request=request,
        )

    request = update_model_endpoint_request.copy()
    request.labels = {
        "team": "infra",
        "product": "my_product",
        "user_id": "test_labels_user",
    }
    with pytest.raises(EndpointLabelsException):
        await use_case.execute(
            user=user,
            model_endpoint_id=model_endpoint_1.record.id,
            request=request,
        )

    request = update_model_endpoint_request.copy()
    request.labels = {
        "team": "infra",
        "product": "my_product",
        "endpoint_name": "test_labels_endpoint_name",
    }
    with pytest.raises(EndpointLabelsException):
        await use_case.execute(
            user=user,
            model_endpoint_id=model_endpoint_1.record.id,
            request=request,
        )


@pytest.mark.asyncio
async def test_update_model_endpoint_invalid_team_raises_endpoint_labels_exception(
    fake_model_bundle_repository,
    fake_model_endpoint_service,
    model_bundle_1: ModelBundle,
    model_bundle_2: ModelBundle,
    model_endpoint_1: ModelEndpoint,
    update_model_endpoint_request: UpdateModelEndpointV1Request,
):
    fake_model_bundle_repository.add_model_bundle(model_bundle_1)
    fake_model_bundle_repository.add_model_bundle(model_bundle_2)
    fake_model_endpoint_service.add_model_endpoint(model_endpoint_1)
    fake_model_endpoint_service.model_bundle_repository = fake_model_bundle_repository
    use_case = UpdateModelEndpointByIdV1UseCase(
        model_bundle_repository=fake_model_bundle_repository,
        model_endpoint_service=fake_model_endpoint_service,
    )

    request = update_model_endpoint_request.copy()
    request.labels = {
        "team": "invalid_team",
        "product": "some_product",
    }
    user_id = model_endpoint_1.record.created_by
    user = User(user_id=user_id, team_id=user_id, is_privileged_user=True)
    with pytest.raises(EndpointLabelsException):
        await use_case.execute(
            user=user,
            model_endpoint_id=model_endpoint_1.record.id,
            request=request,
        )

    for team in ALLOWED_TEAMS:
        # Conversely, make sure that all the ALLOWED_TEAMS are, well, allowed.
        request = update_model_endpoint_request.copy()
        request.labels = {
            "team": team,
            "product": "my_product",
        }
        await use_case.execute(
            user=user, model_endpoint_id=model_endpoint_1.record.id, request=request
        )


@pytest.mark.asyncio
async def test_delete_model_endpoint_success(
    fake_model_endpoint_service,
    model_endpoint_1: ModelEndpoint,
):
    fake_model_endpoint_service.add_model_endpoint(model_endpoint_1)
    use_case = DeleteModelEndpointByIdV1UseCase(
        model_endpoint_service=fake_model_endpoint_service,
    )
    user_id = model_endpoint_1.record.created_by
    user = User(user_id=user_id, team_id=user_id, is_privileged_user=True)
    response = await use_case.execute(user=user, model_endpoint_id=model_endpoint_1.record.id)
    assert response.deleted
    assert isinstance(response, DeleteModelEndpointV1Response)


@pytest.mark.asyncio
async def test_delete_model_endpoint_success_team(
    fake_model_endpoint_service,
    model_endpoint_1: ModelEndpoint,
    test_api_key_user_on_other_team: str,
    test_api_key_user_on_other_team_2: str,
    test_api_key_team: str,
):
    # User can delete endpoint created by other user on the same team
    model_endpoint_1.record.created_by = test_api_key_user_on_other_team
    model_endpoint_1.record.owner = test_api_key_team
    fake_model_endpoint_service.add_model_endpoint(model_endpoint_1)
    use_case = DeleteModelEndpointByIdV1UseCase(
        model_endpoint_service=fake_model_endpoint_service,
    )
    user = User(
        user_id=test_api_key_user_on_other_team_2,
        team_id=test_api_key_team,
        is_privileged_user=True,
    )
    response = await use_case.execute(user=user, model_endpoint_id=model_endpoint_1.record.id)
    assert response.deleted
    assert isinstance(response, DeleteModelEndpointV1Response)


@pytest.mark.asyncio
async def test_delete_model_endpoint_raises_not_found(
    fake_model_endpoint_service,
    model_endpoint_1: ModelEndpoint,
):
    fake_model_endpoint_service.add_model_endpoint(model_endpoint_1)
    use_case = DeleteModelEndpointByIdV1UseCase(
        model_endpoint_service=fake_model_endpoint_service,
    )
    user_id = model_endpoint_1.record.created_by
    user = User(user_id=user_id, team_id=user_id, is_privileged_user=True)
    with pytest.raises(ObjectNotFoundException):
        await use_case.execute(user=user, model_endpoint_id="invalid_user_id")


@pytest.mark.asyncio
async def test_delete_model_endpoint_raises_not_authorized(
    fake_model_endpoint_service,
    model_endpoint_1: ModelEndpoint,
):
    fake_model_endpoint_service.add_model_endpoint(model_endpoint_1)
    use_case = DeleteModelEndpointByIdV1UseCase(
        model_endpoint_service=fake_model_endpoint_service,
    )
    user = User(user_id="invalid_user_id", team_id="invalid_user_id", is_privileged_user=True)
    with pytest.raises(ObjectNotAuthorizedException):
        await use_case.execute(user=user, model_endpoint_id=model_endpoint_1.record.id)
