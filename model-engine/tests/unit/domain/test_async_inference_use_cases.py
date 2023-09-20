from typing import Any, Dict, Tuple

import pytest
from model_engine_server.common.dtos.tasks import EndpointPredictV1Request, TaskStatus
from model_engine_server.core.auth.authentication_repository import User
from model_engine_server.domain.entities import ModelEndpoint
from model_engine_server.domain.exceptions import (
    ObjectNotAuthorizedException,
    ObjectNotFoundException,
)
from model_engine_server.domain.use_cases.async_inference_use_cases import (
    CreateAsyncInferenceTaskV1UseCase,
    GetAsyncInferenceTaskV1UseCase,
)


@pytest.mark.asyncio
async def test_create_async_inference_task_use_case_success(
    fake_model_endpoint_service,
    model_endpoint_1: ModelEndpoint,
    endpoint_predict_request_1: Tuple[EndpointPredictV1Request, Dict[str, Any]],
):
    fake_model_endpoint_service.add_model_endpoint(model_endpoint_1)
    use_case = CreateAsyncInferenceTaskV1UseCase(model_endpoint_service=fake_model_endpoint_service)
    user_id = model_endpoint_1.record.created_by
    user = User(user_id=user_id, team_id=user_id, is_privileged_user=True)
    response = await use_case.execute(
        user=user,
        model_endpoint_id=model_endpoint_1.record.id,
        request=endpoint_predict_request_1[0],
    )
    assert response.dict() == {"task_id": "test_task_id"}


@pytest.mark.asyncio
async def test_create_async_inference_task_use_case_endpoint_not_found_raises(
    fake_model_endpoint_service,
    model_endpoint_1: ModelEndpoint,
    endpoint_predict_request_1: Tuple[EndpointPredictV1Request, Dict[str, Any]],
):
    fake_model_endpoint_service.add_model_endpoint(model_endpoint_1)
    use_case = CreateAsyncInferenceTaskV1UseCase(model_endpoint_service=fake_model_endpoint_service)
    user_id = model_endpoint_1.record.created_by
    user = User(user_id=user_id, team_id=user_id, is_privileged_user=True)
    with pytest.raises(ObjectNotFoundException):
        await use_case.execute(
            user=user,
            model_endpoint_id="invalid_model_endpoint_id",
            request=endpoint_predict_request_1[0],
        )


@pytest.mark.asyncio
async def test_create_async_inference_task_use_case_endpoint_not_authorized_raises(
    fake_model_endpoint_service,
    model_endpoint_1: ModelEndpoint,
    endpoint_predict_request_1: Tuple[EndpointPredictV1Request, Dict[str, Any]],
):
    fake_model_endpoint_service.add_model_endpoint(model_endpoint_1)
    use_case = CreateAsyncInferenceTaskV1UseCase(model_endpoint_service=fake_model_endpoint_service)
    user = User(user_id="invalid_user_id", team_id="invalid_user_id", is_privileged_user=True)
    with pytest.raises(ObjectNotAuthorizedException):
        await use_case.execute(
            user=user,
            model_endpoint_id=model_endpoint_1.record.id,
            request=endpoint_predict_request_1[0],
        )


@pytest.mark.asyncio
async def test_create_async_inference_task_use_case_endpoint_public_endpoint_authorized(
    fake_model_endpoint_service,
    model_endpoint_public: ModelEndpoint,
    endpoint_predict_request_1: Tuple[EndpointPredictV1Request, Dict[str, Any]],
):
    fake_model_endpoint_service.add_model_endpoint(model_endpoint_public)
    use_case = CreateAsyncInferenceTaskV1UseCase(model_endpoint_service=fake_model_endpoint_service)
    user = User(user_id="invalid_user_id", team_id="invalid_user_id", is_privileged_user=True)
    # Should not raise
    await use_case.execute(
        user=user,
        model_endpoint_id=model_endpoint_public.record.id,
        request=endpoint_predict_request_1[0],
    )


def test_get_async_inference_task_use_case_success(
    fake_model_endpoint_service,
    model_endpoint_1: ModelEndpoint,
    endpoint_predict_request_1: Tuple[EndpointPredictV1Request, Dict[str, Any]],
):
    fake_model_endpoint_service.add_model_endpoint(model_endpoint_1)
    use_case = GetAsyncInferenceTaskV1UseCase(model_endpoint_service=fake_model_endpoint_service)
    user_id = model_endpoint_1.record.created_by
    user = User(user_id=user_id, team_id=user_id, is_privileged_user=True)
    response = use_case.execute(user=user, task_id="test_task_id")
    assert response.status == TaskStatus.SUCCESS
