import pytest

from .rest_api_utils import (
    CREATE_MODEL_BUNDLE_REQUEST_RUNNABLE_IMAGE,
    CREATE_MODEL_BUNDLE_REQUEST_SIMPLE,
    USER_ID_0,
    USER_ID_1,
    create_model_bundle,
    get_latest_model_bundle,
)


@pytest.mark.parametrize("user", [USER_ID_0, USER_ID_1])
@pytest.mark.parametrize(
    "create_bundle_request",
    [
        CREATE_MODEL_BUNDLE_REQUEST_SIMPLE,
        CREATE_MODEL_BUNDLE_REQUEST_RUNNABLE_IMAGE,
    ],
)
def test_model_bundle(user, create_bundle_request):
    create_model_bundle(create_bundle_request, user)
    bundle = get_latest_model_bundle(
        create_bundle_request["name"], user
    )
    assert bundle["name"] == create_bundle_request["name"]
    assert bundle["metadata"] == create_bundle_request["metadata"]
