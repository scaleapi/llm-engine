import pytest

from .rest_api_utils import (
    CREATE_MODEL_BUNDLE_REQUEST_CUSTOM_IMAGE,
    CREATE_MODEL_BUNDLE_REQUEST_RUNNABLE_IMAGE,
    CREATE_MODEL_BUNDLE_REQUEST_SIMPLE,
    USER_ID_0,
    USER_ID_1,
    create_model_bundle,
)


@pytest.mark.parametrize("user", [USER_ID_0, USER_ID_1])
@pytest.mark.parametrize(
    "create_bundle_request",
    [
        CREATE_MODEL_BUNDLE_REQUEST_SIMPLE,
        CREATE_MODEL_BUNDLE_REQUEST_CUSTOM_IMAGE,
        CREATE_MODEL_BUNDLE_REQUEST_RUNNABLE_IMAGE,
    ],
)
@pytest.mark.parametrize("route_version", ["v1", "v2"])
def test_model_bundle(user, create_bundle_request, route_version):
    create_model_bundle(create_bundle_request, user, route_version)
