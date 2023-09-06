import pytest

from .rest_api_utils import (
    CREATE_MODEL_BUNDLE_REQUEST_RUNNABLE_IMAGE,
    CREATE_MODEL_BUNDLE_REQUEST_SIMPLE,
    USER_ID_0,
    USER_ID_1,
    create_model_bundle,
    get_latest_model_bundle,
)


@pytest.fixture(scope="session")
def model_bundles():
    for user in [USER_ID_0, USER_ID_1]:
        for create_bundle_request in [
            CREATE_MODEL_BUNDLE_REQUEST_SIMPLE,
            CREATE_MODEL_BUNDLE_REQUEST_RUNNABLE_IMAGE,
        ]:
            create_model_bundle(create_bundle_request, user, "v2")
            bundle = get_latest_model_bundle(create_bundle_request["name"], user, "v2")
            assert bundle["name"] == create_bundle_request["name"]
            assert bundle["metadata"] == create_bundle_request["metadata"]
