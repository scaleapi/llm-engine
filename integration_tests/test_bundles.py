import pytest
from tenacity import retry, stop_after_attempt, wait_fixed

from .rest_api_utils import (
    CREATE_MODEL_BUNDLE_REQUEST_RUNNABLE_IMAGE,
    CREATE_MODEL_BUNDLE_REQUEST_SIMPLE,
    USER_ID_0,
    create_model_bundle,
    ensure_launch_gateway_healthy,
    get_latest_model_bundle,
)


@pytest.fixture(scope="session")
@retry(stop=stop_after_attempt(10), wait=wait_fixed(30))
def model_bundles():
    ensure_launch_gateway_healthy()
    user = USER_ID_0
    for create_bundle_request in [
        CREATE_MODEL_BUNDLE_REQUEST_SIMPLE,
        CREATE_MODEL_BUNDLE_REQUEST_RUNNABLE_IMAGE,
    ]:
        create_model_bundle(create_bundle_request, user, "v2")
        bundle = get_latest_model_bundle(create_bundle_request["name"], user, "v2")
        assert bundle["name"] == create_bundle_request["name"]
        assert bundle["metadata"] == create_bundle_request["metadata"]
