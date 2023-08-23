import pytest
from model_engine_server.common.dtos.endpoint_builder import BuildEndpointRequest
from model_engine_server.common.dtos.resource_manager import CreateOrUpdateResourcesRequest


@pytest.fixture
def create_resources_request_sync_pytorch(
    test_api_key: str, build_endpoint_request_sync_pytorch: BuildEndpointRequest
) -> CreateOrUpdateResourcesRequest:
    create_resources_request = CreateOrUpdateResourcesRequest(
        build_endpoint_request=build_endpoint_request_sync_pytorch,
        image="test_image",
    )
    return create_resources_request


@pytest.fixture
def create_resources_request_async_runnable_image(
    test_api_key: str, build_endpoint_request_async_runnable_image: BuildEndpointRequest
) -> CreateOrUpdateResourcesRequest:
    create_resources_request = CreateOrUpdateResourcesRequest(
        build_endpoint_request=build_endpoint_request_async_runnable_image,
        image="test_image",
    )
    return create_resources_request


@pytest.fixture
def create_resources_request_sync_runnable_image(
    test_api_key: str, build_endpoint_request_sync_runnable_image: BuildEndpointRequest
) -> CreateOrUpdateResourcesRequest:
    create_resources_request = CreateOrUpdateResourcesRequest(
        build_endpoint_request=build_endpoint_request_sync_runnable_image,
        image="test_image",
    )
    return create_resources_request


@pytest.fixture
def create_resources_request_streaming_runnable_image(
    test_api_key: str, build_endpoint_request_streaming_runnable_image: BuildEndpointRequest
) -> CreateOrUpdateResourcesRequest:
    create_resources_request = CreateOrUpdateResourcesRequest(
        build_endpoint_request=build_endpoint_request_streaming_runnable_image,
        image="test_image",
    )
    return create_resources_request
