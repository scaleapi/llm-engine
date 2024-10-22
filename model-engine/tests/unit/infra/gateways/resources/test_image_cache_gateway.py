from typing import Dict, Set
from unittest.mock import AsyncMock, patch

import pytest
from model_engine_server.infra.gateways.resources.image_cache_gateway import (
    CachedImages,
    ImageCacheGateway,
)

MODULE_PATH = "model_engine_server.infra.gateways.resources.k8s_endpoint_resource_delegate"


@pytest.fixture
def mock_get_kubernetes_cluster_version():
    mock_version = "1.26"
    with patch(
        f"{MODULE_PATH}.get_kubernetes_cluster_version",
        return_value=mock_version,
    ):
        yield mock_version


@pytest.fixture
def mock_apps_client():
    mock_client = AsyncMock()
    with patch(
        f"{MODULE_PATH}.get_kubernetes_apps_client",
        return_value=mock_client,
    ):
        yield mock_client


@pytest.fixture
def mock_core_client():
    mock_client = AsyncMock()
    with patch(
        f"{MODULE_PATH}.get_kubernetes_core_client",
        return_value=mock_client,
    ):
        yield mock_client


@pytest.fixture
def mock_autoscaling_client():
    mock_client = AsyncMock()
    with patch(
        f"{MODULE_PATH}.get_kubernetes_autoscaling_client",
        return_value=mock_client,
    ):
        yield mock_client


@pytest.fixture
def mock_policy_client():
    mock_client = AsyncMock()
    with patch(
        f"{MODULE_PATH}.get_kubernetes_policy_client",
        return_value=mock_client,
    ):
        yield mock_client


@pytest.fixture
def mock_custom_objects_client():
    mock_client = AsyncMock()
    with patch(
        f"{MODULE_PATH}.get_kubernetes_custom_objects_client",
        return_value=mock_client,
    ):
        yield mock_client


def test_create_or_update_image_cache(
    mock_apps_client,
    mock_core_client,
    mock_autoscaling_client,
    mock_policy_client,
    mock_custom_objects_client,
):
    gateway = ImageCacheGateway()

    gateway.create_or_update_image_cache(
        CachedImages(
            cpu=["cpu_image"],
            a10=["a10_image"],
            a100=["a100_image"],
            t4=["t4_image"],
            h100=["h100_image"],
            h100_mig_3g_40gb=["h100_mig_3g_40gb_image"],
            h100_mig_1g_20gb=["h100_mig_1g_20gb_image"],
        )
    )

    expected_images: Dict[str, Set[str]] = {
        "cpu": {"cpu_image"},
        "a10": {"a10_image"},
        "a100": {"a100_image"},
        "t4": {"t4_image"},
        "h100": {"h100_image"},
        "h100_mig_3g_40gb": {"h100_mig_3g_40gb_image"},
        "h100_mig_1g_20gb": {"h100_mig_1g_20gb_image"},
    }

    actual_images: Dict[str, Set[str]] = {
        "cpu": set(),
        "a10": set(),
        "a100": set(),
        "t4": set(),
        "h100": set(),
        "h100_mig_3g_40gb": set(),
        "h100_mig_1g_20gb": set(),
    }

    for call_args in mock_apps_client.create_namespaced_daemon_set.call_args_list:
        _, kwargs = call_args
        compute_type = kwargs["body"]["metadata"]["name"].split("-")[-1]
        actual_images[compute_type] = set(
            kwargs["body"]["spec"]["template"]["spec"]["containers"][0]["image"]
        )

    assert actual_images == expected_images
