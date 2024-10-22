from typing import Dict, Set
from unittest.mock import AsyncMock, patch

import pytest
from model_engine_server.infra.gateways.resources.image_cache_gateway import (
    CachedImages,
    ImageCacheGateway,
)

MODULE_PATH = "model_engine_server.infra.gateways.resources.image_cache_gateway"

@pytest.fixture
def mock_apps_client():
    mock_client = AsyncMock()
    with patch(
        f"{MODULE_PATH}.get_kubernetes_apps_client",
        return_value=mock_client,
    ):
        yield mock_client


@pytest.mark.asyncio
async def test_create_or_update_image_cache(
    mock_apps_client,
):
    gateway = ImageCacheGateway()
    await gateway.create_or_update_image_cache(
        CachedImages(
            cpu=["cpu_image"],
            a10=["a10_image"],
            a100=["a100_image"],
            t4=["t4_image"],
            h100=["h100_image"],
        )
    )

    # Needs to correspond with model_engine_server/infra/gateways/resources/templates/service_template_config_map_circleci.yaml
    expected_images: Dict[str, Set[str]] = {
        "cpu": {"cpu_image"},
        "a10": {"a10_image"},
        "a100": {"a100_image"},
        "t4": {"t4_image"},
        "h100": {"h100_image"},
    }

    actual_images: Dict[str, Set[str]] = {
        "cpu": set(),
        "a10": set(),
        "a100": set(),
        "t4": set(),
        "h100": set(),
    }

    for call_args in mock_apps_client.create_namespaced_daemon_set.call_args_list:
        _, kwargs = call_args
        compute_type = kwargs["body"]["metadata"]["name"].split("-")[-1]
        actual_images[compute_type] = set(
            container["image"] for container in kwargs["body"]["spec"]["template"]["spec"]["containers"]
        )

    for k in expected_images.keys():
        assert expected_images[k].issubset(actual_images[k]), f"Missing {expected_images[k].difference(actual_images[k])}"

