from types import SimpleNamespace
from typing import Dict, Set
from unittest.mock import AsyncMock, patch

import pytest
from kubernetes_asyncio.client.rest import ApiException
from model_engine_server.infra.gateways.resources.image_cache_gateway import (
    TRUSTED_BIN_INIT_CONTAINER_NAME,
    TRUSTED_BUSYBOX_DIGEST,
    TRUSTED_BUSYBOX_IMAGE,
    CachedImages,
    ImageCacheGateway,
    _trusted_busybox_image,
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
            container["image"]
            for container in kwargs["body"]["spec"]["template"]["spec"]["containers"]
        )

    for k in expected_images.keys():
        assert expected_images[k].issubset(
            actual_images[k]
        ), f"Missing {expected_images[k].difference(actual_images[k])}"


@pytest.mark.asyncio
async def test_create_or_update_image_cache_runs_only_trusted_busybox(mock_apps_client):
    await ImageCacheGateway().create_or_update_image_cache(
        CachedImages(cpu=["cpu_image"], a10=[], a100=[], t4=[], h100=[])
    )

    bodies = [
        kwargs["body"] for _, kwargs in mock_apps_client.create_namespaced_daemon_set.call_args_list
    ]
    cpu_body = next(body for body in bodies if body["metadata"]["name"].endswith("-cpu"))
    pod_spec = cpu_body["spec"]["template"]["spec"]
    assert pod_spec["automountServiceAccountToken"] is False
    assert pod_spec["securityContext"]["runAsNonRoot"] is True
    assert [c["image"] for c in pod_spec["initContainers"]] == [TRUSTED_BUSYBOX_IMAGE]
    cached = next(c for c in pod_spec["containers"] if c["image"] == "cpu_image")
    assert cached["command"][0] == "/trusted/busybox"
    assert cached["securityContext"]["capabilities"] == {"drop": ["ALL"]}
    assert cached["securityContext"]["allowPrivilegeEscalation"] is False
    assert cached["resources"]["limits"] == {"cpu": "10m", "memory": "16Mi"}


@pytest.mark.parametrize(
    "template_image,expected_repository",
    [
        pytest.param(
            "public.ecr.aws/docker/library/busybox:latest",
            "public.ecr.aws/docker/library/busybox",
            id="default",
        ),
        pytest.param(
            "mirror.local:5000/library/busybox:1.36",
            "mirror.local:5000/library/busybox",
            id="mirror",
        ),
        pytest.param("busybox", "busybox", id="bare-name"),
        pytest.param(None, "public.ecr.aws/docker/library/busybox", id="no-template-busybox"),
    ],
)
def test_trusted_busybox_image_keeps_the_pinned_digest(template_image, expected_repository):
    containers = [{"name": "busybox", "image": template_image}] if template_image else []
    assert _trusted_busybox_image(containers) == f"{expected_repository}@{TRUSTED_BUSYBOX_DIGEST}"


def _existing_daemonset(name: str, images: Set[str], init_containers):
    return SimpleNamespace(
        metadata=SimpleNamespace(name=name),
        spec=SimpleNamespace(
            template=SimpleNamespace(
                spec=SimpleNamespace(
                    containers=[SimpleNamespace(image=image) for image in images],
                    init_containers=init_containers,
                )
            )
        ),
    )


@pytest.mark.parametrize(
    "init_containers,expect_replace",
    [
        pytest.param(None, True, id="legacy-daemonset-without-init-container"),
        pytest.param(
            [SimpleNamespace(name=TRUSTED_BIN_INIT_CONTAINER_NAME, image="busybox:old")],
            True,
            id="init-container-with-other-image",
        ),
        pytest.param(
            [SimpleNamespace(name=TRUSTED_BIN_INIT_CONTAINER_NAME, image=TRUSTED_BUSYBOX_IMAGE)],
            False,
            id="unchanged",
        ),
    ],
)
@pytest.mark.asyncio
async def test_create_image_cache_replaces_daemonsets_without_trusted_busybox(
    mock_apps_client, init_containers, expect_replace
):
    name = "launch-image-cache-cpu"
    images = ["cpu_image", "public.ecr.aws/docker/library/busybox:latest"]
    mock_apps_client.create_namespaced_daemon_set.side_effect = ApiException(status=409)
    mock_apps_client.list_namespaced_daemon_set.return_value = SimpleNamespace(
        items=[_existing_daemonset(name, set(images), init_containers)]
    )

    body = {"spec": {"template": {"spec": {"initContainers": [{"image": TRUSTED_BUSYBOX_IMAGE}]}}}}
    await ImageCacheGateway._create_image_cache(body, name, images)

    assert mock_apps_client.replace_namespaced_daemon_set.called is expect_replace
