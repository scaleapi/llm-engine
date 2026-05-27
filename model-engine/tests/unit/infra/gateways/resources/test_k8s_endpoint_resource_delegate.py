import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import ANY, AsyncMock, MagicMock, Mock, patch

import pytest
from kubernetes_asyncio.client.rest import ApiException
from model_engine_server.common.config import hmi_config
from model_engine_server.common.dtos.resource_manager import CreateOrUpdateResourcesRequest
from model_engine_server.common.env_vars import GIT_TAG
from model_engine_server.domain.entities import (
    LLMInferenceFramework,
    ModelBundle,
    ModelEndpointConfig,
    ModelEndpointType,
    ModelEndpointUserConfigState,
)
from model_engine_server.domain.exceptions import EndpointResourceInfraException
from model_engine_server.infra.gateways.resources.k8s_endpoint_resource_delegate import (
    DATADOG_ENV_VAR,
    MODEL_CACHE_VOLUME_NAME,
    K8SEndpointResourceDelegate,
    add_datadog_env_to_container,
    add_pod_metadata_env_to_container,
    get_main_container_from_deployment_template,
    k8s_yaml_exists,
    load_k8s_yaml,
)
from model_engine_server.infra.gateways.resources.k8s_resource_types import (
    DictStrInt,
    DictStrStr,
    PersistentVolumeClaimArguments,
    ResourceArguments,
    get_endpoint_resource_arguments_from_request,
)
from tests.unit.infra.gateways.k8s_fake_objects import FakeK8sDeploymentContainer, FakeK8sEnvVar

MODULE_PATH = "model_engine_server.infra.gateways.resources.k8s_endpoint_resource_delegate"

EXAMPLE_LWS_CONFIG_PATH = os.path.abspath(os.path.join(__file__, "..", "example_lws_config.json"))
with open(EXAMPLE_LWS_CONFIG_PATH, "r") as f:
    EXAMPLE_LWS_CONFIG = json.load(f)


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


@pytest.fixture
def autoscaling_params():
    return dict(
        min_workers=1,
        max_workers=3,
        per_worker=4,
    )


@pytest.fixture
def common_endpoint_params():
    return dict(
        cpus="1",
        memory="8G",
        gpus=1,
        gpu_type="nvidia-tesla-t4",
        storage=None,
        bundle_url="test_url",
        aws_role="default",
        results_s3_bucket="test_bucket",
        image="test_image",
        labels=dict(team="test_team", product="test_product"),
    )


@pytest.fixture
def k8s_endpoint_resource_delegate(
    autoscaling_params,
    common_endpoint_params,
) -> K8SEndpointResourceDelegate:
    gateway = K8SEndpointResourceDelegate()
    gateway.__setattr__("_get_async_autoscaling_params", AsyncMock(return_value=autoscaling_params))
    gateway.__setattr__("_get_sync_autoscaling_params", AsyncMock(return_value=autoscaling_params))
    gateway.__setattr__(
        "_get_common_endpoint_params", AsyncMock(return_value=common_endpoint_params)
    )
    return gateway


def test_k8s_yaml_exists():
    # This is tied to
    # llm-engine/model-engine/model_engine_server/infra/gateways/resources/templates/service_template_config_map_circleci.yaml
    assert k8s_yaml_exists("image-cache-h100.yaml"), "image-cache-h100.yaml should exist"
    assert not k8s_yaml_exists(
        "image-cache-abc9001.yaml"
    ), "image-cache-abc9001.yaml should not exist"


def _render_service_template_config_map(extra_args: Optional[List[str]] = None) -> str:
    if shutil.which("helm") is None:
        pytest.skip("helm is not installed")

    repo_root = Path(__file__).resolve().parents[6]
    chart_path = repo_root / "charts" / "model-engine"
    values_path = chart_path / "values_circleci.yaml"
    command = [
        "helm",
        "template",
        "test-release",
        str(chart_path),
        "--show-only",
        "templates/service_template_config_map.yaml",
        "-f",
        str(values_path),
    ]
    if extra_args is not None:
        command.extend(extra_args)

    return subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
    ).stdout


def _write_rendered_service_template_config_map(
    tmp_path: Path,
    extra_args: Optional[List[str]] = None,
) -> Path:
    rendered_config_map_path = tmp_path / "service_template_config_map.yaml"
    rendered_config_map_path.write_text(_render_service_template_config_map(extra_args))
    return rendered_config_map_path


def _pod_spec_has_model_cache(pod_spec: Dict[str, Any]) -> bool:
    return any(
        volume.get("name") == MODEL_CACHE_VOLUME_NAME for volume in pod_spec.get("volumes", [])
    ) or any(
        volume_mount.get("name") == MODEL_CACHE_VOLUME_NAME
        for container in pod_spec.get("containers", [])
        for volume_mount in container.get("volumeMounts", [])
    )


def _assert_pod_spec_has_injected_model_cache(
    pod_spec: Dict[str, Any], expected_claim_name: str
) -> None:
    volume = next(
        volume for volume in pod_spec["volumes"] if volume.get("name") == MODEL_CACHE_VOLUME_NAME
    )
    assert volume["persistentVolumeClaim"]["claimName"] == expected_claim_name
    container = next(
        container for container in pod_spec["containers"] if container["name"] == "main"
    )
    volume_mount = next(
        volume_mount
        for volume_mount in container["volumeMounts"]
        if volume_mount.get("name") == MODEL_CACHE_VOLUME_NAME
    )
    assert volume_mount["mountPath"] == "/mnt/model-cache"


def test_model_cache_enabled_helm_template_renders_pvc_without_base_mounts(tmp_path):
    rendered_template = _render_service_template_config_map(["--set", "modelCache.enabled=true"])

    assert "persistent-volume-claim.yaml" in rendered_template
    assert '        - "ReadWriteMany"' in rendered_template
    assert "name: model-cache" not in rendered_template
    assert "claimName: ${MODEL_CACHE_PVC_NAME}" not in rendered_template
    assert "mountPath: /mnt/model-cache" not in rendered_template

    rendered_config_map_path = tmp_path / "service_template_config_map.yaml"
    rendered_config_map_path.write_text(rendered_template)
    persistent_volume_claim_arguments = PersistentVolumeClaimArguments(
        RESOURCE_NAME="launch-endpoint-id-test",
        NAMESPACE="launch-inference",
        ENDPOINT_ID="end_test",
        ENDPOINT_NAME="test-endpoint",
        TEAM="test-team",
        PRODUCT="test-product",
        CREATED_BY="test-creator",
        OWNER="test-owner",
        GIT_TAG="test-git-tag",
        MODEL_CACHE_PVC_NAME="launch-endpoint-id-test-model-cache",
    )

    with (
        patch(f"{MODULE_PATH}.LAUNCH_SERVICE_TEMPLATE_FOLDER", None),
        patch(
            f"{MODULE_PATH}.LAUNCH_SERVICE_TEMPLATE_CONFIG_MAP_PATH",
            str(rendered_config_map_path),
        ),
    ):
        assert k8s_yaml_exists("persistent-volume-claim.yaml")
        persistent_volume_claim = load_k8s_yaml(
            "persistent-volume-claim.yaml",
            persistent_volume_claim_arguments,
        )

    assert persistent_volume_claim["kind"] == "PersistentVolumeClaim"
    assert persistent_volume_claim["metadata"]["name"] == "launch-endpoint-id-test-model-cache"
    assert persistent_volume_claim["metadata"]["namespace"] == "launch-inference"
    assert persistent_volume_claim["metadata"]["labels"]["endpoint_id"] == "end_test"
    assert persistent_volume_claim["metadata"]["labels"]["endpoint_name"] == "test-endpoint"
    assert persistent_volume_claim["metadata"]["labels"]["team"] == "test-team"
    assert persistent_volume_claim["metadata"]["labels"]["product"] == "test-product"
    assert persistent_volume_claim["spec"]["accessModes"] == ["ReadWriteMany"]
    assert persistent_volume_claim["spec"]["resources"]["requests"]["storage"] == "100Gi"


def test_model_cache_disabled_helm_template_has_no_base_cache_placeholders():
    rendered_template = _render_service_template_config_map(["--set", "modelCache.enabled=false"])

    assert "persistent-volume-claim.yaml" in rendered_template
    assert "name: model-cache" not in rendered_template
    assert "claimName: ${MODEL_CACHE_PVC_NAME}" not in rendered_template
    assert "mountPath: /mnt/model-cache" not in rendered_template


def test_model_cache_null_helm_template_has_no_base_cache_placeholders(tmp_path):
    null_values_path = tmp_path / "modelcache-null.yaml"
    null_values_path.write_text("modelCache: null\n")

    rendered_template = _render_service_template_config_map(["-f", str(null_values_path)])

    assert "persistent-volume-claim.yaml" in rendered_template
    assert "name: model-cache" not in rendered_template
    assert "claimName: ${MODEL_CACHE_PVC_NAME}" not in rendered_template
    assert "mountPath: /mnt/model-cache" not in rendered_template


def test_deployment_template_substitution_does_not_require_model_cache_pvc_name(
    tmp_path,
    create_resources_request_sync_runnable_image: CreateOrUpdateResourcesRequest,
):
    rendered_config_map_path = _write_rendered_service_template_config_map(
        tmp_path,
        ["--set", "modelCache.enabled=true"],
    )
    resource_arguments = dict(
        get_endpoint_resource_arguments_from_request(
            k8s_resource_group_name="launch-endpoint-id-test",
            request=create_resources_request_sync_runnable_image,
            sqs_queue_name="my_queue",
            sqs_queue_url="https://my_queue",
            endpoint_resource_name="deployment-runnable-image-sync-gpu",
        )
    )
    resource_arguments.pop("MODEL_CACHE_PVC_NAME", None)

    with (
        patch(f"{MODULE_PATH}.LAUNCH_SERVICE_TEMPLATE_FOLDER", None),
        patch(
            f"{MODULE_PATH}.LAUNCH_SERVICE_TEMPLATE_CONFIG_MAP_PATH",
            str(rendered_config_map_path),
        ),
    ):
        deployment = load_k8s_yaml(
            "deployment-runnable-image-sync-gpu.yaml",
            resource_arguments,
        )

    assert deployment["kind"] == "Deployment"
    assert not _pod_spec_has_model_cache(deployment["spec"]["template"]["spec"])


@pytest.mark.asyncio
async def test_model_cache_enabled_non_vllm_endpoint_does_not_create_or_mount_pvc(
    tmp_path,
    k8s_endpoint_resource_delegate,
    mock_apps_client,
    mock_core_client,
    mock_autoscaling_client,
    mock_policy_client,
    mock_custom_objects_client,
    mock_get_kubernetes_cluster_version,
    create_resources_request_sync_runnable_image: CreateOrUpdateResourcesRequest,
):
    rendered_config_map_path = _write_rendered_service_template_config_map(
        tmp_path,
        ["--set", "modelCache.enabled=true"],
    )

    with (
        patch(f"{MODULE_PATH}.LAUNCH_SERVICE_TEMPLATE_FOLDER", None),
        patch(
            f"{MODULE_PATH}.LAUNCH_SERVICE_TEMPLATE_CONFIG_MAP_PATH",
            str(rendered_config_map_path),
        ),
    ):
        await k8s_endpoint_resource_delegate.create_or_update_resources(
            create_resources_request_sync_runnable_image,
            sqs_queue_name="my_queue",
            sqs_queue_url="https://my_queue",
        )

    mock_core_client.create_namespaced_persistent_volume_claim.assert_not_called()
    deployment = mock_apps_client.create_namespaced_deployment.call_args.kwargs["body"]
    assert not _pod_spec_has_model_cache(deployment["spec"]["template"]["spec"])


@pytest.mark.asyncio
async def test_model_cache_enabled_vllm_endpoint_creates_and_mounts_pvc(
    tmp_path,
    k8s_endpoint_resource_delegate,
    mock_apps_client,
    mock_core_client,
    mock_autoscaling_client,
    mock_policy_client,
    mock_custom_objects_client,
    mock_get_kubernetes_cluster_version,
    create_resources_request_sync_runnable_image: CreateOrUpdateResourcesRequest,
):
    create_resources_request_sync_runnable_image.build_endpoint_request.model_endpoint_record.metadata = {
        "_llm": {
            "inference_framework": LLMInferenceFramework.VLLM.value,
            "model_cache_enabled": True,
        }
    }
    rendered_config_map_path = _write_rendered_service_template_config_map(
        tmp_path,
        ["--set", "modelCache.enabled=true"],
    )

    with (
        patch(f"{MODULE_PATH}.LAUNCH_SERVICE_TEMPLATE_FOLDER", None),
        patch(
            f"{MODULE_PATH}.LAUNCH_SERVICE_TEMPLATE_CONFIG_MAP_PATH",
            str(rendered_config_map_path),
        ),
    ):
        await k8s_endpoint_resource_delegate.create_or_update_resources(
            create_resources_request_sync_runnable_image,
            sqs_queue_name="my_queue",
            sqs_queue_url="https://my_queue",
        )

    mock_core_client.create_namespaced_persistent_volume_claim.assert_called_once()
    deployment = mock_apps_client.create_namespaced_deployment.call_args.kwargs["body"]
    _assert_pod_spec_has_injected_model_cache(
        deployment["spec"]["template"]["spec"],
        f"{deployment['metadata']['name']}-model-cache",
    )


@pytest.mark.asyncio
async def test_cached_vllm_endpoint_keeps_pvc_mount_when_helm_cache_disabled(
    tmp_path,
    k8s_endpoint_resource_delegate,
    mock_apps_client,
    mock_core_client,
    mock_autoscaling_client,
    mock_policy_client,
    mock_custom_objects_client,
    mock_get_kubernetes_cluster_version,
    create_resources_request_sync_runnable_image: CreateOrUpdateResourcesRequest,
):
    create_resources_request_sync_runnable_image.build_endpoint_request.model_endpoint_record.metadata = {
        "_llm": {
            "inference_framework": LLMInferenceFramework.VLLM.value,
            "model_cache_enabled": True,
        }
    }
    rendered_config_map_path = _write_rendered_service_template_config_map(
        tmp_path,
        ["--set", "modelCache.enabled=false"],
    )

    with (
        patch(f"{MODULE_PATH}.LAUNCH_SERVICE_TEMPLATE_FOLDER", None),
        patch(
            f"{MODULE_PATH}.LAUNCH_SERVICE_TEMPLATE_CONFIG_MAP_PATH",
            str(rendered_config_map_path),
        ),
    ):
        await k8s_endpoint_resource_delegate.create_or_update_resources(
            create_resources_request_sync_runnable_image,
            sqs_queue_name="my_queue",
            sqs_queue_url="https://my_queue",
        )

    mock_core_client.create_namespaced_persistent_volume_claim.assert_called_once()
    deployment = mock_apps_client.create_namespaced_deployment.call_args.kwargs["body"]
    _assert_pod_spec_has_injected_model_cache(
        deployment["spec"]["template"]["spec"],
        f"{deployment['metadata']['name']}-model-cache",
    )


@pytest.mark.asyncio
async def test_model_cache_enabled_vllm_endpoint_without_cached_bundle_does_not_mount_pvc(
    tmp_path,
    k8s_endpoint_resource_delegate,
    mock_apps_client,
    mock_core_client,
    mock_autoscaling_client,
    mock_policy_client,
    mock_custom_objects_client,
    mock_get_kubernetes_cluster_version,
    create_resources_request_sync_runnable_image: CreateOrUpdateResourcesRequest,
):
    create_resources_request_sync_runnable_image.build_endpoint_request.model_endpoint_record.metadata = {
        "_llm": {"inference_framework": LLMInferenceFramework.VLLM.value}
    }
    rendered_config_map_path = _write_rendered_service_template_config_map(
        tmp_path,
        ["--set", "modelCache.enabled=true"],
    )

    with (
        patch(f"{MODULE_PATH}.LAUNCH_SERVICE_TEMPLATE_FOLDER", None),
        patch(
            f"{MODULE_PATH}.LAUNCH_SERVICE_TEMPLATE_CONFIG_MAP_PATH",
            str(rendered_config_map_path),
        ),
    ):
        await k8s_endpoint_resource_delegate.create_or_update_resources(
            create_resources_request_sync_runnable_image,
            sqs_queue_name="my_queue",
            sqs_queue_url="https://my_queue",
        )

    mock_core_client.create_namespaced_persistent_volume_claim.assert_not_called()
    deployment = mock_apps_client.create_namespaced_deployment.call_args.kwargs["body"]
    assert not _pod_spec_has_model_cache(deployment["spec"]["template"]["spec"])


@pytest.mark.asyncio
async def test_model_cache_enabled_non_vllm_lws_does_not_create_or_mount_pvc(
    tmp_path,
    k8s_endpoint_resource_delegate,
    mock_apps_client,
    mock_core_client,
    mock_autoscaling_client,
    mock_policy_client,
    mock_custom_objects_client,
    mock_get_kubernetes_cluster_version,
    create_resources_request_streaming_runnable_image: CreateOrUpdateResourcesRequest,
    model_bundle_5: ModelBundle,
):
    model_bundle_5.flavor.worker_env = {"fake_env": "fake_value"}
    model_bundle_5.flavor.worker_command = ["fake_command"]
    create_resources_request_streaming_runnable_image.build_endpoint_request.model_endpoint_record.current_model_bundle = (
        model_bundle_5
    )
    create_resources_request_streaming_runnable_image.build_endpoint_request.model_endpoint_record.endpoint_type = (
        ModelEndpointType.STREAMING
    )
    create_resources_request_streaming_runnable_image.build_endpoint_request.nodes_per_worker = 2
    rendered_config_map_path = _write_rendered_service_template_config_map(
        tmp_path,
        ["--set", "modelCache.enabled=true"],
    )

    with (
        patch(f"{MODULE_PATH}.LAUNCH_SERVICE_TEMPLATE_FOLDER", None),
        patch(
            f"{MODULE_PATH}.LAUNCH_SERVICE_TEMPLATE_CONFIG_MAP_PATH",
            str(rendered_config_map_path),
        ),
    ):
        await k8s_endpoint_resource_delegate.create_or_update_resources(
            create_resources_request_streaming_runnable_image,
            sqs_queue_name="my_queue",
            sqs_queue_url="https://my_queue",
        )

    mock_core_client.create_namespaced_persistent_volume_claim.assert_not_called()
    lws = next(
        call_args.kwargs["body"]
        for call_args in mock_custom_objects_client.create_namespaced_custom_object.call_args_list
        if call_args.kwargs["group"] == "leaderworkerset.x-k8s.io"
    )
    leader_worker_template = lws["spec"]["leaderWorkerTemplate"]
    assert not _pod_spec_has_model_cache(leader_worker_template["leaderTemplate"]["spec"])
    assert not _pod_spec_has_model_cache(leader_worker_template["workerTemplate"]["spec"])


@pytest.mark.parametrize("resource_arguments_type", ResourceArguments.__args__)
def test_resource_arguments_type_and_add_datadog_env_to_main_container(
    resource_arguments_type,
):
    # Convert the name of the type to a kebab case string
    # e.g. "BatchJobOrchestrationJobArguments" -> "batch-job-orchestration-job-arguments"
    resource_arguments_type_name = resource_arguments_type.__name__
    resource_arguments_type_name = "".join(
        "-" + c.lower() if c.isupper() else c for c in resource_arguments_type_name
    ).lstrip("-")
    resource_arguments_type_name = resource_arguments_type_name.replace("-arguments", "")
    if resource_arguments_type is PersistentVolumeClaimArguments and not k8s_yaml_exists(
        "persistent-volume-claim.yaml"
    ):
        return

    # Hack for image cache, which has a special naming system
    if resource_arguments_type_name == "image-cache":
        resource_arguments_type_name = "image-cache-a10"

    # Create a default instance of a TypedDict
    type_to_default_value = {
        DictStrInt: "foo: 2",
        DictStrStr: "foo: bar",
        List[Dict[str, Any]]: [
            {
                "name": "foo",
                "value": "bar",
            }
        ],
        List[str]: ["foo", "bar"],
        bool: True,
        float: 1.1,
        int: 1,
        str: "foo",
        Optional[int]: 1,
        Optional[str]: "foo",
    }
    resource_arguments = {
        key: type_to_default_value[type_]
        for key, type_ in resource_arguments_type.__annotations__.items()
    }

    deployment_template = load_k8s_yaml(f"{resource_arguments_type_name}.yaml", resource_arguments)
    if "runnable-image" in resource_arguments_type_name:
        user_container = get_main_container_from_deployment_template(deployment_template)
        add_datadog_env_to_container(deployment_template, user_container)

        user_container = get_main_container_from_deployment_template(deployment_template)

        datadog_env = DATADOG_ENV_VAR.copy()
        for env_var in user_container["env"]:
            if env_var["name"] in datadog_env:
                datadog_env.remove(env_var["name"])
        assert len(datadog_env) == 0


def _verify_deployment_labels(
    body: Dict[str, Any],
    create_resources_request: CreateOrUpdateResourcesRequest,
):
    build_endpoint_request = create_resources_request.build_endpoint_request
    model_endpoint_record = build_endpoint_request.model_endpoint_record
    user_id = model_endpoint_record.created_by
    labels = build_endpoint_request.labels
    endpoint_name = model_endpoint_record.name
    env = "circleci"

    k8s_resource_group_name = f"launch-endpoint-id-{model_endpoint_record.id.replace('_', '-')}"

    assert body["metadata"]["name"] == k8s_resource_group_name
    assert body["metadata"]["namespace"] == hmi_config.endpoint_namespace
    assert labels

    expected_labels = {
        "created_by": user_id,
        "user_id": user_id,
        "endpoint_id": model_endpoint_record.id,
        "endpoint_name": endpoint_name,
        "managed-by": "model-engine",
        "owner": user_id,
        "team": labels["team"],
        "product": labels["product"],
        "env": env,
        "tags.datadoghq.com/env": env,
        "tags.datadoghq.com/service": endpoint_name,
        "tags.datadoghq.com/version": GIT_TAG,
        "use_scale_launch_endpoint_network_policy": "true",
    }
    assert body["metadata"]["labels"] == expected_labels

    expected_template_labels = {
        "app": k8s_resource_group_name,
        "created_by": user_id,
        "user_id": user_id,
        "endpoint_id": model_endpoint_record.id,
        "endpoint_name": endpoint_name,
        "managed-by": "model-engine",
        "owner": user_id,
        "team": labels["team"],
        "product": labels["product"],
        "env": env,
        "version": "v1",
        "tags.datadoghq.com/env": env,
        "tags.datadoghq.com/service": endpoint_name,
        "tags.datadoghq.com/version": GIT_TAG,
        "use_scale_launch_endpoint_network_policy": "true",
    }

    if model_endpoint_record.endpoint_type == ModelEndpointType.ASYNC:
        expected_template_labels["sidecar.istio.io/inject"] = "false"

    assert body["spec"]["template"]["metadata"]["labels"] == expected_template_labels


def _verify_non_deployment_labels(
    body: Dict[str, Any],
    create_resources_request: CreateOrUpdateResourcesRequest,
):
    build_endpoint_request = create_resources_request.build_endpoint_request
    model_endpoint_record = build_endpoint_request.model_endpoint_record
    user_id = model_endpoint_record.created_by
    labels = build_endpoint_request.labels
    endpoint_name = model_endpoint_record.name
    env = "circleci"

    k8s_resource_group_name = f"launch-endpoint-id-{model_endpoint_record.id.replace('_', '-')}"

    assert k8s_resource_group_name in body["metadata"]["name"]
    assert body["metadata"]["namespace"] == hmi_config.endpoint_namespace
    assert labels

    expected_labels = {
        "created_by": user_id,
        "managed-by": "model-engine",
        "owner": user_id,
        "user_id": user_id,
        "endpoint_id": model_endpoint_record.id,
        "endpoint_name": endpoint_name,
        "team": labels["team"],
        "product": labels["product"],
        "env": env,
        "tags.datadoghq.com/env": env,
        "tags.datadoghq.com/service": endpoint_name,
        "tags.datadoghq.com/version": GIT_TAG,
        "use_scale_launch_endpoint_network_policy": "true",
    }
    assert body["metadata"]["labels"] == expected_labels


def _verify_custom_object_plurals(call_args_list, expected_plurals: List[str]) -> None:
    for plural in expected_plurals:
        for call in call_args_list:
            if call.kwargs["plural"] == plural:
                break
        else:
            pytest.fail(
                f"Expecting to find plural {plural} in calls to create_namespaced_custom_object"
            )


@pytest.mark.asyncio
async def test_create_async_endpoint_has_correct_labels_and_dest(
    k8s_endpoint_resource_delegate,
    mock_apps_client,
    mock_core_client,
    mock_autoscaling_client,
    mock_policy_client,
    mock_custom_objects_client,
    mock_get_kubernetes_cluster_version,
    create_resources_request_async_runnable_image: CreateOrUpdateResourcesRequest,
):
    for request in [
        create_resources_request_async_runnable_image,
    ]:
        dest = await k8s_endpoint_resource_delegate.create_or_update_resources(
            request, sqs_queue_name="my_queue", sqs_queue_url="https://my_queue"
        )
        assert dest == "my_queue"

        # Verify deployment labels
        create_deployment_call_args = mock_apps_client.create_namespaced_deployment.call_args
        deployment_body = create_deployment_call_args.kwargs["body"]
        _verify_deployment_labels(deployment_body, request)

        # Make sure that a Service is *not* created for async endpoints.
        create_service_call_args = mock_core_client.create_namespaced_service.call_args
        assert create_service_call_args is None

        # Verify config_map labels
        create_config_map_call_args = mock_core_client.create_namespaced_config_map.call_args
        config_map_body = create_config_map_call_args.kwargs["body"]
        _verify_non_deployment_labels(config_map_body, request)

        # Make sure that an HPA is *not* created for async endpoints.
        create_hpa_call_args = (
            mock_autoscaling_client.create_namespaced_horizontal_pod_autoscaler.call_args
        )
        assert create_hpa_call_args is None

        # Make sure that an VPA is created if optimize_costs is True.
        build_endpoint_request = request.build_endpoint_request
        optimize_costs = build_endpoint_request.optimize_costs
        create_custom_object_call_args_list = (
            mock_custom_objects_client.create_namespaced_custom_object.call_args_list
        )
        delete_custom_object_call_args_list = (
            mock_custom_objects_client.delete_namespaced_custom_object.call_args_list
        )
        if optimize_costs:
            _verify_custom_object_plurals(
                call_args_list=create_custom_object_call_args_list,
                expected_plurals=["verticalpodautoscalers"],
            )
            assert delete_custom_object_call_args_list == []

        # Verify PDB labels
        create_pdb_call_args = mock_policy_client.create_namespaced_pod_disruption_budget.call_args
        pdb_body = create_pdb_call_args.kwargs["body"]
        _verify_non_deployment_labels(pdb_body, request)

        if build_endpoint_request.model_endpoint_record.endpoint_type == ModelEndpointType.SYNC:
            assert create_custom_object_call_args_list == []
            _verify_custom_object_plurals(
                call_args_list=delete_custom_object_call_args_list,
                expected_plurals=["verticalpodautoscalers"],
            )

        mock_custom_objects_client.reset_mock()


@pytest.mark.asyncio
async def test_create_streaming_endpoint_has_correct_labels_and_dest(
    k8s_endpoint_resource_delegate,
    mock_apps_client,
    mock_core_client,
    mock_autoscaling_client,
    mock_policy_client,
    mock_custom_objects_client,
    mock_get_kubernetes_cluster_version,
    create_resources_request_streaming_runnable_image: CreateOrUpdateResourcesRequest,
):
    request = create_resources_request_streaming_runnable_image
    dest = await k8s_endpoint_resource_delegate.create_or_update_resources(
        request,
        sqs_queue_name="my_queue",
        sqs_queue_url="https://my_queue",
    )
    service_name = mock_core_client.create_namespaced_service.call_args.kwargs["body"]["metadata"][
        "name"
    ]
    assert dest == service_name

    # Verify deployment labels
    create_deployment_call_args = mock_apps_client.create_namespaced_deployment.call_args
    deployment_body = create_deployment_call_args.kwargs["body"]
    _verify_deployment_labels(deployment_body, request)

    # Verify service labels
    create_service_call_args = mock_core_client.create_namespaced_service.call_args
    service_body = create_service_call_args.kwargs["body"]
    _verify_non_deployment_labels(service_body, request)

    # Verify config_map labels
    create_config_map_call_args = mock_core_client.create_namespaced_config_map.call_args
    config_map_body = create_config_map_call_args.kwargs["body"]
    _verify_non_deployment_labels(config_map_body, request)

    # Verify PDB labels
    create_pdb_call_args = mock_policy_client.create_namespaced_pod_disruption_budget.call_args
    pdb_body = create_pdb_call_args.kwargs["body"]
    _verify_non_deployment_labels(pdb_body, request)

    # Verify HPA labels
    create_hpa_call_args = (
        mock_autoscaling_client.create_namespaced_horizontal_pod_autoscaler.call_args
    )
    hpa_body = create_hpa_call_args.kwargs["body"]
    _verify_non_deployment_labels(hpa_body, request)

    # Make sure that an VPA is created if optimize_costs is True.
    build_endpoint_request = request.build_endpoint_request
    optimize_costs = build_endpoint_request.optimize_costs
    create_custom_object_call_args_list = (
        mock_custom_objects_client.create_namespaced_custom_object.call_args_list
    )
    if optimize_costs:
        _verify_custom_object_plurals(
            call_args_list=create_custom_object_call_args_list,
            expected_plurals=[
                "verticalpodautoscalers",
                "virtualservices",
                "destinationrules",
            ],
        )
    if build_endpoint_request.model_endpoint_record.endpoint_type == ModelEndpointType.SYNC:
        _verify_custom_object_plurals(
            call_args_list=create_custom_object_call_args_list,
            expected_plurals=["virtualservices", "destinationrules"],
        )

    mock_custom_objects_client.reset_mock()

    # Make sure that an VPA is created if optimize_costs is True.
    optimize_costs = request.build_endpoint_request.optimize_costs
    create_vpa_call_args = mock_custom_objects_client.create_namespaced_custom_objects.call_args
    if optimize_costs:
        assert create_vpa_call_args is not None
    else:
        assert create_vpa_call_args is None


@pytest.mark.asyncio
async def test_create_sync_endpoint_has_correct_labels_and_dest(
    k8s_endpoint_resource_delegate,
    mock_apps_client,
    mock_core_client,
    mock_autoscaling_client,
    mock_policy_client,
    mock_custom_objects_client,
    mock_get_kubernetes_cluster_version,
    create_resources_request_sync_runnable_image: CreateOrUpdateResourcesRequest,
):
    for request in [
        create_resources_request_sync_runnable_image,
    ]:
        dest = await k8s_endpoint_resource_delegate.create_or_update_resources(
            request,
            sqs_queue_name="my_queue",
            sqs_queue_url="https://my_queue,",
        )
        service_name = mock_core_client.create_namespaced_service.call_args.kwargs["body"][
            "metadata"
        ]["name"]
        assert dest == service_name

        # Verify deployment labels
        create_deployment_call_args = mock_apps_client.create_namespaced_deployment.call_args
        deployment_body = create_deployment_call_args.kwargs["body"]
        _verify_deployment_labels(deployment_body, request)

        # Verify service labels
        create_service_call_args = mock_core_client.create_namespaced_service.call_args
        service_body = create_service_call_args.kwargs["body"]
        _verify_non_deployment_labels(service_body, request)

        # Verify config_map labels
        create_config_map_call_args = mock_core_client.create_namespaced_config_map.call_args
        config_map_body = create_config_map_call_args.kwargs["body"]
        _verify_non_deployment_labels(config_map_body, request)

        # Verify HPA labels
        create_hpa_call_args = (
            mock_autoscaling_client.create_namespaced_horizontal_pod_autoscaler.call_args
        )
        hpa_body = create_hpa_call_args.kwargs["body"]
        _verify_non_deployment_labels(hpa_body, request)

        # Verify PDB labels
        create_pdb_call_args = mock_policy_client.create_namespaced_pod_disruption_budget.call_args
        pdb_body = create_pdb_call_args.kwargs["body"]
        _verify_non_deployment_labels(pdb_body, request)

        # Make sure that an VPA is created if optimize_costs is True.
        build_endpoint_request = request.build_endpoint_request
        optimize_costs = build_endpoint_request.optimize_costs
        create_custom_object_call_args_list = (
            mock_custom_objects_client.create_namespaced_custom_object.call_args_list
        )
        if optimize_costs:
            _verify_custom_object_plurals(
                call_args_list=create_custom_object_call_args_list,
                expected_plurals=[
                    "verticalpodautoscalers",
                    "virtualservices",
                    "destinationrules",
                ],
            )
        if build_endpoint_request.model_endpoint_record.endpoint_type == ModelEndpointType.SYNC:
            _verify_custom_object_plurals(
                call_args_list=create_custom_object_call_args_list,
                expected_plurals=["virtualservices", "destinationrules"],
            )

        mock_custom_objects_client.reset_mock()

    # Make sure that an VPA is created if optimize_costs is True.
    optimize_costs = (
        create_resources_request_sync_runnable_image.build_endpoint_request.optimize_costs
    )
    create_vpa_call_args = mock_custom_objects_client.create_namespaced_custom_objects.call_args
    if optimize_costs:
        assert create_vpa_call_args is not None
    else:
        assert create_vpa_call_args is None


@pytest.mark.asyncio
async def test_create_sync_endpoint_has_correct_k8s_service_type(
    k8s_endpoint_resource_delegate,
    mock_apps_client,
    mock_core_client,
    mock_autoscaling_client,
    mock_policy_client,
    mock_custom_objects_client,
    mock_get_kubernetes_cluster_version,
    create_resources_request_sync_runnable_image: CreateOrUpdateResourcesRequest,
):
    await k8s_endpoint_resource_delegate.create_or_update_resources(
        create_resources_request_sync_runnable_image,
        sqs_queue_name="my_queue",
        sqs_queue_url="https://my_queue",
    )

    # Verify service labels
    create_service_call_args = mock_core_client.create_namespaced_service.call_args
    service_body = create_service_call_args.kwargs["body"]

    assert service_body["spec"] is not None


@pytest.mark.asyncio
async def test_create_multinode_endpoint_creates_lws_and_correct_dest(
    k8s_endpoint_resource_delegate,
    mock_apps_client,
    mock_core_client,
    mock_autoscaling_client,
    mock_policy_client,
    mock_custom_objects_client,
    mock_get_kubernetes_cluster_version,
    create_resources_request_streaming_runnable_image: CreateOrUpdateResourcesRequest,
    model_bundle_5: ModelBundle,
):
    # Patch model bundle so that it supports multinode
    model_bundle_5.flavor.worker_env = {"fake_env": "fake_value"}
    model_bundle_5.flavor.worker_command = ["fake_command"]
    create_resources_request_streaming_runnable_image.build_endpoint_request.model_endpoint_record.current_model_bundle = (
        model_bundle_5
    )
    create_resources_request_streaming_runnable_image.build_endpoint_request.model_endpoint_record.endpoint_type = (
        ModelEndpointType.STREAMING
    )

    create_resources_request_streaming_runnable_image.build_endpoint_request.nodes_per_worker = 2
    dest = await k8s_endpoint_resource_delegate.create_or_update_resources(
        create_resources_request_streaming_runnable_image,
        sqs_queue_name="my_queue",
        sqs_queue_url="https://my_queue",
    )
    service_name = mock_core_client.create_namespaced_service.call_args.kwargs["body"]["metadata"][
        "name"
    ]
    assert dest == service_name
    # Verify call to custom objects client with LWS is made
    create_custom_objects_call_args_list = (
        mock_custom_objects_client.create_namespaced_custom_object.call_args_list
    )
    assert any(
        call_args.kwargs["group"] == "leaderworkerset.x-k8s.io"
        for call_args in create_custom_objects_call_args_list
    )


@pytest.mark.asyncio
async def test_create_endpoint_raises_k8s_endpoint_resource_delegate(
    k8s_endpoint_resource_delegate,
    create_resources_request_sync_pytorch: CreateOrUpdateResourcesRequest,
):
    k8s_endpoint_resource_delegate.__setattr__(
        "_create_or_update_resources",
        AsyncMock(side_effect=ApiException),
    )
    with pytest.raises(EndpointResourceInfraException):
        await k8s_endpoint_resource_delegate.create_or_update_resources(
            create_resources_request_sync_pytorch,
            sqs_queue_name="my_queue",
            sqs_queue_url="https://my_queue",
        )


@pytest.mark.asyncio
async def test_get_resources_raises_k8s_endpoint_resource_delegate(
    k8s_endpoint_resource_delegate,
):
    k8s_endpoint_resource_delegate.__setattr__(
        "_get_resources",
        AsyncMock(side_effect=ApiException),
    )
    with pytest.raises(EndpointResourceInfraException):
        await k8s_endpoint_resource_delegate.get_resources(
            endpoint_id="", deployment_name="", endpoint_type=ModelEndpointType.ASYNC
        )


@pytest.mark.asyncio
async def test_get_resources_async_success(
    k8s_endpoint_resource_delegate,
    mock_apps_client,
    mock_core_client,
    mock_autoscaling_client,
    mock_policy_client,
    mock_custom_objects_client,
):
    # Pretend that LWS get gives an ApiException since it doesn't exist
    mock_custom_objects_client.get_namespaced_custom_object = AsyncMock(side_effect=ApiException)
    k8s_endpoint_resource_delegate.__setattr__(
        "_get_common_endpoint_params",
        Mock(
            return_value=dict(
                aws_role="test_aws_role",
                results_s3_bucket="test_bucket",
                labels={},
                cpus="1",
                gpus=1,
                gpu_type="nvidia-tesla-t4",
                memory="8G",
                storage="10G",
                image="test_image",
            ),
        ),
    )
    k8s_endpoint_resource_delegate.__setattr__(
        "_get_async_autoscaling_params",
        Mock(
            return_value=dict(
                min_workers=1,
                max_workers=3,
                per_worker=2,
                concurrent_requests_per_worker=1,
            )
        ),
    )
    k8s_endpoint_resource_delegate.__setattr__(
        "_get_main_container",
        Mock(return_value=FakeK8sDeploymentContainer(env=[])),
    )
    k8s_endpoint_resource_delegate.__setattr__(
        "_get_launch_container",
        Mock(
            return_value=FakeK8sDeploymentContainer(
                env=[FakeK8sEnvVar(name="PREWARM", value="true")]
            )
        ),
    )
    k8s_endpoint_resource_delegate.__setattr__(
        "_translate_k8s_config_maps_to_user_config_data",
        Mock(
            return_value=ModelEndpointUserConfigState(
                app_config=None,
                endpoint_config=ModelEndpointConfig(
                    endpoint_name="test_endpoint",
                    bundle_name="test_bundle",
                    post_inference_hooks=["callback"],
                ),
            )
        ),
    )
    infra_state = await k8s_endpoint_resource_delegate.get_resources(
        endpoint_id="", deployment_name="", endpoint_type=ModelEndpointType.ASYNC
    )
    assert infra_state


@pytest.mark.asyncio
async def test_get_resources_sync_success(
    k8s_endpoint_resource_delegate,
    mock_apps_client,
    mock_core_client,
    mock_autoscaling_client,
    mock_policy_client,
    mock_custom_objects_client,
):
    # Pretend that LWS get and keda get give an ApiException
    mock_custom_objects_client.get_namespaced_custom_object = AsyncMock(side_effect=ApiException)
    k8s_endpoint_resource_delegate.__setattr__(
        "_get_common_endpoint_params",
        Mock(
            return_value=dict(
                aws_role="test_aws_role",
                results_s3_bucket="test_bucket",
                labels={},
                cpus="1",
                gpus=1,
                gpu_type="nvidia-tesla-t4",
                memory="8G",
                storage="10G",
                image="test_image",
            )
        ),
    )
    k8s_endpoint_resource_delegate.__setattr__(
        "_get_sync_autoscaling_params",
        Mock(
            return_value=dict(
                min_workers=1,
                max_workers=3,
                per_worker=2,
                concurrent_requests_per_worker=200,
            )
        ),
    )
    k8s_endpoint_resource_delegate.__setattr__(
        "_get_main_container", Mock(return_value=FakeK8sDeploymentContainer(env=[]))
    )
    k8s_endpoint_resource_delegate.__setattr__(
        "_get_launch_container", Mock(return_value=FakeK8sDeploymentContainer(env=[]))
    )
    k8s_endpoint_resource_delegate.__setattr__(
        "_translate_k8s_config_maps_to_user_config_data",
        Mock(
            return_value=ModelEndpointUserConfigState(
                app_config=None,
                endpoint_config=ModelEndpointConfig(
                    endpoint_name="test_endpoint",
                    bundle_name="test_bundle",
                    post_inference_hooks=["callback"],
                ),
            )
        ),
    )
    infra_state = await k8s_endpoint_resource_delegate.get_resources(
        endpoint_id="", deployment_name="", endpoint_type=ModelEndpointType.SYNC
    )
    assert infra_state


@pytest.mark.asyncio
async def test_get_resources_multinode_success(
    k8s_endpoint_resource_delegate,
    mock_apps_client,
    mock_core_client,
    mock_autoscaling_client,
    mock_policy_client,
    mock_custom_objects_client,
):
    k8s_endpoint_resource_delegate.__setattr__(
        "_translate_k8s_config_maps_to_user_config_data",
        Mock(
            return_value=ModelEndpointUserConfigState(
                app_config=None,
                endpoint_config=ModelEndpointConfig(
                    endpoint_name="test_endpoint",
                    bundle_name="test_bundle",
                    post_inference_hooks=["callback"],
                ),
            )
        ),
    )

    mock_custom_objects_client.get_namespaced_custom_object = AsyncMock(
        return_value=EXAMPLE_LWS_CONFIG
    )

    infra_state = await k8s_endpoint_resource_delegate.get_resources(
        endpoint_id="", deployment_name="", endpoint_type=ModelEndpointType.STREAMING
    )
    assert infra_state
    assert infra_state.resource_state.nodes_per_worker == 2


@pytest.mark.asyncio
async def test_delete_resources_invalid_endpoint_type_returns_false(
    k8s_endpoint_resource_delegate,
):
    deleted = await k8s_endpoint_resource_delegate.delete_resources(
        endpoint_id="",
        deployment_name="",
        endpoint_type=None,  # type: ignore
    )
    assert not deleted


@pytest.mark.asyncio
async def test_delete_resources_async_success(
    k8s_endpoint_resource_delegate,
    mock_apps_client,
    mock_core_client,
    mock_autoscaling_client,
    mock_policy_client,
    mock_custom_objects_client,
):
    deleted = await k8s_endpoint_resource_delegate.delete_resources(
        endpoint_id="", deployment_name="", endpoint_type=ModelEndpointType.ASYNC
    )
    assert deleted


@pytest.mark.asyncio
async def test_delete_resources_sync_success(
    k8s_endpoint_resource_delegate,
    mock_apps_client,
    mock_core_client,
    mock_autoscaling_client,
    mock_policy_client,
    mock_custom_objects_client,
):
    deleted = await k8s_endpoint_resource_delegate.delete_resources(
        endpoint_id="", deployment_name="", endpoint_type=ModelEndpointType.SYNC
    )
    assert deleted


@pytest.mark.asyncio
async def test_delete_resources_async_ignores_pvc_delete_failure(
    k8s_endpoint_resource_delegate,
    mock_apps_client,
    mock_core_client,
    mock_autoscaling_client,
    mock_policy_client,
    mock_custom_objects_client,
):
    mock_core_client.delete_namespaced_persistent_volume_claim.side_effect = ApiException(
        status=500
    )

    deleted = await k8s_endpoint_resource_delegate.delete_resources(
        endpoint_id="", deployment_name="", endpoint_type=ModelEndpointType.ASYNC
    )

    assert deleted


@pytest.mark.asyncio
async def test_delete_resources_sync_ignores_pvc_delete_failure(
    k8s_endpoint_resource_delegate,
    mock_apps_client,
    mock_core_client,
    mock_autoscaling_client,
    mock_policy_client,
    mock_custom_objects_client,
):
    mock_core_client.delete_namespaced_persistent_volume_claim.side_effect = ApiException(
        status=500
    )

    deleted = await k8s_endpoint_resource_delegate.delete_resources(
        endpoint_id="", deployment_name="", endpoint_type=ModelEndpointType.SYNC
    )

    assert deleted


@pytest.mark.asyncio
async def test_delete_resources_multinode_success(
    k8s_endpoint_resource_delegate,
    mock_apps_client,
    mock_core_client,
    mock_autoscaling_client,
    mock_policy_client,
    mock_custom_objects_client,
):
    mock_custom_objects_client.get_namespaced_custom_object = AsyncMock(
        return_value=EXAMPLE_LWS_CONFIG
    )
    mock_custom_objects_client.delete_namespaced_custom_object = AsyncMock()
    deleted = await k8s_endpoint_resource_delegate.delete_resources(
        endpoint_id="", deployment_name="", endpoint_type=ModelEndpointType.STREAMING
    )
    assert deleted
    delete_called_for_lws = False
    for call_args in mock_custom_objects_client.delete_namespaced_custom_object.call_args_list:
        # 'group' is kwargs in delete_namespaced_custom_object
        if call_args[1]["group"] == "leaderworkerset.x-k8s.io":
            delete_called_for_lws = True
            break
    assert delete_called_for_lws


@pytest.mark.asyncio
async def test_create_persistent_volume_claim_409_is_success(
    k8s_endpoint_resource_delegate,
    mock_core_client,
):
    pvc = {"metadata": {"name": "test-pvc"}}

    mock_core_client.create_namespaced_persistent_volume_claim.side_effect = ApiException(
        status=409
    )

    await k8s_endpoint_resource_delegate._create_persistent_volume_claim(pvc, "test-pvc")

    mock_core_client.create_namespaced_persistent_volume_claim.assert_called_once_with(
        namespace=hmi_config.endpoint_namespace,
        body=pvc,
    )
    mock_core_client.patch_namespaced_persistent_volume_claim.assert_not_called()


@pytest.mark.asyncio
async def test_delete_persistent_volume_claim_404_is_success(
    k8s_endpoint_resource_delegate,
    mock_core_client,
):
    mock_core_client.delete_namespaced_persistent_volume_claim.side_effect = ApiException(
        status=404
    )

    deleted = await k8s_endpoint_resource_delegate._delete_persistent_volume_claim(
        endpoint_id="end_test"
    )

    assert deleted


@pytest.mark.asyncio
async def test_delete_persistent_volume_claim_non_404_failure(
    k8s_endpoint_resource_delegate,
    mock_core_client,
):
    mock_core_client.delete_namespaced_persistent_volume_claim.side_effect = ApiException(
        status=500
    )

    deleted = await k8s_endpoint_resource_delegate._delete_persistent_volume_claim(
        endpoint_id="end_test"
    )

    assert not deleted


@pytest.mark.asyncio
async def test_create_pdb(
    k8s_endpoint_resource_delegate,
    mock_policy_client,
):
    # Mock the necessary objects and functions
    pdb = {
        "metadata": {"name": "test-pdb", "namespace": "test-namespace"},
        "spec": {"maxUnavailable": "50%"},
    }
    name = "test-pdb"

    # Test successful creation
    await k8s_endpoint_resource_delegate._create_pdb(pdb, name)

    mock_policy_client.create_namespaced_pod_disruption_budget.assert_called_once_with(
        namespace=hmi_config.endpoint_namespace,
        body=pdb,
    )

    # Test creation when PDB already exists
    mock_policy_client.create_namespaced_pod_disruption_budget.side_effect = ApiException(
        status=409
    )

    existing_pdb = Mock()
    existing_pdb.metadata.resource_version = "123"
    mock_policy_client.read_namespaced_pod_disruption_budget.return_value = existing_pdb

    await k8s_endpoint_resource_delegate._create_pdb(pdb, name)

    mock_policy_client.read_namespaced_pod_disruption_budget.assert_called_once_with(
        name=name, namespace=hmi_config.endpoint_namespace
    )

    expected_replace_pdb = pdb.copy()
    expected_replace_pdb["metadata"]["resourceVersion"] = "123"

    mock_policy_client.replace_namespaced_pod_disruption_budget.assert_called_once_with(
        name=name,
        namespace=hmi_config.endpoint_namespace,
        body=expected_replace_pdb,
    )

    # Test creation with other API exception
    mock_policy_client.create_namespaced_pod_disruption_budget.side_effect = ApiException(
        status=500
    )

    with pytest.raises(ApiException):
        await k8s_endpoint_resource_delegate._create_pdb(pdb, name)


@pytest.mark.asyncio
async def test_restart_deployment(
    k8s_endpoint_resource_delegate,
    mock_apps_client,
):
    await k8s_endpoint_resource_delegate.restart_deployment(deployment_name="test_deployment")
    mock_apps_client.patch_namespaced_deployment.assert_called_once_with(
        name="test_deployment",
        namespace=hmi_config.endpoint_namespace,
        body=ANY,
    )


@pytest.mark.asyncio
async def test_get_async_autoscaling_params(k8s_endpoint_resource_delegate):
    deployment_config = MagicMock()
    celery_forwarder = MagicMock()
    main_container = MagicMock()  # empty because it's not used
    celery_forwarder.name = "celery-forwarder"
    celery_forwarder.command = ["a", "b", "--num-workers", "24", "c", "d"]
    deployment_config.metadata.annotations = {
        "celery.scaleml.autoscaler/minWorkers": 1,
        "celery.scaleml.autoscaler/maxWorkers": 2,
        "celery.scaleml.autoscaler/perWorker": 5,
    }
    deployment_config.spec.template.spec.containers = [celery_forwarder, main_container]
    autoscaling_params = K8SEndpointResourceDelegate._get_async_autoscaling_params(
        deployment_config
    )
    assert autoscaling_params == dict(
        min_workers=1,
        max_workers=2,
        per_worker=5,
        concurrent_requests_per_worker=24,
    )

    # test old format for backwards compatibility
    celery_forwarder.command = ["a", "b", "c", "d"]
    autoscaling_params = K8SEndpointResourceDelegate._get_async_autoscaling_params(
        deployment_config
    )
    assert autoscaling_params == dict(
        min_workers=1,
        max_workers=2,
        per_worker=5,
        concurrent_requests_per_worker=1,
    )


def test_add_pod_metadata_env_to_container():
    """Test that pod metadata env vars are added correctly."""
    container = {"env": [{"name": "EXISTING_VAR", "value": "existing_value"}]}

    add_pod_metadata_env_to_container(container)

    env_names = {env["name"] for env in container["env"]}
    assert "EXISTING_VAR" in env_names
    assert "POD_UID" in env_names
    assert "POD_NAME" in env_names
    assert "NODE_NAME" in env_names

    # Verify the fieldRef values
    pod_uid_env = next(e for e in container["env"] if e["name"] == "POD_UID")
    assert pod_uid_env["valueFrom"]["fieldRef"]["fieldPath"] == "metadata.uid"

    pod_name_env = next(e for e in container["env"] if e["name"] == "POD_NAME")
    assert pod_name_env["valueFrom"]["fieldRef"]["fieldPath"] == "metadata.name"

    node_name_env = next(e for e in container["env"] if e["name"] == "NODE_NAME")
    assert node_name_env["valueFrom"]["fieldRef"]["fieldPath"] == "spec.nodeName"
