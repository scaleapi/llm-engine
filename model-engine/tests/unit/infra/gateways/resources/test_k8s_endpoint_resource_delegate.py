from typing import Any, Dict, List
from unittest.mock import AsyncMock, Mock, patch

import pytest
from kubernetes_asyncio.client.rest import ApiException
from model_engine_server.common.config import hmi_config
from model_engine_server.common.dtos.resource_manager import CreateOrUpdateResourcesRequest
from model_engine_server.common.env_vars import GIT_TAG
from model_engine_server.domain.entities import (
    ModelBundle,
    ModelEndpointConfig,
    ModelEndpointType,
    ModelEndpointUserConfigState,
)
from model_engine_server.domain.entities.model_bundle_entity import (
    WORKER_COMMAND_METADATA_KEY,
    WORKER_ENV_METADATA_KEY,
)
from model_engine_server.domain.exceptions import EndpointResourceInfraException
from model_engine_server.infra.gateways.resources.k8s_endpoint_resource_delegate import (
    DATADOG_ENV_VAR,
    K8SEndpointResourceDelegate,
    add_datadog_env_to_container,
    get_main_container_from_deployment_template,
    load_k8s_yaml,
)
from model_engine_server.infra.gateways.resources.k8s_resource_types import (
    DictStrInt,
    DictStrStr,
    ResourceArguments,
)
from tests.unit.infra.gateways.k8s_fake_objects import FakeK8sDeploymentContainer, FakeK8sEnvVar

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
def lws_config():
    return {'apiVersion': 'leaderworkerset.x-k8s.io/v1', 'kind': 'LeaderWorkerSet', 'metadata': {'creationTimestamp': '2024-07-27T01:38:14Z', 'generation': 1, 'labels': {'created_by': '5f2887089e45140017c85dec', 'endpoint_id': 'end_cqi4v12d6mt002nap72g', 'endpoint_name': 'llama-2-7b.test', 'env': 'training', 'managed-by': 'model-engine', 'owner': '5f2887089e45140017c85dec', 'product': 'testing', 'tags.datadoghq.com/env': 'training', 'tags.datadoghq.com/service': 'llama-2-7b.test', 'tags.datadoghq.com/version': '05fb96620692a205e52d33980eff475d6a52748a', 'team': 'infra', 'use_scale_launch_endpoint_network_policy': 'true', 'user_id': '5f2887089e45140017c85dec'}, 'managedFields': [{'apiVersion': 'leaderworkerset.x-k8s.io/v1', 'fieldsType': 'FieldsV1', 'fieldsV1': {'f:metadata': {'f:labels': {'.': {}, 'f:created_by': {}, 'f:endpoint_id': {}, 'f:endpoint_name': {}, 'f:env': {}, 'f:managed-by': {}, 'f:owner': {}, 'f:product': {}, 'f:tags.datadoghq.com/env': {}, 'f:tags.datadoghq.com/service': {}, 'f:tags.datadoghq.com/version': {}, 'f:team': {}, 'f:use_scale_launch_endpoint_network_policy': {}, 'f:user_id': {}}}, 'f:spec': {'.': {}, 'f:leaderWorkerTemplate': {'.': {}, 'f:leaderTemplate': {'.': {}, 'f:metadata': {'.': {}, 'f:annotations': {'.': {}, 'f:ad.datadoghq.com/main.logs': {}, 'f:kubernetes.io/change-cause': {}}, 'f:labels': {'.': {}, 'f:app': {}, 'f:created_by': {}, 'f:endpoint_id': {}, 'f:endpoint_name': {}, 'f:env': {}, 'f:managed-by': {}, 'f:owner': {}, 'f:product': {}, 'f:sidecar.istio.io/inject': {}, 'f:tags.datadoghq.com/env': {}, 'f:tags.datadoghq.com/service': {}, 'f:tags.datadoghq.com/version': {}, 'f:team': {}, 'f:use_scale_launch_endpoint_network_policy': {}, 'f:user_id': {}, 'f:version': {}}}, 'f:spec': {'.': {}, 'f:affinity': {'.': {}, 'f:podAffinity': {'.': {}, 'f:preferredDuringSchedulingIgnoredDuringExecution': {}}}, 'f:containers': {}, 'f:nodeSelector': {}, 'f:priorityClassName': {}, 'f:serviceAccount': {}, 'f:terminationGracePeriodSeconds': {}, 'f:tolerations': {}, 'f:volumes': {}}}, 'f:restartPolicy': {}, 'f:size': {}, 'f:workerTemplate': {'.': {}, 'f:metadata': {'.': {}, 'f:annotations': {'.': {}, 'f:ad.datadoghq.com/main.logs': {}, 'f:kubernetes.io/change-cause': {}}, 'f:labels': {'.': {}, 'f:app': {}, 'f:created_by': {}, 'f:endpoint_id': {}, 'f:endpoint_name': {}, 'f:env': {}, 'f:managed-by': {}, 'f:owner': {}, 'f:product': {}, 'f:sidecar.istio.io/inject': {}, 'f:tags.datadoghq.com/env': {}, 'f:tags.datadoghq.com/service': {}, 'f:tags.datadoghq.com/version': {}, 'f:team': {}, 'f:use_scale_launch_endpoint_network_policy': {}, 'f:user_id': {}, 'f:version': {}}}, 'f:spec': {'.': {}, 'f:affinity': {'.': {}, 'f:podAffinity': {'.': {}, 'f:preferredDuringSchedulingIgnoredDuringExecution': {}}}, 'f:containers': {}, 'f:nodeSelector': {}, 'f:priorityClassName': {}, 'f:serviceAccount': {}, 'f:terminationGracePeriodSeconds': {}, 'f:tolerations': {}, 'f:volumes': {}}}}, 'f:replicas': {}, 'f:startupPolicy': {}}}, 'manager': 'OpenAPI-Generator', 'operation': 'Update', 'time': '2024-07-27T01:38:14Z'}, {'apiVersion': 'leaderworkerset.x-k8s.io/v1', 'fieldsType': 'FieldsV1', 'fieldsV1': {'f:status': {'.': {}, 'f:conditions': {}, 'f:hpaPodSelector': {}}}, 'manager': 'manager', 'operation': 'Update', 'subresource': 'status', 'time': '2024-07-27T01:38:14Z'}], 'name': 'launch-endpoint-id-end-cqi4v12d6mt002nap72g', 'namespace': 'scale-deploy', 'resourceVersion': '2289583184', 'uid': '1d66ad78-3148-41b3-83fd-fb71d7656fb1'}, 'spec': {'leaderWorkerTemplate': {'leaderTemplate': {'metadata': {'annotations': {'ad.datadoghq.com/main.logs': '[{"service": "llama-2-7b.test", "source": "python"}]', 'kubernetes.io/change-cause': 'Deployment at 2024-07-27 01:38:13.814158 UTC. Using deployment constructed from model bundle ID bun_cqi4v12d6mt002nap720, model bundle name llama-2-7b.test, endpoint ID end_cqi4v12d6mt002nap72g'}, 'labels': {'app': 'launch-endpoint-id-end-cqi4v12d6mt002nap72g', 'created_by': '5f2887089e45140017c85dec', 'endpoint_id': 'end_cqi4v12d6mt002nap72g', 'endpoint_name': 'llama-2-7b.test', 'env': 'training', 'managed-by': 'model-engine', 'owner': '5f2887089e45140017c85dec', 'product': 'testing', 'sidecar.istio.io/inject': 'false', 'tags.datadoghq.com/env': 'training', 'tags.datadoghq.com/service': 'llama-2-7b.test', 'tags.datadoghq.com/version': '05fb96620692a205e52d33980eff475d6a52748a', 'team': 'infra', 'use_scale_launch_endpoint_network_policy': 'true', 'user_id': '5f2887089e45140017c85dec', 'version': 'v1'}}, 'spec': {'affinity': {'podAffinity': {'preferredDuringSchedulingIgnoredDuringExecution': [{'podAffinityTerm': {'labelSelector': {'matchExpressions': [{'key': 'app', 'operator': 'In', 'values': ['launch-endpoint-id-end-cqi4v12d6mt002nap72g']}]}, 'topologyKey': 'kubernetes.io/hostname'}, 'weight': 1}, {'podAffinityTerm': {'labelSelector': {'matchExpressions': [{'key': '3d45a96760a60018eb4a9d874e919aef', 'operator': 'In', 'values': ['True']}]}, 'topologyKey': 'kubernetes.io/hostname'}, 'weight': 100}]}}, 'containers': [{'command': ['/usr/bin/dumb-init', '--', 'ddtrace-run', 'python', '-m', 'model_engine_server.inference.forwarding.http_forwarder', '--config', '/workspace/model-engine/model_engine_server/inference/configs/service--http_forwarder.yaml', '--port', '5000', '--num-workers', '2', '--set', 'forwarder.sync.predict_route=/predict', '--set', 'forwarder.stream.predict_route=/stream', '--set', 'forwarder.sync.healthcheck_route=/health', '--set', 'forwarder.stream.healthcheck_route=/health'], 'env': [{'name': 'DD_TRACE_ENABLED', 'value': 'True'}, {'name': 'DD_REMOTE_CONFIGURATION_ENABLED', 'value': 'false'}, {'name': 'DD_SERVICE', 'value': 'llama-2-7b.test'}, {'name': 'DD_ENV', 'value': 'training'}, {'name': 'DD_VERSION', 'value': '05fb96620692a205e52d33980eff475d6a52748a'}, {'name': 'DD_AGENT_HOST', 'valueFrom': {'fieldRef': {'fieldPath': 'status.hostIP'}}}, {'name': 'AWS_PROFILE', 'value': 'ml-worker'}, {'name': 'AWS_CONFIG_FILE', 'value': '/opt/.aws/config'}, {'name': 'RESULTS_S3_BUCKET', 'value': 'scale-ml'}, {'name': 'BASE_PATH', 'value': '/workspace'}, {'name': 'ML_INFRA_SERVICES_CONFIG_PATH', 'value': '/workspace/model-engine-internal/resources/configs/infra_config_training.yaml'}], 'image': '692474966980.dkr.ecr.us-west-2.amazonaws.com/model-engine-internal:05fb96620692a205e52d33980eff475d6a52748a', 'imagePullPolicy': 'IfNotPresent', 'name': 'http-forwarder', 'ports': [{'containerPort': 5000, 'name': 'http', 'protocol': 'TCP'}], 'readinessProbe': {'httpGet': {'path': '/readyz', 'port': 5000}, 'initialDelaySeconds': 10, 'periodSeconds': 5, 'timeoutSeconds': 5}, 'resources': {'limits': {'cpu': '1', 'ephemeral-storage': '1G', 'memory': '2Gi'}, 'requests': {'cpu': '1', 'ephemeral-storage': '100M', 'memory': '100M'}}, 'volumeMounts': [{'mountPath': '/opt/.aws/config', 'name': 'config-volume', 'subPath': 'config'}, {'mountPath': '/workspace/user_config', 'name': 'user-config', 'subPath': 'raw_data'}, {'mountPath': '/workspace/endpoint_config', 'name': 'endpoint-config', 'subPath': 'raw_data'}]}, {'command': ['/bin/bash', '-c', "./s5cmd --numworkers 512 cp --concurrency 10 --include '*.model' --include '*.json' --include '*.safetensors' --exclude 'optimizer*' s3://scale-ml/hosted-model-inference/fine_tuned_weights/5f2887089e45140017c85dec/models--llama-2-7b.demo.2023-08-10-22-38-10/* model_files;/workspace/init_ray.sh leader --ray_cluster_size=$RAY_CLUSTER_SIZE --own_address=$K8S_OWN_POD_NAME.$K8S_LWS_NAME.$K8S_OWN_NAMESPACE.svc.cluster.local;python -m vllm_server --model model_files --tensor-parallel-size 1 --port 5005 --disable-log-requests--enforce-eager"], 'env': [{'name': 'VLLM_HOST_IP', 'value': '$(K8S_LEADER_NAME).$(K8S_LWS_NAME).$(K8S_OWN_NAMESPACE).svc.cluster.local'}, {'name': 'NCCL_SOCKET_IFNAME', 'value': 'eth0'}, {'name': 'GLOO_SOCKET_IFNAME', 'value': 'eth0'}, {'name': 'NCCL_DEBUG', 'value': 'INFO'}, {'name': 'VLLM_LOGGING_LEVEL', 'value': 'INFO'}, {'name': 'AWS_PROFILE', 'value': 'ml-worker'}, {'name': 'AWS_CONFIG_FILE', 'value': '/opt/.aws/config'}, {'name': 'K8S_OWN_POD_NAME', 'valueFrom': {'fieldRef': {'fieldPath': 'metadata.name'}}}, {'name': 'K8S_OWN_NAMESPACE', 'valueFrom': {'fieldRef': {'fieldPath': 'metadata.namespace'}}}, {'name': 'K8S_LWS_NAME', 'valueFrom': {'fieldRef': {'fieldPath': "metadata.labels['leaderworkerset.sigs.k8s.io/name']"}}}, {'name': 'K8S_LWS_LEADER_NAME', 'valueFrom': {'fieldRef': {'fieldPath': "metadata.labels['leaderworkerset.sigs.k8s.io/leader-name']"}}}, {'name': 'K8S_LWS_CLUSTER_SIZE', 'valueFrom': {'fieldRef': {'fieldPath': "metadata.annotations['leaderworkerset.sigs.k8s.io/size']"}}}, {'name': 'DD_TRACE_ENABLED', 'value': 'true'}, {'name': 'DD_SERVICE', 'value': 'llama-2-7b.test'}, {'name': 'DD_ENV', 'value': 'training'}, {'name': 'DD_VERSION', 'value': '05fb96620692a205e52d33980eff475d6a52748a'}, {'name': 'DD_AGENT_HOST', 'valueFrom': {'fieldRef': {'fieldPath': 'status.hostIP'}}}], 'image': '692474966980.dkr.ecr.us-west-2.amazonaws.com/vllm:0.5.3.post1', 'imagePullPolicy': 'IfNotPresent', 'name': 'lws_leader', 'ports': [{'containerPort': 5005, 'name': 'http', 'protocol': 'TCP'}], 'readinessProbe': {'httpGet': {'path': '/health', 'port': 5005}, 'initialDelaySeconds': 10, 'periodSeconds': 5, 'timeoutSeconds': 5}, 'resources': {'limits': {'cpu': '10', 'ephemeral-storage': '94Gi', 'memory': '40Gi', 'nvidia.com/gpu': '1'}, 'requests': {'cpu': '10', 'ephemeral-storage': '94Gi', 'memory': '40Gi', 'nvidia.com/gpu': '1'}}, 'volumeMounts': [{'mountPath': '/opt/.aws/config', 'name': 'config-volume', 'subPath': 'config'}, {'mountPath': '/dev/shm', 'name': 'dshm'}, {'mountPath': '/app/user_config', 'name': 'user-config', 'subPath': 'raw_data'}, {'mountPath': '/app/endpoint_config', 'name': 'endpoint-config', 'subPath': 'raw_data'}]}], 'nodeSelector': {'k8s.amazonaws.com/accelerator': 'nvidia-ampere-a10', 'node-lifecycle': 'normal'}, 'priorityClassName': 'model-engine-high-priority', 'serviceAccount': 'ml-worker', 'terminationGracePeriodSeconds': 600, 'tolerations': [{'effect': 'NoSchedule', 'key': 'nvidia.com/gpu', 'operator': 'Exists'}], 'volumes': [{'configMap': {'name': 'ml-worker-config'}, 'name': 'config-volume'}, {'configMap': {'name': 'launch-endpoint-id-end-cqi4v12d6mt002nap72g'}, 'name': 'user-config'}, {'configMap': {'name': 'launch-endpoint-id-end-cqi4v12d6mt002nap72g-endpoint-config'}, 'name': 'endpoint-config'}, {'emptyDir': {'medium': 'Memory'}, 'name': 'dshm'}]}}, 'restartPolicy': 'RecreateGroupOnPodRestart', 'size': 2, 'workerTemplate': {'metadata': {'annotations': {'ad.datadoghq.com/main.logs': '[{"service": "llama-2-7b.test", "source": "python"}]', 'kubernetes.io/change-cause': 'Deployment at 2024-07-27 01:38:13.814158 UTC. Using deployment constructed from model bundle ID bun_cqi4v12d6mt002nap720, model bundle name llama-2-7b.test, endpoint ID end_cqi4v12d6mt002nap72g'}, 'labels': {'app': 'launch-endpoint-id-end-cqi4v12d6mt002nap72g', 'created_by': '5f2887089e45140017c85dec', 'endpoint_id': 'end_cqi4v12d6mt002nap72g', 'endpoint_name': 'llama-2-7b.test', 'env': 'training', 'managed-by': 'model-engine', 'owner': '5f2887089e45140017c85dec', 'product': 'testing', 'sidecar.istio.io/inject': 'false', 'tags.datadoghq.com/env': 'training', 'tags.datadoghq.com/service': 'llama-2-7b.test', 'tags.datadoghq.com/version': '05fb96620692a205e52d33980eff475d6a52748a', 'team': 'infra', 'use_scale_launch_endpoint_network_policy': 'true', 'user_id': '5f2887089e45140017c85dec', 'version': 'v1'}}, 'spec': {'affinity': {'podAffinity': {'preferredDuringSchedulingIgnoredDuringExecution': [{'podAffinityTerm': {'labelSelector': {'matchExpressions': [{'key': 'app', 'operator': 'In', 'values': ['launch-endpoint-id-end-cqi4v12d6mt002nap72g']}]}, 'topologyKey': 'kubernetes.io/hostname'}, 'weight': 1}, {'podAffinityTerm': {'labelSelector': {'matchExpressions': [{'key': '3d45a96760a60018eb4a9d874e919aef', 'operator': 'In', 'values': ['True']}]}, 'topologyKey': 'kubernetes.io/hostname'}, 'weight': 100}]}}, 'containers': [{'command': ['/bin/bash', '-c', "./s5cmd --numworkers 512 cp --concurrency 10 --include '*.model' --include '*.json' --include '*.safetensors' --exclude 'optimizer*' s3://scale-ml/hosted-model-inference/fine_tuned_weights/5f2887089e45140017c85dec/models--llama-2-7b.demo.2023-08-10-22-38-10/* model_files;/workspace/init_ray.sh worker --ray_cluster_size=$RAY_CLUSTER_SIZE --ray_address=$K8S_LEADER_NAME.$K8S_LWS_NAME.$K8S_OWN_NAMESPACE.svc.cluster.local --own_address=$K8S_OWN_POD_NAME.$K8S_LWS_NAME.$K8S_OWN_NAMESPACE.svc.cluster.local"], 'env': [{'name': 'VLLM_HOST_IP', 'value': '$(K8S_LEADER_NAME).$(K8S_LWS_NAME).$(K8S_OWN_NAMESPACE).svc.cluster.local'}, {'name': 'NCCL_SOCKET_IFNAME', 'value': 'eth0'}, {'name': 'GLOO_SOCKET_IFNAME', 'value': 'eth0'}, {'name': 'NCCL_DEBUG', 'value': 'INFO'}, {'name': 'VLLM_LOGGING_LEVEL', 'value': 'INFO'}, {'name': 'AWS_PROFILE', 'value': 'ml-worker'}, {'name': 'AWS_CONFIG_FILE', 'value': '/opt/.aws/config'}, {'name': 'K8S_OWN_POD_NAME', 'valueFrom': {'fieldRef': {'fieldPath': 'metadata.name'}}}, {'name': 'K8S_OWN_NAMESPACE', 'valueFrom': {'fieldRef': {'fieldPath': 'metadata.namespace'}}}, {'name': 'K8S_LWS_NAME', 'valueFrom': {'fieldRef': {'fieldPath': "metadata.labels['leaderworkerset.sigs.k8s.io/name']"}}}, {'name': 'K8S_LWS_LEADER_NAME', 'valueFrom': {'fieldRef': {'fieldPath': "metadata.labels['leaderworkerset.sigs.k8s.io/leader-name']"}}}, {'name': 'K8S_LWS_CLUSTER_SIZE', 'valueFrom': {'fieldRef': {'fieldPath': "metadata.annotations['leaderworkerset.sigs.k8s.io/size']"}}}, {'name': 'DD_TRACE_ENABLED', 'value': 'true'}, {'name': 'DD_SERVICE', 'value': 'llama-2-7b.test'}, {'name': 'DD_ENV', 'value': 'training'}, {'name': 'DD_VERSION', 'value': '05fb96620692a205e52d33980eff475d6a52748a'}, {'name': 'DD_AGENT_HOST', 'valueFrom': {'fieldRef': {'fieldPath': 'status.hostIP'}}}], 'image': '692474966980.dkr.ecr.us-west-2.amazonaws.com/vllm:0.5.3.post1', 'imagePullPolicy': 'IfNotPresent', 'name': 'lws_worker', 'ports': [{'containerPort': 5005, 'name': 'http', 'protocol': 'TCP'}], 'resources': {'limits': {'cpu': '10', 'ephemeral-storage': '94Gi', 'memory': '40Gi', 'nvidia.com/gpu': '1'}, 'requests': {'cpu': '10', 'ephemeral-storage': '94Gi', 'memory': '40Gi', 'nvidia.com/gpu': '1'}}, 'volumeMounts': [{'mountPath': '/opt/.aws/config', 'name': 'config-volume', 'subPath': 'config'}, {'mountPath': '/dev/shm', 'name': 'dshm'}, {'mountPath': '/app/user_config', 'name': 'user-config', 'subPath': 'raw_data'}, {'mountPath': '/app/endpoint_config', 'name': 'endpoint-config', 'subPath': 'raw_data'}]}], 'nodeSelector': {'k8s.amazonaws.com/accelerator': 'nvidia-ampere-a10', 'node-lifecycle': 'normal'}, 'priorityClassName': 'model-engine-high-priority', 'serviceAccount': 'ml-worker', 'terminationGracePeriodSeconds': 600, 'tolerations': [{'effect': 'NoSchedule', 'key': 'nvidia.com/gpu', 'operator': 'Exists'}], 'volumes': [{'configMap': {'name': 'ml-worker-config'}, 'name': 'config-volume'}, {'configMap': {'name': 'launch-endpoint-id-end-cqi4v12d6mt002nap72g'}, 'name': 'user-config'}, {'configMap': {'name': 'launch-endpoint-id-end-cqi4v12d6mt002nap72g-endpoint-config'}, 'name': 'endpoint-config'}, {'emptyDir': {'medium': 'Memory'}, 'name': 'dshm'}]}}}, 'replicas': 0, 'rolloutStrategy': {'rollingUpdateConfiguration': {'maxSurge': 0, 'maxUnavailable': 1}, 'type': 'RollingUpdate'}, 'startupPolicy': 'LeaderCreated'}, 'status': {'conditions': [{'lastTransitionTime': '2024-07-27T01:38:14Z', 'message': 'All replicas are ready', 'reason': 'AllGroupsReady', 'status': 'True', 'type': 'Available'}], 'hpaPodSelector': 'leaderworkerset.sigs.k8s.io/name=launch-endpoint-id-end-cqi4v12d6mt002nap72g,leaderworkerset.sigs.k8s.io/worker-index=0'}}


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


@pytest.mark.parametrize("resource_arguments_type", ResourceArguments.__args__)
def test_resource_arguments_type_and_add_datadog_env_to_main_container(resource_arguments_type):
    # Convert the name of the type to a kebab case string
    # e.g. "BatchJobOrchestrationJobArguments" -> "batch-job-orchestration-job-arguments"
    resource_arguments_type_name = resource_arguments_type.__name__
    resource_arguments_type_name = "".join(
        "-" + c.lower() if c.isupper() else c for c in resource_arguments_type_name
    ).lstrip("-")
    resource_arguments_type_name = resource_arguments_type_name.replace("-arguments", "")

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
async def test_create_async_endpoint_has_correct_labels(
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
        await k8s_endpoint_resource_delegate.create_or_update_resources(
            request, sqs_queue_name="my_queue", sqs_queue_url="https://my_queue"
        )

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
async def test_create_streaming_endpoint_has_correct_labels(
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
    await k8s_endpoint_resource_delegate.create_or_update_resources(
        request,
        sqs_queue_name="my_queue",
        sqs_queue_url="https://my_queue",
    )

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
            expected_plurals=["verticalpodautoscalers", "virtualservices", "destinationrules"],
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
async def test_create_sync_endpoint_has_correct_labels(
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
        await k8s_endpoint_resource_delegate.create_or_update_resources(
            request,
            sqs_queue_name="my_queue",
            sqs_queue_url="https://my_queue,",
        )

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
                expected_plurals=["verticalpodautoscalers", "virtualservices", "destinationrules"],
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
async def test_create_multinode_endpoint_creates_lws(
    k8s_endpoint_resource_delegate,
    mock_apps_client,
    mock_core_client,
    mock_autoscaling_client,
    mock_policy_client,
    mock_custom_objects_client,
    mock_get_kubernetes_cluster_version,
    create_resources_request_streaming_runnable_image: CreateOrUpdateResourcesRequest,
    model_bundle_5: ModelBundle
):  
    # Patch model bundle so that it supports multinode
    model_bundle_5.metadata[WORKER_ENV_METADATA_KEY] = {"fake_env": "fake_value"}
    model_bundle_5.metadata[WORKER_COMMAND_METADATA_KEY] = ["fake_command"]
    create_resources_request_streaming_runnable_image.build_endpoint_request.model_endpoint_record.current_model_bundle = model_bundle_5
    create_resources_request_streaming_runnable_image.build_endpoint_request.model_endpoint_record.endpoint_type = ModelEndpointType.STREAMING
    
    create_resources_request_streaming_runnable_image.build_endpoint_request.nodes_per_worker = 2
    await k8s_endpoint_resource_delegate.create_or_update_resources(
        create_resources_request_streaming_runnable_image,
        sqs_queue_name="my_queue",
        sqs_queue_url="https://my_queue",
    )
    # Verify call to custom objects client with LWS is made
    create_custom_objects_call_args = mock_custom_objects_client.create_namespaced_custom_object.call_args
    assert create_custom_objects_call_args.kwargs["group"] == "leaderworkerset.x-k8s.io"


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
    # TODO do I need to mock out custom_objects_client.get_namespaced_custom_object to return ApiException
    # if it's asking for a LWS?
    # Pretend that LWS get gives an ApiException
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
        Mock(return_value=dict(min_workers=1, max_workers=3, per_worker=2)),
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
    # TODO do I need to mock out custom_objects_client.get_namespaced_custom_object to return ApiException
    # if it's asking for a LWS?
    # Pretend that LWS get gives an ApiException. TODO may need to mock out the keda call
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
        Mock(return_value=dict(min_workers=1, max_workers=3, per_worker=2)),
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
        "_get_common_endpoint_params_for_lws_type",
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
    # k8s_endpoint_resource_delegate.__setattr__(
    #     "_get_sync_autoscaling_params",
    #     Mock(return_value=dict(min_workers=1, max_workers=3, per_worker=2)),
    # )
    k8s_endpoint_resource_delegate.__setattr__(
        "_get_main_leader_container_from_lws", Mock(return_value=FakeK8sDeploymentContainer(env=[]))
    )
    k8s_endpoint_resource_delegate.__setattr__(
        "_get_launch_container_from_lws", Mock(return_value=FakeK8sDeploymentContainer(env=[]))
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

    # This is kinda brittle TODO
    mock_custom_objects_client.get_namespaced_custom_object = AsyncMock(return_value={
        "spec": {
            "replicas": 1,
            "leaderWorkerTemplate": {
                "leaderTemplate": {
                    "spec": {
                        "priorityClassName": "model-engine-high-priority",
                    }
                },
                "size": 2,
            }
        }
    })

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
        endpoint_id="", deployment_name="", endpoint_type=None  # type: ignore
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
async def test_delete_resources_multinode_success(
    k8s_endpoint_resource_delegate,
    mock_apps_client,
    mock_core_client,
    mock_autoscaling_client,
    mock_policy_client,
    mock_custom_objects_client,
):
    deleted = await k8s_endpoint_resource_delegate.delete_resources(
        endpoint_id="", deployment_name="", endpoint_type=ModelEndpointType.STREAMING
    )
    assert deleted


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
