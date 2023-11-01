import os
from string import Template
from typing import Any, Dict, List, Optional, Tuple

import kubernetes_asyncio
import yaml
from kubernetes import client as kube_client_sync
from kubernetes import config as kube_config_sync
from kubernetes_asyncio import config as kube_config_async
from kubernetes_asyncio.client.models.v1_container import V1Container
from kubernetes_asyncio.client.models.v1_deployment import V1Deployment
from kubernetes_asyncio.client.models.v1_env_var import V1EnvVar
from kubernetes_asyncio.client.models.v2beta2_horizontal_pod_autoscaler import (
    V2beta2HorizontalPodAutoscaler,
)
from kubernetes_asyncio.client.rest import ApiException
from kubernetes_asyncio.config import ConfigException
from model_engine_server.common.config import hmi_config
from model_engine_server.common.dtos.resource_manager import CreateOrUpdateResourcesRequest
from model_engine_server.common.env_vars import (
    CIRCLECI,
    LAUNCH_SERVICE_TEMPLATE_CONFIG_MAP_PATH,
    LAUNCH_SERVICE_TEMPLATE_FOLDER,
)
from model_engine_server.common.serialization_utils import b64_to_python_json, str_to_bool
from model_engine_server.core.config import infra_config
from model_engine_server.core.loggers import logger_name, make_logger
from model_engine_server.domain.entities import (
    ModelEndpointConfig,
    ModelEndpointDeploymentState,
    ModelEndpointInfraState,
    ModelEndpointRecord,
    ModelEndpointResourceState,
    ModelEndpointType,
    ModelEndpointUserConfigState,
    RunnableImageLike,
    TritonEnhancedRunnableImageFlavor,
)
from model_engine_server.domain.exceptions import EndpointResourceInfraException
from model_engine_server.domain.use_cases.model_endpoint_use_cases import MODEL_BUNDLE_CHANGED_KEY
from model_engine_server.infra.gateways.k8s_resource_parser import (
    get_per_worker_value_from_target_concurrency,
)
from model_engine_server.infra.gateways.resources.k8s_resource_types import (
    LAUNCH_HIGH_PRIORITY_CLASS,
    CommonEndpointParams,
    HorizontalAutoscalingEndpointParams,
    ResourceArguments,
    VerticalAutoscalingEndpointParams,
    get_endpoint_resource_arguments_from_request,
)
from packaging import version
from pydantic.utils import deep_update

logger = make_logger(logger_name())

HTTP_PORT = 5000

# Matches the user endpoint docker images, specifically where code gets copied
# and where the user actually owns the files
BASE_PATH_IN_ENDPOINT = "/app"

DATADOG_ENV_VAR = {"DD_TRACE_ENABLED", "DD_SERVICE", "DD_ENV", "DD_VERSION", "DD_AGENT_HOST"}

_lazy_load_kubernetes_clients = True
_kubernetes_apps_api = None
_kubernetes_core_api = None
_kubernetes_autoscaling_api = None
_kubernetes_batch_api = None
_kubernetes_policy_api = None
_kubernetes_custom_objects_api = None
_kubernetes_cluster_version = None


# --- K8s client caching functions
# IMPORTANT: set lazy load to False if running the k8s resource manager gateway in synchronous code
# using asyncio.run (or equivalents). This is because the API clients initialize
# aiohttp.ClientSession objects in their constructor, and wrapping the async code in asyncio.run et
# al. results in these ClientSessions being shared across event loops, which causes the error
# `RuntimeError: Event loop is closed`.
def set_lazy_load_kubernetes_clients(
    should_lazy_load: bool,
) -> bool:  # pragma: no cover
    global _lazy_load_kubernetes_clients
    former = _lazy_load_kubernetes_clients
    _lazy_load_kubernetes_clients = should_lazy_load
    return former


def get_kubernetes_cluster_version():  # pragma: no cover
    if _lazy_load_kubernetes_clients:
        global _kubernetes_cluster_version
    else:
        _kubernetes_cluster_version = None
    if not _kubernetes_cluster_version:
        version_info = kube_client_sync.VersionApi().get_code()
        # kubernetes will use `+` instead of specifying a patch version. This confuses version comparisons so we remove it.
        minor_version = version_info.minor.replace("+", "")
        major_version = version_info.major
        _kubernetes_cluster_version = f"{major_version}.{minor_version}"
    return _kubernetes_cluster_version


def get_kubernetes_apps_client():  # pragma: no cover
    if _lazy_load_kubernetes_clients:
        global _kubernetes_apps_api
    else:
        _kubernetes_apps_api = None
    if not _kubernetes_apps_api:
        _kubernetes_apps_api = kubernetes_asyncio.client.AppsV1Api()
    return _kubernetes_apps_api


def get_kubernetes_core_client():  # pragma: no cover
    if _lazy_load_kubernetes_clients:
        global _kubernetes_core_api
    else:
        _kubernetes_core_api = None
    if not _kubernetes_core_api:
        _kubernetes_core_api = kubernetes_asyncio.client.CoreV1Api()
    return _kubernetes_core_api


def get_kubernetes_autoscaling_client():  # pragma: no cover
    if _lazy_load_kubernetes_clients:
        global _kubernetes_autoscaling_api
    else:
        _kubernetes_autoscaling_api = None
    if not _kubernetes_autoscaling_api:
        cluster_version = get_kubernetes_cluster_version()
        # For k8s cluster versions 1.23 - 1.25 we need to use the v2beta2 api
        # For 1.26+ v2beta2 has been deperecated and merged into v2
        if version.parse(cluster_version) >= version.parse("1.26"):
            _kubernetes_autoscaling_api = kubernetes_asyncio.client.AutoscalingV2Api()
        else:
            _kubernetes_autoscaling_api = kubernetes_asyncio.client.AutoscalingV2beta2Api()
    return _kubernetes_autoscaling_api


def get_kubernetes_batch_client():  # pragma: no cover
    if _lazy_load_kubernetes_clients:
        global _kubernetes_batch_api
    else:
        _kubernetes_batch_api = None
    if not _kubernetes_batch_api:
        _kubernetes_batch_api = kubernetes_asyncio.client.BatchV1Api()
    return _kubernetes_batch_api


def get_kubernetes_policy_client():  # pragma: no cover
    if _lazy_load_kubernetes_clients:
        global _kubernetes_policy_api
    else:
        _kubernetes_policy_api = None
    if not _kubernetes_policy_api:
        _kubernetes_policy_api = kubernetes_asyncio.client.PolicyV1Api()
    return _kubernetes_policy_api


def get_kubernetes_custom_objects_client():  # pragma: no cover
    if _lazy_load_kubernetes_clients:
        global _kubernetes_custom_objects_api
    else:
        _kubernetes_custom_objects_api = None
    if not _kubernetes_custom_objects_api:
        _kubernetes_custom_objects_api = kubernetes_asyncio.client.CustomObjectsApi()
    return _kubernetes_custom_objects_api


def _endpoint_id_to_k8s_resource_group_name(endpoint_id: str) -> str:
    return f"launch-endpoint-id-{endpoint_id}".replace("_", "-")


def _k8s_resource_group_name_to_endpoint_id(k8s_resource_group_name: str) -> str:
    return k8s_resource_group_name.replace("launch-endpoint-id-", "").replace("-", "_")


_kube_config_loaded = False


async def maybe_load_kube_config():
    global _kube_config_loaded
    if _kube_config_loaded:
        return
    try:
        kube_config_async.load_incluster_config()
        kube_config_sync.load_incluster_config()
    except ConfigException:
        try:
            await kube_config_async.load_kube_config()
            kube_config_sync.load_kube_config()
        except:  # noqa: E722
            logger.warning("Could not load kube config.")
    _kube_config_loaded = True


def load_k8s_yaml(key: str, substitution_kwargs: ResourceArguments) -> Dict[str, Any]:
    if LAUNCH_SERVICE_TEMPLATE_FOLDER is not None:
        with open(os.path.join(LAUNCH_SERVICE_TEMPLATE_FOLDER, key), "r") as f:
            template_str = f.read()
    else:
        with open(LAUNCH_SERVICE_TEMPLATE_CONFIG_MAP_PATH, "r") as f:
            config_map_str = yaml.safe_load(f.read())
        template_str = config_map_str["data"][key]

    yaml_str = Template(template_str).substitute(**substitution_kwargs)
    try:
        yaml_obj = yaml.safe_load(yaml_str)
    except:
        logger.exception("Could not load yaml string: %s", yaml_str)
        raise
    return yaml_obj


def get_main_container_from_deployment_template(deployment_template: Dict[str, Any]):
    containers = deployment_template["spec"]["template"]["spec"]["containers"]
    for container in containers:
        if container["name"] == "main":
            user_container = container
            break
    else:
        raise ValueError(
            "main container (container['name'] == 'main') not found in deployment template when adding datadog env to main container."
        )
    return user_container


def add_datadog_env_to_main_container(deployment_template: Dict[str, Any]) -> None:
    user_container = get_main_container_from_deployment_template(deployment_template)

    user_container_envs = []
    for env in user_container["env"]:
        if env["name"] not in DATADOG_ENV_VAR:
            user_container_envs.append(env)

    user_container_envs.extend(
        [
            {
                "name": "DD_TRACE_ENABLED",
                "value": "false" if CIRCLECI else "true",
            },
            {
                "name": "DD_SERVICE",
                "value": deployment_template["metadata"]["labels"]["tags.datadoghq.com/service"],
            },
            {
                "name": "DD_ENV",
                "value": deployment_template["metadata"]["labels"]["tags.datadoghq.com/env"],
            },
            {
                "name": "DD_VERSION",
                "value": deployment_template["metadata"]["labels"]["tags.datadoghq.com/version"],
            },
            {
                "name": "DD_AGENT_HOST",
                "valueFrom": {"fieldRef": {"fieldPath": "status.hostIP"}},
            },
        ]
    )

    user_container["env"] = user_container_envs


class K8SEndpointResourceDelegate:
    async def create_or_update_resources(
        self,
        request: CreateOrUpdateResourcesRequest,
        sqs_queue_name: Optional[str] = None,
        sqs_queue_url: Optional[str] = None,
    ) -> None:
        await maybe_load_kube_config()
        try:
            await self._create_or_update_resources(
                request=request,
                sqs_queue_name=sqs_queue_name,
                sqs_queue_url=sqs_queue_url,
            )
        except ApiException as e:
            logger.exception("create_or_update_resources failed")
            raise EndpointResourceInfraException from e

    async def get_resources(
        self, endpoint_id: str, deployment_name: str, endpoint_type: ModelEndpointType
    ) -> ModelEndpointInfraState:
        await maybe_load_kube_config()
        try:
            return await self._get_resources(
                endpoint_id=endpoint_id,
                deployment_name=deployment_name,
                endpoint_type=endpoint_type,
            )
        except ApiException as e:
            logger.exception("get_resources failed")
            raise EndpointResourceInfraException from e

    async def get_all_resources(
        self,
    ) -> Dict[str, Tuple[bool, ModelEndpointInfraState]]:
        await maybe_load_kube_config()
        try:
            return await self._get_all_resources()
        except ApiException as e:
            logger.exception("get_all_resources failed")
            raise EndpointResourceInfraException from e

    async def delete_resources(
        self, endpoint_id: str, deployment_name: str, endpoint_type: ModelEndpointType
    ) -> bool:
        await maybe_load_kube_config()
        if endpoint_type in {ModelEndpointType.SYNC, ModelEndpointType.STREAMING}:
            return await self._delete_resources_sync(
                endpoint_id=endpoint_id, deployment_name=deployment_name
            )
        elif endpoint_type == ModelEndpointType.ASYNC:
            return await self._delete_resources_async(
                endpoint_id=endpoint_id, deployment_name=deployment_name
            )
        return False

    # --- Private helper functions
    @staticmethod
    def _get_env_value_from_envlist(
        envlist: Optional[List[V1EnvVar]], name: str
    ):  # pragma: no cover
        if envlist is None:
            return None
        for envvar in envlist:
            if envvar.name == name:
                return envvar.value
        return None

    def _get_common_endpoint_params(self, deployment_config: V1Deployment) -> CommonEndpointParams:
        """
        Reads some values from k8s common to both sync and async endpoints
        Args:
            deployment_config: The deployment config from k8s's python api (i.e. a V1Deployment)

        Returns:
            Dictionary with detected values
        """
        main_container = self._get_main_container(deployment_config)
        launch_container = self._get_launch_container(deployment_config)
        resources = main_container.resources
        image = main_container.image

        cpus = resources.requests["cpu"]
        memory = resources.requests["memory"]
        gpus = int((resources.limits or dict()).get("nvidia.com/gpu", 0))
        storage = resources.requests.get("ephemeral-storage")

        envlist = launch_container.env
        # Hack: for LIRA since the bundle_url isn't really a real env var
        # we use the `image` for now. This may change if we allow for unpickling
        # in LIRA.
        bundle_url = self._get_env_value_from_envlist(envlist, "BUNDLE_URL") or image
        aws_role = self._get_env_value_from_envlist(envlist, "AWS_PROFILE")
        results_s3_bucket = self._get_env_value_from_envlist(envlist, "RESULTS_S3_BUCKET")

        # Temporary fix: new LIRA endpoints created should have these env vars
        # but old ones don't, so we can fetch them from the config.
        if aws_role is None:
            aws_role = infra_config().profile_ml_inference_worker
        if results_s3_bucket is None:
            results_s3_bucket = infra_config().s3_bucket

        if bundle_url is None or aws_role is None or results_s3_bucket is None:
            raise ValueError("Failed to fetch common endpoint values.")

        try:
            node_selector = deployment_config.spec.template.spec.node_selector
            gpu_type = node_selector.get("k8s.amazonaws.com/accelerator", None)
        except AttributeError:
            gpu_type = None

        try:
            labels = deployment_config.metadata.labels
        except AttributeError:
            labels = None

        common_build_endpoint_request: CommonEndpointParams = dict(
            cpus=cpus,
            memory=memory,
            gpus=gpus,
            gpu_type=gpu_type,
            storage=storage,
            bundle_url=bundle_url,
            aws_role=aws_role,
            results_s3_bucket=results_s3_bucket,
            image=image,
            labels=labels,
        )
        return common_build_endpoint_request

    @staticmethod
    def _get_main_container(deployment_config: V1Deployment) -> V1Container:
        pod_containers = deployment_config.spec.template.spec.containers
        name_to_container = {container.name: container for container in pod_containers}
        if "main" not in name_to_container:
            raise ValueError("No main container detected")
        return name_to_container["main"]

    @staticmethod
    def _get_launch_container(deployment_config: V1Deployment) -> V1Container:
        pod_containers = deployment_config.spec.template.spec.containers
        name_to_container = {container.name: container for container in pod_containers}

        # If a celery forwarder is present, use that
        if "celery-forwarder" in name_to_container:
            return name_to_container["celery-forwarder"]

        # If a http forwarder is present, use that
        if "http-forwarder" in name_to_container:
            return name_to_container["http-forwarder"]

        # Fall back to the main container
        if "main" not in name_to_container:
            raise ValueError("No main container detected")
        return name_to_container["main"]

    # --- Private low level fns that interact with k8s

    @staticmethod
    async def _create_deployment(
        model_endpoint_record: ModelEndpointRecord, deployment: Dict[str, Any], name: str
    ) -> None:
        """
        Lower-level function to create/patch a k8s deployment
        Args:
            deployment: Deployment body (a nested Dict in the format specified by Kubernetes)
            name: The name of the deployment on K8s

        Returns:
            Nothing; raises a k8s ApiException if failure

        """
        apps_client = get_kubernetes_apps_client()
        try:
            # Istio injection is handled by the `sidecar.istio.io/inject: "true"` annotation
            # and deploying in a namespace with a label "istio-injection: enabled"
            await apps_client.create_namespaced_deployment(
                namespace=hmi_config.endpoint_namespace,
                body=deployment,
            )
        except ApiException as exc:
            if exc.status == 409:
                if (
                    model_endpoint_record.metadata is not None
                    and MODEL_BUNDLE_CHANGED_KEY in model_endpoint_record.metadata
                ):
                    logger.info(
                        f"Deployment {name} already exists, replacing since model bundle has changed"
                    )
                    logger.info(f"Deployment {name} contents: {deployment}")
                    await apps_client.replace_namespaced_deployment(
                        name=name,
                        namespace=hmi_config.endpoint_namespace,
                        body=deployment,
                    )
                else:
                    logger.info(f"Deployment {name} already exists, patching")

                    if "replicas" in deployment["spec"]:
                        # Don't pass in replicas if we're doing an update, because we want to just
                        # let the autoscaler do its thing.
                        del deployment["spec"]["replicas"]
                    logger.info(f"Deployment {name} contents: {deployment}")

                    try:
                        await apps_client.patch_namespaced_deployment(
                            name=name,
                            namespace=hmi_config.endpoint_namespace,
                            body=deployment,
                        )
                    except ApiException as exc2:
                        if exc2.status in [409, 422]:
                            logger.info(
                                f"Deployment {name} failed to patch, falling back to replacing"
                            )
                            await apps_client.replace_namespaced_deployment(
                                name=name,
                                namespace=hmi_config.endpoint_namespace,
                                body=deployment,
                            )
                        else:
                            logger.exception(
                                "Got an exception when trying to replace the Deployment"
                            )
                            raise
            else:
                logger.exception("Got an exception when trying to apply the Deployment")
                raise

    @staticmethod
    async def _create_config_map(config_map: Dict[str, Any], name: str) -> None:
        """
        Creates a k8s ConfigMap from a config_map body
        Args:
            config_map: ConfigMap body (nested Dict in K8s-specified format)
            name: Name of config_map on K8s

        Returns:
            Nothing; raises a k8s ApiException if failure
        """
        core_api = get_kubernetes_core_client()
        try:
            await core_api.create_namespaced_config_map(
                hmi_config.endpoint_namespace, body=config_map
            )
        except ApiException as exc:
            if exc.status == 409:
                logger.info(f"ConfigMap {name} already exists, replacing")
                await core_api.patch_namespaced_config_map(
                    name=name,
                    namespace=hmi_config.endpoint_namespace,
                    body=config_map,
                )
            else:
                logger.exception("Got an exception when trying to apply the ConfigMap")
                raise

    @staticmethod
    async def _create_hpa(hpa: Dict[str, Any], name: str) -> None:
        """
        Lower-level function to create/patch a k8s HorizontalPodAutoscaler (hpa)
        Args:
            hpa: HPA body (a nested Dict in the format specified by Kubernetes)
            name: The name of the hpa on K8s

        Returns:
            Nothing; raises a k8s ApiException if failure

        """
        autoscaling_api = get_kubernetes_autoscaling_client()
        try:
            await autoscaling_api.create_namespaced_horizontal_pod_autoscaler(
                namespace=hmi_config.endpoint_namespace, body=hpa
            )
        except ApiException as exc:
            if exc.status == 409:
                logger.info(f"HorizontalPodAutoscaler {name} already exists, replacing")
                try:
                    await autoscaling_api.patch_namespaced_horizontal_pod_autoscaler(
                        name=name, namespace=hmi_config.endpoint_namespace, body=hpa
                    )
                except ValueError as exc2:
                    # The k8s api has a bug where a ValueError is thrown. This catches and drops it.
                    if str(exc2) == "Invalid value for `conditions`, must not be `None`":
                        # Workaround from https://github.com/kubernetes-client/python/issues/1098#issuecomment-663031331
                        logger.info("Skipping invalid 'conditions' value...")
                    else:
                        raise exc2
            else:
                logger.exception(
                    "Got an exception when trying to apply the HorizontalPodAutoscaler"
                )
                raise
        except ValueError as exc2:
            if str(exc2) == "Invalid value for `conditions`, must not be `None`":
                # Workaround from https://github.com/kubernetes-client/python/issues/1098#issuecomment-663031331
                logger.info("Skipping invalid 'conditions' value...")
            else:
                raise exc2

    @staticmethod
    async def _create_vpa(vpa: Dict[str, Any], name: str) -> None:
        """
        Lower-level function to create/patch a k8s VerticalPodAutoscaler (vpa)
        Args:
            vpa: VPA body (a nested Dict in the format specified by Kubernetes)
            name: The name of the vpa on K8s

        Returns:
            Nothing; raises a k8s ApiException if failure

        """
        custom_objects_api = get_kubernetes_custom_objects_client()
        try:
            await custom_objects_api.create_namespaced_custom_object(
                group="autoscaling.k8s.io",
                version="v1",
                namespace=hmi_config.endpoint_namespace,
                plural="verticalpodautoscalers",
                body=vpa,
            )
        except ApiException as exc:
            if exc.status == 409:
                logger.info(f"VerticalPodAutoscaler {name} already exists, replacing")

                # The async k8s client has a bug with patching custom objects, so we manually
                # merge the new VPA with the old one and then replace the old one with the merged
                # one.
                existing_vpa = await custom_objects_api.get_namespaced_custom_object(
                    group="autoscaling.k8s.io",
                    version="v1",
                    namespace=hmi_config.endpoint_namespace,
                    plural="verticalpodautoscalers",
                    name=name,
                )
                new_vpa = deep_update(existing_vpa, vpa)
                await custom_objects_api.replace_namespaced_custom_object(
                    group="autoscaling.k8s.io",
                    version="v1",
                    namespace=hmi_config.endpoint_namespace,
                    plural="verticalpodautoscalers",
                    name=name,
                    body=new_vpa,
                )
            else:
                logger.exception("Got an exception when trying to apply the VerticalPodAutoscaler")
                raise

    @staticmethod
    async def _create_pdb(pdb: Dict[str, Any], name: str) -> None:
        """
        Lower-level function to create/patch a k8s PodDisruptionBudget (pdb)
        Args:
            pdb: PDB body (a nested Dict in the format specified by Kubernetes)
            name: The name of the pdb on K8s

        Returns:
            Nothing; raises a k8s ApiException if failure

        """
        policy_api = get_kubernetes_policy_client()
        try:
            await policy_api.create_namespaced_pod_disruption_budget(
                namespace=hmi_config.endpoint_namespace,
                body=pdb,
            )
        except ApiException as exc:
            if exc.status == 409:
                logger.info(f"PodDisruptionBudget {name} already exists, replacing")

                await policy_api.patch_namespaced_pod_disruption_budget(
                    name=name,
                    namespace=hmi_config.endpoint_namespace,
                    body=pdb,
                )
            else:
                logger.exception("Got an exception when trying to apply the PodDisruptionBudget")
                raise

    @staticmethod
    async def _create_keda_scaled_object(scaled_object: Dict[str, Any], name: str) -> None:
        custom_objects_api = get_kubernetes_custom_objects_client()
        try:
            await custom_objects_api.create_namespaced_custom_object(
                group="keda.sh",
                version="v1alpha1",
                namespace=hmi_config.endpoint_namespace,
                plural="scaledobjects",
                body=scaled_object,
            )
        except ApiException as exc:
            if exc.status == 409:
                logger.info(f"ScaledObject {name} already exists, replacing")

                # The async k8s client has a bug with patching custom objects, so we manually
                # merge the new ScaledObject with the old one and then replace the old one with the merged
                # one. See _create_vpa for more details.
                # There is a setting `restoreToOriginalReplicaCount` in the keda ScaledObject that should be set to
                # false which should make it safe to do this replace (as opposed to a patch)
                existing_scaled_object = await custom_objects_api.get_namespaced_custom_object(
                    group="keda.sh",
                    version="v1alpha1",
                    namespace=hmi_config.endpoint_namespace,
                    plural="scaledobjects",
                    name=name,
                )
                new_scaled_object = deep_update(existing_scaled_object, scaled_object)
                await custom_objects_api.replace_namespaced_custom_object(
                    group="keda.sh",
                    version="v1alpha1",
                    namespace=hmi_config.endpoint_namespace,
                    plural="scaledobjects",
                    name=name,
                    body=new_scaled_object,
                )
            else:
                logger.exception("Got an exception when trying to apply the ScaledObject")
                raise

    @staticmethod
    async def _create_destination_rule(destination_rule: Dict[str, Any], name: str) -> None:
        """
        Lower-level function to create/patch an Istio DestinationRule. This is only created for sync endpoints.
        Args:
            destination_rule: DestinationRule body (a nested Dict in the format specified by Kubernetes)
            name: The name of the DestinationRule resource.

        Returns:
            Nothing; raises a k8s ApiException if failure
        pass
        """
        custom_objects_api = get_kubernetes_custom_objects_client()
        try:
            await custom_objects_api.create_namespaced_custom_object(
                group="networking.istio.io",
                version="v1beta1",
                namespace=hmi_config.endpoint_namespace,
                plural="destinationrules",
                body=destination_rule,
            )
        except ApiException as exc:
            if exc.status == 409:
                logger.info(f"DestinationRule {name} already exists, replacing")
                # The async k8s client has a bug with patching custom objects, so we manually
                # merge the new DestinationRule with the old one and then replace the old one with the merged
                # one.
                existing_destination_rule = await custom_objects_api.get_namespaced_custom_object(
                    group="networking.istio.io",
                    version="v1beta1",
                    namespace=hmi_config.endpoint_namespace,
                    plural="destinationrules",
                    name=name,
                )
                new_destination_rule = deep_update(existing_destination_rule, destination_rule)
                await custom_objects_api.replace_namespaced_custom_object(
                    group="networking.istio.io",
                    version="v1beta1",
                    namespace=hmi_config.endpoint_namespace,
                    plural="destinationrules",
                    name=name,
                    body=new_destination_rule,
                )
            else:
                logger.exception("Got an exception when trying to apply the DestinationRule")
                raise

    @staticmethod
    async def _create_virtual_service(virtual_service: Dict[str, Any], name: str) -> None:
        """
        Lower-level function to create/patch an Istio VirtualService. This is only created for sync endpoints.
        Args:
            virtual_service: VirtualService body (a nested Dict in the format specified by Kubernetes)
            name: The name of the VirtualService resource.

        Returns:
            Nothing; raises a k8s ApiException if failure
        pass
        """
        custom_objects_api = get_kubernetes_custom_objects_client()
        try:
            await custom_objects_api.create_namespaced_custom_object(
                group="networking.istio.io",
                version="v1alpha3",
                namespace=hmi_config.endpoint_namespace,
                plural="virtualservices",
                body=virtual_service,
            )
        except ApiException as exc:
            if exc.status == 409:
                logger.info(f"VirtualService {name} already exists, replacing")
                # The async k8s client has a bug with patching custom objects, so we manually
                # merge the new VirtualService with the old one and then replace the old one with the merged
                # one.
                existing_virtual_service = await custom_objects_api.get_namespaced_custom_object(
                    group="networking.istio.io",
                    version="v1alpha3",
                    namespace=hmi_config.endpoint_namespace,
                    plural="virtualservices",
                    name=name,
                )
                new_virtual_service = deep_update(existing_virtual_service, virtual_service)
                await custom_objects_api.replace_namespaced_custom_object(
                    group="networking.istio.io",
                    version="v1alpha3",
                    namespace=hmi_config.endpoint_namespace,
                    plural="virtualservices",
                    name=name,
                    body=new_virtual_service,
                )
            else:
                logger.exception("Got an exception when trying to apply the VirtualService")
                raise

    @staticmethod
    async def _create_service(service, name: str) -> None:
        """
        Lower-level function to create/patch a k8s Service
        Args:
            service: Service body (a nested Dict in the format specified by Kubernetes)
            name: The name of the service on K8s

        Returns:
            Nothing; raises a k8s ApiException if failure

        """
        core_api = get_kubernetes_core_client()
        try:
            await core_api.create_namespaced_service(
                namespace=hmi_config.endpoint_namespace, body=service
            )
        except ApiException as exc:
            if exc.status in [409, 422]:
                logger.info(f"Service {name} already exists, replacing")
                await core_api.patch_namespaced_service(
                    name=name,
                    namespace=hmi_config.endpoint_namespace,
                    body=service,
                )
            else:
                logger.exception("Got an exception when trying to apply the Service")
                raise

    @staticmethod
    async def _get_config_maps(
        endpoint_id: str,
        deployment_name: str,
    ) -> List[kubernetes_asyncio.client.models.v1_config_map.V1ConfigMap]:
        """
        Gets ConfigMaps associated with a given user id + endpoint name
        This should be considered the same abstraction level as get_deployment

        """
        k8s_core_api = get_kubernetes_core_client()
        endpoint_id_label_selector = f"endpoint_id={endpoint_id}"
        config_maps = await k8s_core_api.list_namespaced_config_map(
            namespace=hmi_config.endpoint_namespace,
            label_selector=endpoint_id_label_selector,
        )
        if config_maps.items:
            return config_maps.items
        else:
            logger.warning(
                f"Could not find any ConfigMaps with label selector {endpoint_id_label_selector}, "
                "falling back to deployment_name"
            )
            deployment_name_label_selector = f"deployment_name={deployment_name}"
            config_maps = await k8s_core_api.list_namespaced_config_map(
                namespace=hmi_config.endpoint_namespace,
                label_selector=deployment_name_label_selector,
            )
            return config_maps.items

    @staticmethod
    async def _get_all_config_maps() -> List[
        kubernetes_asyncio.client.models.v1_config_map.V1ConfigMap
    ]:
        k8s_core_api = get_kubernetes_core_client()
        config_maps = await k8s_core_api.list_namespaced_config_map(
            namespace=hmi_config.endpoint_namespace
        )
        return config_maps.items

    @classmethod
    def _read_endpoint_config_map_from_fetched(
        cls,
        config_map_name: str,
        fetched_config_map_map: Dict[
            str, kubernetes_asyncio.client.models.v1_config_map.V1ConfigMap
        ],
    ) -> Optional[Dict[str, Any]]:
        """
        Reads config map data. First looks at already fetched values, then at k8s.
        """
        config_map = fetched_config_map_map.get(config_map_name, None)
        if config_map is None:
            return None
        return b64_to_python_json(config_map.data["raw_data"])

    @classmethod
    def _translate_k8s_config_maps_to_user_config_data(
        cls,
        deployment_name: str,
        fetched_config_map_list: List[kubernetes_asyncio.client.models.v1_config_map.V1ConfigMap],
    ):
        config_map_map = {cm.metadata.name: cm for cm in fetched_config_map_list}
        app_config_data = cls._read_endpoint_config_map_from_fetched(
            deployment_name, config_map_map
        )
        endpoint_config_data = cls._read_endpoint_config_map_from_fetched(
            f"{deployment_name}-endpoint-config", config_map_map
        )
        endpoint_config = None
        if endpoint_config_data is not None:
            endpoint_config = ModelEndpointConfig.parse_obj(endpoint_config_data)
        return ModelEndpointUserConfigState(
            app_config=app_config_data,
            endpoint_config=endpoint_config,
        )

    @staticmethod
    async def _delete_deployment(endpoint_id: str, deployment_name: str) -> bool:
        apps_client = get_kubernetes_apps_client()
        k8s_resource_group_name = _endpoint_id_to_k8s_resource_group_name(endpoint_id)
        try:
            await apps_client.delete_namespaced_deployment(
                name=k8s_resource_group_name, namespace=hmi_config.endpoint_namespace
            )
        except ApiException as e:
            if e.status == 404:
                # Try the legacy deployment_name
                logger.warning(
                    f"Could not find resource, falling back to legacy deployment_name: "
                    f"k8s_resource_group_name={k8s_resource_group_name}, endpoint_id={endpoint_id}, "
                    f"deployment_name={deployment_name}"
                )
                try:
                    k8s_resource_group_name = deployment_name
                    await apps_client.delete_namespaced_deployment(
                        name=k8s_resource_group_name,
                        namespace=hmi_config.endpoint_namespace,
                    )
                except ApiException as e2:
                    if e2.status == 404:
                        # Deployment doesn't exist, might as well continue to delete the db entry
                        logger.warning(
                            f"Trying to delete nonexistent Deployment {k8s_resource_group_name}"
                        )
                    else:
                        logger.exception(f"Deletion of Deployment {k8s_resource_group_name} failed")
                        return False
            else:
                logger.exception(f"Deletion of Deployment {k8s_resource_group_name} failed")
                return False
        return True

    async def _delete_config_maps(self, endpoint_id: str, deployment_name: str) -> bool:
        core_client = get_kubernetes_core_client()
        config_map_names = [
            config_map.metadata.name
            for config_map in await self._get_config_maps(
                endpoint_id=endpoint_id, deployment_name=deployment_name
            )
        ]
        for config_map_name in config_map_names:
            try:
                await core_client.delete_namespaced_config_map(
                    config_map_name, hmi_config.endpoint_namespace
                )
            except ApiException as e:
                if e.status == 404:
                    logger.warning(f"Trying to delete nonexistent ConfigMap {config_map_name}")
                else:
                    logger.error(
                        f"Deletion of ConfigMap {config_map_name} failed with error" f" {e}"
                    )
                    return False
        return True

    @staticmethod
    async def _delete_service(endpoint_id: str, deployment_name: str) -> bool:
        core_client = get_kubernetes_core_client()
        k8s_resource_group_name = _endpoint_id_to_k8s_resource_group_name(endpoint_id)
        try:
            await core_client.delete_namespaced_service(
                name=k8s_resource_group_name, namespace=hmi_config.endpoint_namespace
            )
        except ApiException as e:
            if e.status == 404:
                logger.warning(
                    f"Could not find resource, falling back to legacy deployment_name: "
                    f"k8s_resource_group_name={k8s_resource_group_name}, endpoint_id={endpoint_id}, "
                    f"deployment_name={deployment_name}"
                )
                try:
                    k8s_resource_group_name = deployment_name
                    await core_client.delete_namespaced_service(
                        name=deployment_name, namespace=hmi_config.endpoint_namespace
                    )
                except ApiException as e2:
                    if e2.status == 404:
                        logger.warning(
                            f"Trying to delete nonexistent Service {k8s_resource_group_name}"
                        )
                    else:
                        logger.exception(f"Deletion of Service {k8s_resource_group_name} failed")
                        return False
            else:
                logger.exception(f"Deletion of Service {k8s_resource_group_name} failed")
                return False
        return True

    @staticmethod
    async def _delete_destination_rule(endpoint_id: str) -> bool:
        custom_objects_client = get_kubernetes_custom_objects_client()
        k8s_resource_group_name = _endpoint_id_to_k8s_resource_group_name(endpoint_id)
        try:
            await custom_objects_client.delete_namespaced_custom_object(
                group="networking.istio.io",
                version="v1beta1",
                namespace=hmi_config.endpoint_namespace,
                plural="destinationrules",
                name=k8s_resource_group_name,
            )
        except ApiException as e:
            if e.status == 404:
                logger.warning(
                    f"Trying to delete nonexistent DestinationRule {k8s_resource_group_name}"
                )
            else:
                logger.exception(f"Deletion of DestinationRule {k8s_resource_group_name} failed")
                return False
        return True

    @staticmethod
    async def _delete_virtual_service(endpoint_id: str) -> bool:
        custom_objects_client = get_kubernetes_custom_objects_client()
        k8s_resource_group_name = _endpoint_id_to_k8s_resource_group_name(endpoint_id)
        try:
            await custom_objects_client.delete_namespaced_custom_object(
                group="networking.istio.io",
                version="v1alpha3",
                namespace=hmi_config.endpoint_namespace,
                plural="virtualservices",
                name=k8s_resource_group_name,
            )
        except ApiException as e:
            if e.status == 404:
                logger.warning(
                    f"Trying to delete nonexistent VirtualService {k8s_resource_group_name}"
                )
            else:
                logger.exception(f"Deletion of VirtualService {k8s_resource_group_name} failed")
                return False
        return True

    @staticmethod
    async def _delete_vpa(endpoint_id: str) -> bool:
        custom_objects_client = get_kubernetes_custom_objects_client()
        k8s_resource_group_name = _endpoint_id_to_k8s_resource_group_name(endpoint_id)
        try:
            await custom_objects_client.delete_namespaced_custom_object(
                group="autoscaling.k8s.io",
                version="v1",
                namespace=hmi_config.endpoint_namespace,
                plural="verticalpodautoscalers",
                name=k8s_resource_group_name,
            )
        except ApiException as e:
            if e.status == 404:
                logger.warning(f"Trying to delete nonexistent VPA {k8s_resource_group_name}")
            else:
                logger.exception(
                    f"Deletion of VerticalPodAutoscaler {k8s_resource_group_name} failed"
                )
                return False
        return True

    @staticmethod
    async def _delete_hpa(endpoint_id: str, deployment_name: str) -> bool:
        autoscaling_client = get_kubernetes_autoscaling_client()
        k8s_resource_group_name = _endpoint_id_to_k8s_resource_group_name(endpoint_id)
        try:
            await autoscaling_client.delete_namespaced_horizontal_pod_autoscaler(
                name=k8s_resource_group_name, namespace=hmi_config.endpoint_namespace
            )
        except ApiException as e:
            if e.status == 404:
                logger.warning(
                    f"Could not find resource, falling back to legacy deployment_name: "
                    f"k8s_resource_group_name={k8s_resource_group_name}, endpoint_id={endpoint_id}, "
                    f"deployment_name={deployment_name}"
                )
                try:
                    k8s_resource_group_name = deployment_name
                    await autoscaling_client.delete_namespaced_horizontal_pod_autoscaler(
                        name=k8s_resource_group_name,
                        namespace=hmi_config.endpoint_namespace,
                    )
                except ApiException as e2:
                    if e2.status == 404:
                        logger.warning(
                            f"Trying to delete nonexistent HPA {k8s_resource_group_name}"
                        )
                    else:
                        logger.exception(
                            f"Deletion of HorizontalPodAutoscaler {k8s_resource_group_name} failed"
                        )
                        return False
            else:
                logger.exception(
                    f"Deletion of HorizontalPodAutoscaler {k8s_resource_group_name} failed"
                )
                return False
        return True

    @staticmethod
    async def _delete_pdb(endpoint_id: str) -> bool:
        policy_client = get_kubernetes_policy_client()
        k8s_resource_group_name = _endpoint_id_to_k8s_resource_group_name(endpoint_id)
        try:
            await policy_client.delete_namespaced_pod_disruption_budget(
                namespace=hmi_config.endpoint_namespace,
                name=k8s_resource_group_name,
            )
        except ApiException as e:
            if e.status == 404:
                logger.warning(
                    f"Trying to delete nonexistent PodDisruptionBudget {k8s_resource_group_name}"
                )
            else:
                logger.exception(
                    f"Deletion of PodDisruptionBudget {k8s_resource_group_name} failed"
                )
                return False
        return True

    @staticmethod
    async def _delete_keda_scaled_object(endpoint_id: str) -> bool:
        custom_objects_client = get_kubernetes_custom_objects_client()
        k8s_resource_group_name = _endpoint_id_to_k8s_resource_group_name(endpoint_id)
        try:
            await custom_objects_client.delete_namespaced_custom_object(
                group="keda.sh",
                version="v1alpha1",
                namespace=hmi_config.endpoint_namespace,
                plural="scaledobjects",
                name=k8s_resource_group_name,
            )
        except ApiException as e:
            if e.status == 404:
                logger.warning(
                    f"Trying to delete nonexistent ScaledObject {k8s_resource_group_name}"
                )
            else:
                logger.exception(f"Deletion of ScaledObject {k8s_resource_group_name} failed")
                return False
        return True

    # --- Private higher level fns that interact with k8s

    @staticmethod
    def _get_deployment_resource_name(request: CreateOrUpdateResourcesRequest) -> str:
        build_endpoint_request = request.build_endpoint_request
        model_endpoint_record = build_endpoint_request.model_endpoint_record
        flavor = model_endpoint_record.current_model_bundle.flavor

        if isinstance(flavor, TritonEnhancedRunnableImageFlavor):
            flavor_class = "triton-enhanced-runnable-image"
        else:
            flavor_class = "runnable-image"

        mode = model_endpoint_record.endpoint_type.value
        device = "gpu" if build_endpoint_request.gpus > 0 else "cpu"

        deployment_resource_name = f"deployment-{flavor_class}-{mode}-{device}"
        return deployment_resource_name

    async def _create_or_update_resources(
        self,
        request: CreateOrUpdateResourcesRequest,
        sqs_queue_name: Optional[str] = None,
        sqs_queue_url: Optional[str] = None,
    ) -> None:
        sqs_queue_name_str = sqs_queue_name or ""
        sqs_queue_url_str = sqs_queue_url or ""
        build_endpoint_request = request.build_endpoint_request
        model_endpoint_record = build_endpoint_request.model_endpoint_record
        k8s_resource_group_name = _endpoint_id_to_k8s_resource_group_name(
            build_endpoint_request.model_endpoint_record.id
        )

        deployment_resource_name = self._get_deployment_resource_name(request)
        deployment_arguments = get_endpoint_resource_arguments_from_request(
            k8s_resource_group_name=k8s_resource_group_name,
            request=request,
            sqs_queue_name=sqs_queue_name_str,
            sqs_queue_url=sqs_queue_url_str,
            endpoint_resource_name=deployment_resource_name,
        )
        deployment_template = load_k8s_yaml(
            f"{deployment_resource_name}.yaml", deployment_arguments
        )
        if isinstance(
            request.build_endpoint_request.model_endpoint_record.current_model_bundle.flavor,
            RunnableImageLike,
        ):
            add_datadog_env_to_main_container(deployment_template)
        await self._create_deployment(
            model_endpoint_record=request.build_endpoint_request.model_endpoint_record,
            deployment=deployment_template,
            name=k8s_resource_group_name,
        )

        user_config_arguments = get_endpoint_resource_arguments_from_request(
            k8s_resource_group_name=k8s_resource_group_name,
            request=request,
            sqs_queue_name=sqs_queue_name_str,
            sqs_queue_url=sqs_queue_url_str,
            endpoint_resource_name="user-config",
        )
        user_config_template = load_k8s_yaml("user-config.yaml", user_config_arguments)
        await self._create_config_map(
            config_map=user_config_template,
            name=k8s_resource_group_name,
        )

        endpoint_config_arguments = get_endpoint_resource_arguments_from_request(
            k8s_resource_group_name=k8s_resource_group_name,
            request=request,
            sqs_queue_name=sqs_queue_name_str,
            sqs_queue_url=sqs_queue_url_str,
            endpoint_resource_name="endpoint-config",
        )
        endpoint_config_template = load_k8s_yaml("endpoint-config.yaml", endpoint_config_arguments)
        await self._create_config_map(
            config_map=endpoint_config_template,
            name=f"{k8s_resource_group_name}-endpoint-config",
        )

        if request.build_endpoint_request.optimize_costs:
            vpa_arguments = get_endpoint_resource_arguments_from_request(
                k8s_resource_group_name=k8s_resource_group_name,
                request=request,
                sqs_queue_name=sqs_queue_name_str,
                sqs_queue_url=sqs_queue_url_str,
                endpoint_resource_name="vertical-pod-autoscaler",
            )
            vpa_template = load_k8s_yaml("vertical-pod-autoscaler.yaml", vpa_arguments)
            await self._create_vpa(
                vpa=vpa_template,
                name=k8s_resource_group_name,
            )

        pdb_config_arguments = get_endpoint_resource_arguments_from_request(
            k8s_resource_group_name=k8s_resource_group_name,
            request=request,
            sqs_queue_name=sqs_queue_name_str,
            sqs_queue_url=sqs_queue_url_str,
            endpoint_resource_name="pod-disruption-budget",
        )
        pdb_template = load_k8s_yaml("pod-disruption-budget.yaml", pdb_config_arguments)
        await self._create_pdb(
            pdb=pdb_template,
            name=k8s_resource_group_name,
        )

        if model_endpoint_record.endpoint_type in {
            ModelEndpointType.SYNC,
            ModelEndpointType.STREAMING,
        }:
            cluster_version = get_kubernetes_cluster_version()
            # For k8s cluster versions 1.23 - 1.25 we need to use the v2beta2 api
            # For 1.26+ v2beta2 has been deperecated and merged into v2
            if version.parse(cluster_version) >= version.parse("1.26"):
                api_version = "autoscaling/v2"
            else:
                api_version = "autoscaling/v2beta2"

            # create exactly one of HPA or KEDA ScaledObject, depending if we request more than 0 min_workers
            # Right now, keda only will support scaling from 0 to 1
            # TODO support keda scaling from 1 to N as well
            if request.build_endpoint_request.min_workers > 0:
                # Delete keda scaled object if it exists so exactly one of HPA or KEDA ScaledObject remains
                await self._delete_keda_scaled_object(
                    build_endpoint_request.model_endpoint_record.id
                )
                hpa_arguments = get_endpoint_resource_arguments_from_request(
                    k8s_resource_group_name=k8s_resource_group_name,
                    request=request,
                    sqs_queue_name=sqs_queue_name_str,
                    sqs_queue_url=sqs_queue_url_str,
                    endpoint_resource_name="horizontal-pod-autoscaler",
                    api_version=api_version,
                )
                hpa_template = load_k8s_yaml("horizontal-pod-autoscaler.yaml", hpa_arguments)
                await self._create_hpa(
                    hpa=hpa_template,
                    name=k8s_resource_group_name,
                )
            else:  # min workers == 0, use keda
                # Delete hpa if it exists so exactly one of HPA or KEDA ScaledObject remains
                await self._delete_hpa(
                    build_endpoint_request.model_endpoint_record.id, k8s_resource_group_name
                )
                keda_scaled_object_arguments = get_endpoint_resource_arguments_from_request(
                    k8s_resource_group_name=k8s_resource_group_name,
                    request=request,
                    sqs_queue_name=sqs_queue_name_str,
                    sqs_queue_url=sqs_queue_url_str,
                    endpoint_resource_name="keda-scaled-object",
                )
                keda_scaled_object_template = load_k8s_yaml(
                    "keda-scaled-object.yaml", keda_scaled_object_arguments
                )
                await self._create_keda_scaled_object(
                    scaled_object=keda_scaled_object_template,
                    name=k8s_resource_group_name,
                )

            service_arguments = get_endpoint_resource_arguments_from_request(
                k8s_resource_group_name=k8s_resource_group_name,
                request=request,
                sqs_queue_name=sqs_queue_name_str,
                sqs_queue_url=sqs_queue_url_str,
                endpoint_resource_name="service",
            )
            service_template = load_k8s_yaml("service.yaml", service_arguments)
            await self._create_service(
                service=service_template,
                name=k8s_resource_group_name,
            )

            # TODO wsong: add flag to use istio and use these arguments
            if hmi_config.istio_enabled:
                virtual_service_arguments = get_endpoint_resource_arguments_from_request(
                    k8s_resource_group_name=k8s_resource_group_name,
                    request=request,
                    sqs_queue_name=sqs_queue_name_str,
                    sqs_queue_url=sqs_queue_url_str,
                    endpoint_resource_name="virtual-service",
                )
                virtual_service_template = load_k8s_yaml(
                    "virtual-service.yaml", virtual_service_arguments
                )
                await self._create_virtual_service(
                    virtual_service=virtual_service_template,
                    name=k8s_resource_group_name,
                )

                destination_rule_arguments = get_endpoint_resource_arguments_from_request(
                    k8s_resource_group_name=k8s_resource_group_name,
                    request=request,
                    sqs_queue_name=sqs_queue_name_str,
                    sqs_queue_url=sqs_queue_url_str,
                    endpoint_resource_name="destination-rule",
                )
                destination_rule_template = load_k8s_yaml(
                    "destination-rule.yaml", destination_rule_arguments
                )
                await self._create_destination_rule(
                    destination_rule=destination_rule_template,
                    name=k8s_resource_group_name,
                )

    @staticmethod
    def _get_vertical_autoscaling_params(
        vpa_config,
    ) -> Optional[VerticalAutoscalingEndpointParams]:
        container_policies = vpa_config["spec"]["resourcePolicy"]["containerPolicies"]
        matching_container_policies = [
            policy for policy in container_policies if policy["containerName"] == "main"
        ]
        if len(matching_container_policies) != 1:
            return None
        policy = matching_container_policies[0]
        return dict(
            min_cpu=str(policy["minAllowed"]["cpu"]),
            max_cpu=str(policy["maxAllowed"]["cpu"]),
            min_memory=str(policy["minAllowed"]["memory"]),
            max_memory=str(policy["maxAllowed"]["memory"]),
        )

    @staticmethod
    def _get_async_autoscaling_params(
        deployment_config,
    ) -> HorizontalAutoscalingEndpointParams:
        metadata_annotations = deployment_config.metadata.annotations
        return dict(
            min_workers=metadata_annotations["celery.scaleml.autoscaler/minWorkers"],
            max_workers=metadata_annotations["celery.scaleml.autoscaler/maxWorkers"],
            per_worker=metadata_annotations["celery.scaleml.autoscaler/perWorker"],
        )

    @staticmethod
    def _get_sync_autoscaling_params(
        hpa_config: V2beta2HorizontalPodAutoscaler,
    ) -> HorizontalAutoscalingEndpointParams:
        spec = hpa_config.spec
        per_worker = get_per_worker_value_from_target_concurrency(
            spec.metrics[0].pods.target.average_value
        )
        return dict(
            max_workers=spec.max_replicas,
            min_workers=spec.min_replicas,
            per_worker=per_worker,
        )

    @staticmethod
    def _get_sync_autoscaling_params_from_keda(
        keda_config,
    ) -> HorizontalAutoscalingEndpointParams:
        spec = keda_config["spec"]
        return dict(
            max_workers=spec.get("maxReplicaCount"),
            min_workers=spec.get("minReplicaCount"),
            per_worker=1,  # TODO dummy value, fill in when we autoscale from 0 to 1
        )

    async def _get_resources(
        self, endpoint_id: str, deployment_name: str, endpoint_type: ModelEndpointType
    ) -> ModelEndpointInfraState:
        apps_client = get_kubernetes_apps_client()
        k8s_resource_group_name = _endpoint_id_to_k8s_resource_group_name(endpoint_id)
        try:
            deployment_config = await apps_client.read_namespaced_deployment(
                name=k8s_resource_group_name, namespace=hmi_config.endpoint_namespace
            )
        except ApiException as e:
            if e.status == 404:
                logger.warning(
                    f"Could not find resource, falling back to legacy deployment_name: "
                    f"{k8s_resource_group_name=}, {endpoint_id=}, {deployment_name=}"
                )
                k8s_resource_group_name = deployment_name
                deployment_config = await apps_client.read_namespaced_deployment(
                    name=k8s_resource_group_name,
                    namespace=hmi_config.endpoint_namespace,
                )
            else:
                raise

        common_params = self._get_common_endpoint_params(deployment_config)
        if endpoint_type == ModelEndpointType.ASYNC:
            horizontal_autoscaling_params = self._get_async_autoscaling_params(deployment_config)
        elif endpoint_type in {ModelEndpointType.SYNC, ModelEndpointType.STREAMING}:
            autoscaling_client = get_kubernetes_autoscaling_client()
            custom_object_client = get_kubernetes_custom_objects_client()
            try:
                hpa_config = await autoscaling_client.read_namespaced_horizontal_pod_autoscaler(
                    k8s_resource_group_name, hmi_config.endpoint_namespace
                )
            except ApiException as e:
                if e.status == 404:
                    hpa_config = None
                else:
                    raise e
            try:
                keda_scaled_object_config = await custom_object_client.get_namespaced_custom_object(
                    group="keda.sh",
                    version="v1alpha1",
                    namespace=hmi_config.endpoint_namespace,
                    plural="scaledobjects",
                    name=k8s_resource_group_name,
                )
            except ApiException:
                keda_scaled_object_config = None
            if hpa_config is not None:
                horizontal_autoscaling_params = self._get_sync_autoscaling_params(hpa_config)
            elif keda_scaled_object_config is not None:
                horizontal_autoscaling_params = self._get_sync_autoscaling_params_from_keda(
                    keda_scaled_object_config
                )
            else:
                raise EndpointResourceInfraException(
                    f"Could not find autoscaling config for {endpoint_type}"
                )
        else:
            raise ValueError(f"Unexpected endpoint type {endpoint_type}")

        vertical_autoscaling_params = None
        custom_objects_client = get_kubernetes_custom_objects_client()
        try:
            vpa_config = await custom_objects_client.get_namespaced_custom_object(
                group="autoscaling.k8s.io",
                version="v1",
                namespace=hmi_config.endpoint_namespace,
                plural="verticalpodautoscalers",
                name=k8s_resource_group_name,
            )
            vertical_autoscaling_params = self._get_vertical_autoscaling_params(vpa_config)
        except ApiException as e:
            if e.status == 404:
                pass

        config_maps = await self._get_config_maps(
            endpoint_id=endpoint_id, deployment_name=k8s_resource_group_name
        )
        launch_container = self._get_launch_container(deployment_config)
        envlist = launch_container.env
        # Note: the env var PREWARM is either "true" or "false" string (or doesn't exist for legacy)
        # Convert this as early as possible to Optional[bool] to avoid bugs
        prewarm = str_to_bool(self._get_env_value_from_envlist(envlist, "PREWARM"))

        high_priority = (
            deployment_config.spec.template.spec.priority_class_name == LAUNCH_HIGH_PRIORITY_CLASS
        )

        infra_state = ModelEndpointInfraState(
            deployment_name=k8s_resource_group_name,
            aws_role=common_params["aws_role"],
            results_s3_bucket=common_params["results_s3_bucket"],
            child_fn_info=None,
            labels=common_params["labels"],
            prewarm=prewarm,
            high_priority=high_priority,
            deployment_state=ModelEndpointDeploymentState(
                min_workers=horizontal_autoscaling_params["min_workers"],
                max_workers=horizontal_autoscaling_params["max_workers"],
                per_worker=int(horizontal_autoscaling_params["per_worker"]),
                available_workers=deployment_config.status.available_replicas or 0,
                unavailable_workers=deployment_config.status.unavailable_replicas or 0,
            ),
            resource_state=ModelEndpointResourceState(
                cpus=common_params["cpus"],
                gpus=common_params["gpus"],
                gpu_type=common_params["gpu_type"],  # type: ignore
                memory=common_params["memory"],
                storage=common_params["storage"],
                optimize_costs=(vertical_autoscaling_params is not None),
            ),
            user_config_state=self._translate_k8s_config_maps_to_user_config_data(
                k8s_resource_group_name, config_maps
            ),
            image=common_params["image"],
            num_queued_items=None,
        )
        return infra_state

    async def _get_all_resources(
        self,
    ) -> Dict[str, Tuple[bool, ModelEndpointInfraState]]:
        apps_client = get_kubernetes_apps_client()
        autoscaling_client = get_kubernetes_autoscaling_client()
        custom_objects_client = get_kubernetes_custom_objects_client()
        deployments = (
            await apps_client.list_namespaced_deployment(namespace=hmi_config.endpoint_namespace)
        ).items
        hpas = (
            await autoscaling_client.list_namespaced_horizontal_pod_autoscaler(
                namespace=hmi_config.endpoint_namespace
            )
        ).items
        try:
            vpas = (
                await custom_objects_client.list_namespaced_custom_object(
                    group="autoscaling.k8s.io",
                    version="v1",
                    namespace=hmi_config.endpoint_namespace,
                    plural="verticalpodautoscalers",
                )
            )["items"]
        except ApiException as e:
            if e.status == 404:
                vpas = []
            else:
                raise
        try:
            keda_scaled_objects = (
                await custom_objects_client.list_namespaced_custom_object(
                    group="keda.sh",
                    version="v1alpha1",
                    namespace=hmi_config.endpoint_namespace,
                    plural="scaledobjects",
                )
            )["items"]
        except ApiException as e:
            if e.status == 404:
                keda_scaled_objects = []
            else:
                raise

        deployments_by_name = {deployment.metadata.name: deployment for deployment in deployments}
        hpas_by_name = {hpa.metadata.name: hpa for hpa in hpas}
        vpas_by_name = {vpa["metadata"]["name"]: vpa for vpa in vpas}
        keda_scaled_objects_by_name = {kso["metadata"]["name"]: kso for kso in keda_scaled_objects}
        all_config_maps = await self._get_all_config_maps()
        # can safely assume hpa with same name as deployment corresponds to the same Launch Endpoint
        logger.info(f"Orphaned hpas: {set(hpas_by_name).difference(set(deployments_by_name))}")
        logger.info(f"Orphaned vpas: {set(vpas_by_name).difference(set(deployments_by_name))}")
        infra_states = {}
        logger.info(f"Got data for {list(deployments_by_name.keys())}")
        for name, deployment_config in deployments_by_name.items():
            try:
                hpa_config = hpas_by_name.get(name, None)
                vpa_config = vpas_by_name.get(name, None)
                keda_scaled_object_config = keda_scaled_objects_by_name.get(name, None)
                common_params = self._get_common_endpoint_params(deployment_config)
                launch_container = self._get_launch_container(deployment_config)

                envlist = launch_container.env
                # Convert as early as possible to Optional[bool] to avoid bugs
                prewarm = str_to_bool(self._get_env_value_from_envlist(envlist, "PREWARM"))

                high_priority = (
                    deployment_config.spec.template.spec.priority_class_name
                    == LAUNCH_HIGH_PRIORITY_CLASS
                )

                if hpa_config:
                    # Assume it's a sync endpoint
                    # TODO I think this is correct but only barely, it introduces a coupling between
                    #   an HPA (or keda SO) existing and an endpoint being a sync endpoint. The "more correct"
                    #   thing to do is to query the db to get the endpoints, but it doesn't belong here
                    horizontal_autoscaling_params = self._get_sync_autoscaling_params(hpa_config)
                elif keda_scaled_object_config:
                    # Also assume it's a sync endpoint
                    horizontal_autoscaling_params = self._get_sync_autoscaling_params_from_keda(
                        keda_scaled_object_config
                    )
                else:
                    horizontal_autoscaling_params = self._get_async_autoscaling_params(
                        deployment_config
                    )
                vertical_autoscaling_params = None
                if vpa_config:
                    vertical_autoscaling_params = self._get_vertical_autoscaling_params(vpa_config)
                infra_state = ModelEndpointInfraState(
                    deployment_name=name,
                    aws_role=common_params["aws_role"],
                    results_s3_bucket=common_params["results_s3_bucket"],
                    child_fn_info=None,
                    labels=common_params["labels"],
                    prewarm=prewarm,
                    high_priority=high_priority,
                    deployment_state=ModelEndpointDeploymentState(
                        min_workers=horizontal_autoscaling_params["min_workers"],
                        max_workers=horizontal_autoscaling_params["max_workers"],
                        per_worker=horizontal_autoscaling_params["per_worker"],
                        available_workers=deployment_config.status.available_replicas or 0,
                        unavailable_workers=deployment_config.status.unavailable_replicas or 0,
                    ),
                    resource_state=ModelEndpointResourceState(
                        cpus=common_params["cpus"],
                        gpus=common_params["gpus"],
                        gpu_type=common_params["gpu_type"],  # type: ignore
                        memory=common_params["memory"],
                        storage=common_params["storage"],
                        optimize_costs=(vertical_autoscaling_params is not None),
                    ),
                    user_config_state=self._translate_k8s_config_maps_to_user_config_data(
                        name, all_config_maps
                    ),
                    image=common_params["image"],
                    num_queued_items=None,
                )
                if name.startswith("launch-endpoint-id-"):
                    key = _k8s_resource_group_name_to_endpoint_id(name)
                    is_key_an_endpoint_id = True
                else:
                    key = name
                    is_key_an_endpoint_id = False

                infra_states[key] = (is_key_an_endpoint_id, infra_state)
            except Exception:
                logger.exception(f"Error parsing deployment {name}")
        return infra_states

    async def _delete_resources_async(self, endpoint_id: str, deployment_name: str) -> bool:
        deployment_delete_succeeded = await self._delete_deployment(
            endpoint_id=endpoint_id, deployment_name=deployment_name
        )
        config_map_delete_succeeded = await self._delete_config_maps(
            endpoint_id=endpoint_id, deployment_name=deployment_name
        )
        await self._delete_vpa(endpoint_id=endpoint_id)
        await self._delete_pdb(endpoint_id=endpoint_id)
        return deployment_delete_succeeded and config_map_delete_succeeded

    async def _delete_resources_sync(self, endpoint_id: str, deployment_name: str) -> bool:
        deployment_delete_succeeded = await self._delete_deployment(
            endpoint_id=endpoint_id,
            deployment_name=deployment_name,
        )
        config_map_delete_succeeded = await self._delete_config_maps(
            endpoint_id=endpoint_id, deployment_name=deployment_name
        )
        service_delete_succeeded = await self._delete_service(
            endpoint_id=endpoint_id, deployment_name=deployment_name
        )
        # we should have created exactly one of an HPA or a keda scaled object
        hpa_delete_succeeded = await self._delete_hpa(
            endpoint_id=endpoint_id, deployment_name=deployment_name
        )
        keda_scaled_object_succeeded = await self._delete_keda_scaled_object(
            endpoint_id=endpoint_id
        )
        await self._delete_vpa(endpoint_id=endpoint_id)
        await self._delete_pdb(endpoint_id=endpoint_id)

        destination_rule_delete_succeeded = await self._delete_destination_rule(
            endpoint_id=endpoint_id
        )
        virtual_service_delete_succeeded = await self._delete_virtual_service(
            endpoint_id=endpoint_id
        )

        return (
            deployment_delete_succeeded
            and config_map_delete_succeeded
            and service_delete_succeeded
            and (hpa_delete_succeeded or keda_scaled_object_succeeded)
            and destination_rule_delete_succeeded
            and virtual_service_delete_succeeded
        )
