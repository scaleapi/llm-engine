"""Read configmap from k8s."""

from typing import Dict

from kubernetes_asyncio import client
from kubernetes_asyncio import config as kube_config
from kubernetes_asyncio.client.rest import ApiException
from kubernetes_asyncio.config.config_exception import ConfigException
from model_engine_server.common.config import hmi_config
from model_engine_server.core.loggers import logger_name, make_logger

DEFAULT_NAMESPACE = "default"

logger = make_logger(logger_name())


async def read_config_map(
    config_map_name: str, namespace: str = hmi_config.gateway_namespace
) -> Dict[str, str]:
    try:
        kube_config.load_incluster_config()
    except ConfigException:
        logger.info("No incluster kubernetes config, falling back to local")
        await kube_config.load_kube_config()

    core_api = client.CoreV1Api()

    try:
        config_map = await core_api.read_namespaced_config_map(
            name=config_map_name, namespace=namespace
        )
        return config_map.data
    except ApiException as e:
        logger.exception(f"Error reading configmap {config_map_name}")
        raise e


# TODO: figure out what this does
