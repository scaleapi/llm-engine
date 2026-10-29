import os
from typing import Any, Dict, List, TypedDict, cast

from kubernetes_asyncio.client.rest import ApiException
from model_engine_server.common.config import hmi_config
from model_engine_server.core.loggers import logger_name, make_logger
from model_engine_server.infra.gateways.resources.k8s_endpoint_resource_delegate import (
    get_kubernetes_apps_client,  # If you ever add more imports here, update test_image_cache_gateway accordingly, otherwise you will likely mangle live cluster resources
)
from model_engine_server.infra.gateways.resources.k8s_endpoint_resource_delegate import (
    k8s_yaml_exists,
    load_k8s_yaml,
)
from model_engine_server.infra.gateways.resources.k8s_resource_types import (
    ImageCacheArguments,
    compute_image_hash,
)

logger = make_logger(logger_name())

# Cached images are only pulled, never trusted to run: every container sleeps with this static
# busybox binary, copied in by an init container, instead of a binary from the cached image.
# The digest is pinned; the repository follows the template's busybox image (utilityImages.busybox)
# so a mirror can serve it.
TRUSTED_BUSYBOX_DIGEST = (
    "sha256:5cec3fc171c87218698e85a52af7087de727372aae264a787b8112901a5b0092"  # 1.37.0-musl
)
TRUSTED_BUSYBOX_IMAGE = f"public.ecr.aws/docker/library/busybox@{TRUSTED_BUSYBOX_DIGEST}"
# The largest sleep busybox accepts, so cache containers never exit.
CACHE_CONTAINER_SLEEP_SECONDS = "2147483647"
TRUSTED_BIN_INIT_CONTAINER_NAME = "copy-trusted-busybox"
TRUSTED_BIN_VOLUME_NAME = "trusted-bin"
TRUSTED_BIN_DIR = "/trusted"
NOBODY_ID = 65534
CONTAINER_SECURITY_CONTEXT = {
    "allowPrivilegeEscalation": False,
    "readOnlyRootFilesystem": True,
    "capabilities": {"drop": ["ALL"]},
}
CONTAINER_RESOURCES = {
    "requests": {"cpu": "1m", "memory": "8Mi"},
    "limits": {"cpu": "10m", "memory": "16Mi"},
}
TRUSTED_BIN_VOLUME_MOUNT = {
    "name": TRUSTED_BIN_VOLUME_NAME,
    "mountPath": TRUSTED_BIN_DIR,
    "readOnly": True,
}


def _trusted_busybox_image(template_containers: List[Dict[str, Any]]) -> str:
    template_busybox = next(
        (c["image"] for c in template_containers if c.get("name") == "busybox"), None
    )
    if template_busybox is None:
        return TRUSTED_BUSYBOX_IMAGE
    repository = template_busybox.split("@")[0]
    registry, _, name = repository.rpartition("/")
    repository = f"{registry}/{name.split(':')[0]}" if registry else name.split(":")[0]
    return f"{repository}@{TRUSTED_BUSYBOX_DIGEST}"


class CachedImages(TypedDict):
    cpu: List[str]
    a10: List[str]
    a100: List[str]
    t4: List[str]
    h100: List[str]
    h100_3g40gb: List[str]
    h100_1g20gb: List[str]


class ImageCacheGateway:
    async def create_or_update_image_cache(self, cached_images: CachedImages) -> None:
        """
        Creates or updates the image cache in the gateway.

        Args:
            cached_images: The images to cache.
        """
        base_path = os.getenv("WORKSPACE")
        if base_path is None:
            raise EnvironmentError("WORKSPACE env variable not found")
        base_name = "launch-image-cache"

        for compute_type, images in cached_images.items():
            # Required for mypy TypedDict
            compute_type = cast(str, compute_type)
            compute_type = compute_type.replace("_", "-")  # for k8s valid name
            images = cast(list, images)

            name = f"{base_name}-{compute_type}"
            substitution_kwargs = ImageCacheArguments(
                RESOURCE_NAME=name,
                NAMESPACE=hmi_config.endpoint_namespace,
            )
            resource_key = f"image-cache-{compute_type}.yaml"
            if not k8s_yaml_exists(resource_key):
                logger.info(f"Didn't find yaml for {compute_type}, skipping")
                continue
            image_cache = load_k8s_yaml(resource_key, substitution_kwargs)

            labels = image_cache["spec"]["template"]["metadata"]["labels"]
            containers = image_cache["spec"]["template"]["spec"]["containers"]
            trusted_busybox_image = _trusted_busybox_image(containers)
            for image in images:
                image_hash = compute_image_hash(image)
                labels[image_hash] = "True"

                base_container_dict = {
                    "imagePullPolicy": "IfNotPresent",
                    "command": [
                        f"{TRUSTED_BIN_DIR}/busybox",
                        "sleep",
                        CACHE_CONTAINER_SLEEP_SECONDS,
                    ],
                    "securityContext": CONTAINER_SECURITY_CONTEXT,
                    "resources": CONTAINER_RESOURCES,
                    "volumeMounts": [TRUSTED_BIN_VOLUME_MOUNT],
                }
                base_container_dict["image"] = image
                base_container_dict["name"] = image_hash
                containers.append(base_container_dict)

            image_cache["spec"]["template"]["metadata"]["labels"] = labels
            image_cache["spec"]["template"]["spec"]["containers"] = containers
            pod_spec = image_cache["spec"]["template"]["spec"]
            pod_spec["automountServiceAccountToken"] = False
            pod_spec["securityContext"] = {
                "runAsNonRoot": True,
                "runAsUser": NOBODY_ID,
                "runAsGroup": NOBODY_ID,
            }
            pod_spec["volumes"] = [{"name": TRUSTED_BIN_VOLUME_NAME, "emptyDir": {}}]
            pod_spec["initContainers"] = [
                {
                    "name": TRUSTED_BIN_INIT_CONTAINER_NAME,
                    "image": trusted_busybox_image,
                    "imagePullPolicy": "IfNotPresent",
                    "command": ["/bin/cp", "/bin/busybox", f"{TRUSTED_BIN_DIR}/busybox"],
                    "securityContext": CONTAINER_SECURITY_CONTEXT,
                    "resources": CONTAINER_RESOURCES,
                    "volumeMounts": [{**TRUSTED_BIN_VOLUME_MOUNT, "readOnly": False}],
                }
            ]

            # Add the default image value defined in the yaml to the set of images
            images.append("public.ecr.aws/docker/library/busybox:latest")

            await self._create_image_cache(image_cache, name, images)

    @staticmethod
    async def _create_image_cache(
        image_cache: Dict[str, Any], name: str, images: List[str]
    ) -> None:
        """
        Lower-level function to create/patch a k8s ImageCache
        Args:
            image_cache: Image Cache body
            name: The name of the vpa on K8s

        Returns:
            Nothing; raises a k8s ApiException if failure

        """
        apps_api = get_kubernetes_apps_client()

        try:
            await apps_api.create_namespaced_daemon_set(
                namespace=hmi_config.endpoint_namespace,
                body=image_cache,
            )
            logger.info(f"Created image cache daemonset {name}")
        except ApiException as exc:
            if exc.status == 409:
                # Do not update existing daemonset if the cache is unchanged
                existing_daemonsets = await apps_api.list_namespaced_daemon_set(
                    namespace=hmi_config.endpoint_namespace
                )
                for daemonset in existing_daemonsets.items:
                    if daemonset.metadata.name == name:
                        containers = daemonset.spec.template.spec.containers
                        current_images = set([container.image for container in containers])
                        new_images = set(images)
                        # Daemonsets created before the trusted-busybox init container still run
                        # binaries from the cached images, so they are replaced.
                        init_containers = daemonset.spec.template.spec.init_containers or []
                        trusted_busybox_image = image_cache["spec"]["template"]["spec"][
                            "initContainers"
                        ][0]["image"]
                        has_trusted_bin = any(
                            container.name == TRUSTED_BIN_INIT_CONTAINER_NAME
                            and container.image == trusted_busybox_image
                            for container in init_containers
                        )
                        if current_images == new_images and has_trusted_bin:
                            logger.info(f"Image cache {name} has not changed, not updating")
                            return

                # Patching is an additive merge so using replace instead if the image cache has updated
                logger.info(
                    f"Image cache daemonset {name} already exists, replacing with new values"
                )
                await apps_api.replace_namespaced_daemon_set(
                    name=name, namespace=hmi_config.endpoint_namespace, body=image_cache
                )
            elif exc.status == 404:
                logger.exception("ImageCache API not found. Is the ImageCache CRD installed?")
            else:
                logger.exception(
                    f"Got an exception when trying to apply the image cache {name} daemonset"
                )
                raise
