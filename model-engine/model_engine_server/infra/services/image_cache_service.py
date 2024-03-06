from datetime import datetime
from typing import Dict, NamedTuple, Tuple

import pytz
from model_engine_server.common.config import hmi_config
from model_engine_server.common.env_vars import CIRCLECI, GIT_TAG
from model_engine_server.core.config import infra_config
from model_engine_server.core.loggers import logger_name, make_logger
from model_engine_server.domain.entities import GpuType, ModelEndpointInfraState
from model_engine_server.domain.repositories import DockerRepository
from model_engine_server.infra.gateways.resources.image_cache_gateway import (
    CachedImages,
    ImageCacheGateway,
)
from model_engine_server.infra.repositories.model_endpoint_record_repository import (
    ModelEndpointRecordRepository,
)

logger = make_logger(logger_name())

IMAGES_TO_CACHE_PER_INSTANCE_TYPE = 32

CachePriority = NamedTuple(
    "CachePriority",
    (
        ("is_high_priority", int),
        ("has_no_available_workers", int),
        ("last_updated_at", datetime),
    ),
)

DockerImage = NamedTuple(
    "DockerImage",
    (
        ("repo", str),
        ("tag", str),
    ),
)


class ImageCacheService:
    """
    Represents reading from k8s and writing images to the k8s image cache.
    """

    def __init__(
        self,
        model_endpoint_record_repository: ModelEndpointRecordRepository,
        image_cache_gateway: ImageCacheGateway,
        docker_repository: DockerRepository,
    ):
        self.model_endpoint_record_repository = model_endpoint_record_repository
        self.image_cache_gateway = image_cache_gateway
        self.docker_repository = docker_repository

    def _cache_finetune_llm_images(
        self, images_to_cache_priority: Dict[str, Dict[str, CachePriority]]
    ):
        """
        Cache images used by fine tune LLM endpoints to reduce cold start time.
        """
        # a cache priority to ensure llm endpoint images are always prioritized
        llm_image_cache_priority = CachePriority(
            is_high_priority=1,  # make it a high priority
            has_no_available_workers=1,
            # assuming it has no available workers so that it will be at top after reverse sorting
            last_updated_at=datetime.max.replace(tzinfo=pytz.utc),
            # setting it to max to ensure it will be at top after reverse sorting
        )

        istio_image = DockerImage("gcr.io/istio-release/proxyv2", "1.15.0")
        tgi_image_110 = DockerImage(
            f"{infra_config().docker_repo_prefix}/{hmi_config.tgi_repository}", "1.1.0"
        )
        vllm_image_027 = DockerImage(
            f"{infra_config().docker_repo_prefix}/{hmi_config.vllm_repository}", "0.2.7"
        )
        vllm_image_032 = DockerImage(
            f"{infra_config().docker_repo_prefix}/{hmi_config.vllm_repository}", "0.3.2"
        )
        latest_tag = (
            self.docker_repository.get_latest_image_tag(hmi_config.batch_inference_vllm_repository)
            if not CIRCLECI
            else "fake_docker_repository_latest_image_tag"
        )
        vllm_batch_image_latest = DockerImage(
            f"{infra_config().docker_repo_prefix}/{hmi_config.batch_inference_vllm_repository}",
            latest_tag,
        )
        forwarder_image = DockerImage(
            f"{infra_config().docker_repo_prefix}/launch/gateway", GIT_TAG
        )

        for llm_image in [
            istio_image,
            tgi_image_110,
            vllm_image_027,
            vllm_image_032,
            vllm_batch_image_latest,
            forwarder_image,
        ]:
            if self.docker_repository.is_repo_name(
                llm_image.repo
            ) and not self.docker_repository.image_exists(llm_image.tag, llm_image.repo):
                logger.warning(
                    f"Image {llm_image.repo}:{llm_image.tag} does not exist. Skipping caching ..."
                )
                continue
            image = f"{llm_image.repo}:{llm_image.tag}"
            for key in ["a10", "a100"]:
                images_to_cache_priority[key][image] = llm_image_cache_priority

    async def execute(self, endpoint_infra_states: Dict[str, Tuple[bool, ModelEndpointInfraState]]):
        images_to_cache_priority: Dict[str, Dict[str, CachePriority]] = {
            "cpu": {},
            "a10": {},
            "a100": {},
            "t4": {},
        }

        self._cache_finetune_llm_images(images_to_cache_priority)

        for endpoint_id, (_, state) in endpoint_infra_states.items():
            record = await self.model_endpoint_record_repository.get_model_endpoint_record(
                endpoint_id
            )

            if record is None:
                continue

            last_updated_at = (
                record.last_updated_at.replace(tzinfo=pytz.utc)
                if record.last_updated_at is not None
                else datetime.min.replace(tzinfo=pytz.utc)
            )
            has_no_available_workers = int(state.deployment_state.available_workers == 0)
            is_high_priority = int(state.high_priority is True)

            # TODO: Adding for image cache stability and to make it faster. Remove this
            # condition when things are proven to run smoothly.
            if not state.high_priority:
                continue

            cache_priority = CachePriority(
                is_high_priority=is_high_priority,
                has_no_available_workers=has_no_available_workers,
                last_updated_at=last_updated_at,
            )

            image_repository_and_tag = state.image.split("/", 1)[1]
            repository_name, image_tag = image_repository_and_tag.split(":")
            if state.resource_state.gpus == 0 and (
                (
                    state.image not in images_to_cache_priority["cpu"]
                    or last_updated_at.replace(tzinfo=pytz.utc)
                    > images_to_cache_priority["cpu"][state.image].last_updated_at.replace(
                        tzinfo=pytz.utc
                    )
                )
                and self.docker_repository.image_exists(image_tag, repository_name)
            ):
                images_to_cache_priority["cpu"][state.image] = cache_priority
            elif state.resource_state.gpus > 0:
                for gpu_type, key in [
                    (GpuType.NVIDIA_AMPERE_A10, "a10"),
                    (GpuType.NVIDIA_AMPERE_A100, "a100"),
                    (GpuType.NVIDIA_TESLA_T4, "t4"),
                ]:
                    if state.resource_state.gpu_type == gpu_type and (
                        (
                            state.image not in images_to_cache_priority[key]
                            or last_updated_at.replace(tzinfo=pytz.utc)
                            > images_to_cache_priority[key][state.image].last_updated_at.replace(
                                tzinfo=pytz.utc
                            )
                        )
                        and self.docker_repository.image_exists(image_tag, repository_name)
                    ):
                        images_to_cache_priority[key][state.image] = cache_priority
        images_to_cache = CachedImages(cpu=[], a10=[], a100=[], t4=[])
        for key, val in images_to_cache_priority.items():
            images_to_cache[key] = sorted(  # type: ignore
                val.keys(), key=lambda image: val[image], reverse=True
            )[:IMAGES_TO_CACHE_PER_INSTANCE_TYPE]

        await self.image_cache_gateway.create_or_update_image_cache(images_to_cache)
