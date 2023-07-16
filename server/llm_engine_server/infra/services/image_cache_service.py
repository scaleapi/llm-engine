from datetime import datetime
from typing import Dict, NamedTuple, Tuple

from llm_engine_server.core.loggers import filename_wo_ext, make_logger
from llm_engine_server.domain.entities import GpuType, ModelEndpointInfraState
from llm_engine_server.domain.repositories import DockerRepository
from llm_engine_server.infra.gateways.resources.image_cache_gateway import (
    CachedImages,
    ImageCacheGateway,
)
from llm_engine_server.infra.repositories.model_endpoint_record_repository import (
    ModelEndpointRecordRepository,
)

logger = make_logger(filename_wo_ext(__name__))

IMAGES_TO_CACHE_PER_INSTANCE_TYPE = 32

CachePriority = NamedTuple(
    "CachePriority",
    (
        ("is_high_priority", int),
        ("has_no_available_workers", int),
        ("last_updated_at", datetime),
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

    async def execute(self, endpoint_infra_states: Dict[str, Tuple[bool, ModelEndpointInfraState]]):
        images_to_cache_priority: Dict[str, Dict[str, CachePriority]] = {
            "cpu": {},
            "a10": {},
            "a100": {},
            "t4": {},
        }
        for endpoint_id, (_, state) in endpoint_infra_states.items():
            record = await self.model_endpoint_record_repository.get_model_endpoint_record(
                endpoint_id
            )

            if record is None:
                continue

            last_updated_at = record.last_updated_at or datetime.min
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
                    or last_updated_at
                    > images_to_cache_priority["cpu"][state.image].last_updated_at
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
                            or last_updated_at
                            > images_to_cache_priority[key][state.image].last_updated_at
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
