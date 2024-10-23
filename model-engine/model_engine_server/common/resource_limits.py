from typing import Optional, Union, cast

from model_engine_server.core.loggers import logger_name, make_logger
from model_engine_server.domain.entities import (
    CpuSpecificationType,
    GpuType,
    ModelBundle,
    StorageSpecificationType,
    TritonEnhancedRunnableImageFlavor,
)
from model_engine_server.domain.entities.docker_image_batch_job_bundle_entity import (
    DockerImageBatchJobBundle,
)
from model_engine_server.domain.exceptions import EndpointResourceInvalidRequestException
from model_engine_server.infra.gateways.k8s_resource_parser import (
    format_bytes,
    parse_cpu_request,
    parse_mem_request,
)

MAX_ENDPOINT_SIZE = (
    50  # This may be a bit strict, but for now it feels reasonable, since 250 workers is too much
)
# Also note that this is the default max_workers value for batch tasks.
# Separately, we need user compute limits

# Individual cpu/mem limits for instance types. Corresponds to m5*, g4dn, g5dn AWS instance types.
CPU_INSTANCE_LIMITS = dict(cpus=7, memory="30Gi")
T4_INSTANCE_LIMITS = dict(
    cpus=63, memory="250Gi"
)  # Should we even allow multi-gpu instances? This allows the largest single-gpu g4dn instance.
A10_INSTANCE_LIMITS = dict(
    cpus=63, memory="250Gi"
)  # Should we allow multi-gpu instances? This allows the largest single-gpu g5dn instance.
# p4d.24xlarge, p4de.24xlarge
A100_INSTANCE_LIMITS = dict(cpus=95, memory="1000Gi")
H100_INSTANCE_LIMITS = dict(cpus=191, memory="2000Gi", storage="1300Gi")
H100_1G_20GB_INSTANCE_LIMITS = dict(cpus=47, memory="500Gi")
H100_3G_40GB_INSTANCE_LIMITS = dict(cpus=95, memory="1000Gi")
STORAGE_LIMIT = "640Gi"  # TODO: figure out an actual limit.
REQUESTS_BY_GPU_TYPE = {
    None: CPU_INSTANCE_LIMITS,
    GpuType.NVIDIA_TESLA_T4: T4_INSTANCE_LIMITS,
    GpuType.NVIDIA_AMPERE_A10: A10_INSTANCE_LIMITS,
    GpuType.NVIDIA_AMPERE_A100: A100_INSTANCE_LIMITS,
    GpuType.NVIDIA_AMPERE_A100E: A100_INSTANCE_LIMITS,
    GpuType.NVIDIA_HOPPER_H100: H100_INSTANCE_LIMITS,
    GpuType.NVIDIA_HOPPER_H100_1G_20GB: H100_1G_20GB_INSTANCE_LIMITS,
    GpuType.NVIDIA_HOPPER_H100_3G_40GB: H100_3G_40GB_INSTANCE_LIMITS,
}

FORWARDER_CPU_USAGE = 1
FORWARDER_MEMORY_USAGE = "2Gi"
FORWARDER_STORAGE_USAGE = "1G"
FORWARDER_WORKER_COUNT = 2

logger = make_logger(logger_name())


def validate_resource_requests(
    bundle: Optional[Union[ModelBundle, DockerImageBatchJobBundle]],
    cpus: Optional[CpuSpecificationType],
    memory: Optional[StorageSpecificationType],
    storage: Optional[StorageSpecificationType],
    gpus: Optional[int],
    gpu_type: Optional[GpuType],
) -> None:
    """Validates whether cpu/memory requests are reasonable. Shouldn't need to validate any nodes_per_worker in the multinode case"""

    if (gpus is None or gpus == 0) and gpu_type is not None:
        raise EndpointResourceInvalidRequestException(
            f"Cannot have {gpus=} when gpu_type is not None"
        )

    if gpus is not None and gpus > 0:
        if gpu_type is None:
            raise EndpointResourceInvalidRequestException("Must provide gpu_type if gpus > 0")
        if gpu_type not in REQUESTS_BY_GPU_TYPE:
            raise EndpointResourceInvalidRequestException(f"Unknown gpu_type {gpu_type}")

    resource_limits = REQUESTS_BY_GPU_TYPE[gpu_type]

    if cpus is not None:
        try:
            cpus = parse_cpu_request(cpus) / 1000.0 if isinstance(cpus, str) else cpus
        except ValueError:
            raise EndpointResourceInvalidRequestException(
                f"Requested {cpus=} is incorrectly formatted"
            )

        if cpus <= 0:
            raise EndpointResourceInvalidRequestException("Requested cpus must be positive")

        available_cpus_for_user = cast(float, resource_limits["cpus"])

        if isinstance(bundle, ModelBundle):
            cpus += FORWARDER_CPU_USAGE
            available_cpus_for_user -= FORWARDER_CPU_USAGE
            if isinstance(bundle.flavor, TritonEnhancedRunnableImageFlavor):
                if bundle.flavor.triton_num_cpu is None or bundle.flavor.triton_num_cpu < 1:
                    raise EndpointResourceInvalidRequestException(
                        "Triton deployments require at least one CPU"
                    )
                cpus += bundle.flavor.triton_num_cpu

        if cpus > cast(float, resource_limits["cpus"]):
            raise EndpointResourceInvalidRequestException(
                f"Requested {cpus=} too high. The maximum for {gpu_type=} is {available_cpus_for_user}"
            )

    if memory is not None:
        try:
            memory = parse_mem_request(memory) if isinstance(memory, str) else memory
        except ValueError:
            raise EndpointResourceInvalidRequestException(
                f"Requested {memory=} is incorrectly formatted"
            )

        if memory <= 0:
            raise EndpointResourceInvalidRequestException("Requested memory must be positive")

        available_memory_for_user = parse_mem_request(cast(str, resource_limits["memory"]))

        if isinstance(bundle, ModelBundle):
            memory += parse_mem_request(FORWARDER_MEMORY_USAGE)
            available_memory_for_user -= parse_mem_request(FORWARDER_MEMORY_USAGE)
            if bundle and isinstance(bundle.flavor, TritonEnhancedRunnableImageFlavor):
                if bundle.flavor.triton_memory is None:
                    logger.warning(
                        "No specified memory resources for Triton container! "
                        "You may experience eviction if scheduled onto pods with memory contention!\n"
                        "Set the `memory` and `triton_memory` values to guarantee memory resources!"
                    )
                else:
                    memory += parse_mem_request(bundle.flavor.triton_memory)

        if memory > parse_mem_request(cast(str, resource_limits["memory"])):
            raise EndpointResourceInvalidRequestException(
                f"Requested {memory=} too high. The maximum for {gpu_type=} is {format_bytes(available_memory_for_user)}"
            )

    if storage is not None:
        try:
            storage = parse_mem_request(storage) if isinstance(storage, str) else storage
        except ValueError:
            raise EndpointResourceInvalidRequestException(
                f"Requested {storage=} is incorrectly formatted"
            )

        if storage <= 0:
            raise EndpointResourceInvalidRequestException("Requested storage must be positive")

        available_storage_for_user = parse_mem_request(
            resource_limits.get("storage", STORAGE_LIMIT)  # type: ignore
        )

        if isinstance(bundle, ModelBundle):
            storage += parse_mem_request(FORWARDER_STORAGE_USAGE)
            available_storage_for_user -= parse_mem_request(FORWARDER_STORAGE_USAGE)
            if bundle and isinstance(bundle.flavor, TritonEnhancedRunnableImageFlavor):
                if bundle.flavor.triton_storage is None:
                    logger.warning(
                        "No specified Triton storage resources for deployment! "
                        "You may experience eviction if scheduled onto pods with disk space contention!\n"
                        "Set the `triton_storage` value to guarantee ephemeral-storage for the Triton container!"
                    )
                else:
                    storage += parse_mem_request(bundle.flavor.triton_storage)

        if storage > available_storage_for_user:
            raise EndpointResourceInvalidRequestException(
                f"Requested {storage=} too high. The maximum for {gpu_type=} is {format_bytes(available_storage_for_user)}"
            )

    if isinstance(bundle, ModelBundle) and isinstance(
        bundle.flavor, TritonEnhancedRunnableImageFlavor
    ):
        if gpus is None or gpus < 1:
            raise EndpointResourceInvalidRequestException(
                "Triton deployments require at least one GPU"
            )
        if gpu_type is None:
            raise EndpointResourceInvalidRequestException(
                "Triton deployments require a specific GPU machine type"
            )
