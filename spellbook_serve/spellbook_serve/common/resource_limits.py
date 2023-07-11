from typing import Optional

from spellbook_serve.domain.entities import CpuSpecificationType, GpuType, StorageSpecificationType
from spellbook_serve.domain.exceptions import EndpointResourceInvalidRequestException
from spellbook_serve.infra.gateways.k8s_resource_parser import (
    parse_mem_request,
    validate_mem_request,
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
STORAGE_LIMIT = "500G"  # TODO: figure out an actual limit.
REQUESTS_BY_GPU_TYPE = {
    None: CPU_INSTANCE_LIMITS,
    GpuType.NVIDIA_TESLA_T4: T4_INSTANCE_LIMITS,
    GpuType.NVIDIA_AMPERE_A10: A10_INSTANCE_LIMITS,
    GpuType.NVIDIA_AMPERE_A100: A100_INSTANCE_LIMITS,
}

FORWARDER_CPU_USAGE = 0.5
FORWARDER_MEMORY_USAGE = "1Gi"
FORWARDER_STORAGE_USAGE = "1G"


def validate_resource_requests(
    cpus: Optional[CpuSpecificationType],
    memory: Optional[StorageSpecificationType],
    storage: Optional[StorageSpecificationType],
    gpus: Optional[int],
    gpu_type: Optional[GpuType],
) -> None:
    """Validates whether cpu/memory requests are reasonable"""

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
        # TODO: there should be a parse_cpu_request fn analagous to parse_mem_request.
        if float(cpus) <= 0:
            raise EndpointResourceInvalidRequestException("Must provide positive cpus")
        if float(cpus) > resource_limits["cpus"]:  # type: ignore
            raise EndpointResourceInvalidRequestException(f"Requested cpus {cpus} too high")
    if memory is not None:
        if isinstance(memory, str):
            if not validate_mem_request(memory):
                raise EndpointResourceInvalidRequestException(
                    f"Requested memory {memory} is incorrectly formatted"
                )
            requested_memory = parse_mem_request(memory)
        else:
            requested_memory = memory

        if requested_memory > parse_mem_request(resource_limits["memory"]):  # type: ignore
            raise EndpointResourceInvalidRequestException(f"Requested memory {memory} too high")
    if storage is not None:
        if isinstance(storage, str):
            if not validate_mem_request(storage):
                raise EndpointResourceInvalidRequestException(
                    f"Requested storage {storage} is incorrectly formatted"
                )
            requested_storage = parse_mem_request(storage)
        else:
            requested_storage = storage
        if requested_storage > parse_mem_request(STORAGE_LIMIT):
            raise EndpointResourceInvalidRequestException(f"Requested storage {storage} too high.")
