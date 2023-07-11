from enum import Enum


class GpuType(str, Enum):
    """Lists allowed GPU types for Launch."""

    NVIDIA_TESLA_T4 = "nvidia-tesla-t4"
    NVIDIA_AMPERE_A10 = "nvidia-ampere-a10"
    NVIDIA_AMPERE_A100 = "nvidia-a100"
