from enum import Enum


class GpuType(str, Enum):
    """Lists allowed GPU types for Launch."""

    NVIDIA_TESLA_T4 = "nvidia-tesla-t4"
    NVIDIA_AMPERE_A10 = "nvidia-ampere-a10"
    NVIDIA_AMPERE_A100 = "nvidia-ampere-a100"
    NVIDIA_AMPERE_A100E = "nvidia-ampere-a100e"
    NVIDIA_HOPPER_H100 = "nvidia-hopper-h100"
    NVIDIA_HOPPER_H100_1G_20GB = "nvidia-hopper-h100-1g20gb"
    NVIDIA_HOPPER_H100_3G_40GB = "nvidia-hopper-h100-3g40gb"
