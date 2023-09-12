from dataclasses import dataclass
from enum import Enum
from typing import Optional


class LLMSource(str, Enum):
    HUGGING_FACE = "hugging_face"


class LLMInferenceFramework(str, Enum):
    DEEPSPEED = "deepspeed"
    TEXT_GENERATION_INFERENCE = "text_generation_inference"
    VLLM = "vllm"
    LIGHTLLM = "lightllm"


class Quantization(str, Enum):
    BITSANDBYTES = "bitsandbytes"


@dataclass
class LLMMetadata:
    model_name: str
    source: LLMSource
    inference_framework: LLMInferenceFramework
    inference_framework_image_tag: str
    num_shards: int
    quantize: Optional[Quantization] = None
