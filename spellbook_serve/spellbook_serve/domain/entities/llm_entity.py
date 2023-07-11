from dataclasses import dataclass
from enum import Enum


class LLMSource(str, Enum):
    HUGGING_FACE = "hugging_face"


class LLMInferenceFramework(str, Enum):
    DEEPSPEED = "deepspeed"


@dataclass
class LLMMetadata:
    model_name: str
    source: LLMSource
    inference_framework: LLMInferenceFramework
    inference_framework_image_tag: str
    num_shards: int
