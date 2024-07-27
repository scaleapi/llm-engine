from typing import Any, Dict, List, Optional

from model_engine_server.common.pydantic_types import BaseModel, ConfigDict


class LLMFineTuneTemplate(BaseModel):
    docker_image_batch_job_bundle_id: str
    launch_endpoint_config: Dict[str, Any]
    default_hparams: Dict[str, Any]
    required_params: List[str]
    model_config = ConfigDict(from_attributes=True)


class LLMFineTuneEvent(BaseModel):
    timestamp: Optional[float] = None
    message: str
    level: str
