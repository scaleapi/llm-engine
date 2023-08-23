from typing import Any, Dict, List, Optional

from pydantic import BaseModel


class LLMFineTuneTemplate(BaseModel):
    docker_image_batch_job_bundle_id: str
    launch_endpoint_config: Dict[str, Any]
    default_hparams: Dict[str, Any]
    required_params: List[str]

    class Config:
        orm_mode = True


class LLMFineTuneEvent(BaseModel):
    timestamp: Optional[float] = None
    message: str
    level: str
