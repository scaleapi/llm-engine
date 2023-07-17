from typing import Any, Dict, List

from pydantic import BaseModel


class LLMFineTuneJobTemplate(BaseModel):
    docker_image_batch_job_bundle_id: str
    launch_bundle_config: Dict[str, Any]
    launch_endpoint_config: Dict[str, Any]
    default_hparams: Dict[str, Any]
    required_params: List[str]

    class Config:
        orm_mode = True
