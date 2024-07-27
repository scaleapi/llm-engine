from typing import Dict, Optional

from model_engine_server.common.pydantic_types import BaseModel


class BuildImageRequest(BaseModel):
    repo: str
    image_tag: str
    aws_profile: str
    base_path: str
    dockerfile: str
    base_image: str
    requirements_folder: Optional[str] = None
    substitution_args: Optional[Dict[str, str]] = None


class BuildImageResponse(BaseModel):
    status: bool
    logs: str
    job_name: str


# TODO: We may want to add a DTO for streaming logs from the docker build to users.
