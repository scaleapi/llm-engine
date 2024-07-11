import datetime
from typing import Dict, List, Optional

from model_engine_server.domain.entities import GpuType
from model_engine_server.domain.entities.owned_entity import OwnedEntity
from pydantic import ConfigDict


class DockerImageBatchJobBundle(OwnedEntity):
    id: str
    name: str
    created_by: str
    created_at: datetime.datetime
    owner: str
    image_repository: str
    image_tag: str
    command: List[str]
    env: Dict[str, str]
    mount_location: Optional[str] = None
    cpus: Optional[str] = None
    memory: Optional[str] = None
    storage: Optional[str] = None
    gpus: Optional[int] = None
    gpu_type: Optional[GpuType] = None
    public: Optional[bool] = None
    model_config = ConfigDict(from_attributes=True)
