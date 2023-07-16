import datetime
from typing import Dict, List, Optional

from llm_engine_server.domain.entities import GpuType
from llm_engine_server.domain.entities.owned_entity import OwnedEntity


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
    mount_location: Optional[str]
    cpus: Optional[str]
    memory: Optional[str]
    storage: Optional[str]
    gpus: Optional[int]
    gpu_type: Optional[GpuType]

    class Config:
        orm_mode = True
