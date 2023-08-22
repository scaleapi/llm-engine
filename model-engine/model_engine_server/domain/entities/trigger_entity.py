import datetime
from typing import Any, Dict, Optional

from model_engine_server.domain.entities.owned_entity import OwnedEntity


class Trigger(OwnedEntity):
    id: str
    name: str
    owner: str
    created_by: str
    created_at: datetime.datetime

    cron_schedule: str
    docker_image_batch_job_bundle_id: str
    default_job_config: Optional[Dict[str, Any]]
    default_job_metadata: Optional[Dict[str, str]]

    class Config:
        orm_mode = True
