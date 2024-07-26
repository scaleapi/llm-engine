"""
Contains various input and output types relating to Triggers for the server.
"""
import datetime
from typing import Any, Dict, List, Optional

from model_engine_server.common.pydantic_types import BaseModel, ConfigDict, Field


class CreateTriggerV1Request(BaseModel):
    name: str
    cron_schedule: str
    bundle_id: str
    default_job_config: Optional[Dict[str, Any]] = None
    default_job_metadata: Optional[Dict[str, str]] = None


class CreateTriggerV1Response(BaseModel):
    trigger_id: str


class GetTriggerV1Response(BaseModel):
    id: str
    name: str
    owner: str
    created_by: str
    created_at: datetime.datetime
    cron_schedule: str
    docker_image_batch_job_bundle_id: str
    default_job_config: Optional[Dict[str, Any]] = Field(default=None)
    default_job_metadata: Optional[Dict[str, str]] = Field(default=None)
    model_config = ConfigDict(from_attributes=True)


class ListTriggersV1Response(BaseModel):
    triggers: List[GetTriggerV1Response]


class UpdateTriggerV1Request(BaseModel):
    cron_schedule: Optional[str] = None
    suspend: Optional[bool] = None


class UpdateTriggerV1Response(BaseModel):
    success: bool


class DeleteTriggerV1Response(BaseModel):
    success: bool
