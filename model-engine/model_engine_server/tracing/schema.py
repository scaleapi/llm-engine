from typing import Any, Optional

from pydantic import BaseModel, Field


class TraceConfig(BaseModel):
    """
    Schema for the data encoded in the x-sgp-trace-config header.
    """
    account_id: str = Field(None, description="SGP account id")
    user_id: str = Field(None, description="SGP user id of user who created the trace")
    group_id: Optional[str] = Field(None, description="Identifier for the group of spans")
    trace_id: str = Field(..., description="Unique identifier for the trace")
    parent_span_id: Optional[str] = Field(None, description="Identifier of the parent span")
    sgp_base_url: Optional[str] = Field(None, description="Base URL for SGP private API")
    default_metadata: Optional[dict[str, Any]] = Field(
        None, description="Additional metadata for the trace"
    )