from typing import Any, Optional

from pydantic import BaseModel


class EndpointPredictPayload(BaseModel):
    url: Optional[str] = None
    args: Optional[Any] = None
    cloudpickle: Optional[str] = None
    return_pickled: bool
