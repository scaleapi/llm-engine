from typing import Any, Dict, Optional

from pydantic import BaseModel


# Note that this is a simplified version of the Span model, actual trace implementations don't necessarily extend
# this class.
class Span(BaseModel):
    trace_id: Optional[str] = None
    span_id: Optional[str] = None
    parent_span_id: Optional[str] = None
    group_id: Optional[str] = None
    input: Optional[Dict[str, Any]] = (None,)
    output: Optional[Dict[str, Any]] = (None,)
    metadata: Optional[Dict[str, Any]] = (None,)
