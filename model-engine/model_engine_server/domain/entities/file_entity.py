from datetime import datetime

from pydantic import BaseModel


class FileMetadata(BaseModel):
    """
    This is the entity-layer class for a File from the Files API.
    """

    id: str
    filename: str
    size: int
    owner: str
    updated_at: datetime
