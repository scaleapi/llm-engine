from pydantic.v1 import BaseModel


class OwnedEntity(BaseModel):
    """
    This is the base class for entities that can be owned and need to be authorized.
    """

    created_by: str
    owner: str
