import pydantic

if int(pydantic.__version__.split(".")[0]) > 1:
    from pydantic.v1 import BaseModel  # pragma: no cover
else:
    from pydantic import BaseModel


class OwnedEntity(BaseModel):
    """
    This is the base class for entities that can be owned and need to be authorized.
    """

    created_by: str
    owner: str
