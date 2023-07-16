from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional


@dataclass
class User:
    user_id: str
    team_id: str
    is_privileged_user: bool


class AuthenticationRepository(ABC):
    """
    Base class for a repository that returns authentication information to the application.
    With the context of the Model Primitive service, this just refers to a (user_id, team_id) pair.
    """

    @abstractmethod
    def get_auth_from_user_id(self, user_id: str) -> Optional[User]:
        """
        Returns authentication information associated with a given user_id.
        """

    @abstractmethod
    def get_auth_from_api_key(self, api_key: str) -> Optional[User]:
        """
        Returns authentication information associated with a given api_key.
        """

    @abstractmethod
    async def get_auth_from_user_id_async(self, user_id: str) -> Optional[User]:
        """
        Returns authentication information associated with a given user_id.
        """

    @abstractmethod
    async def get_auth_from_api_key_async(self, api_key: str) -> Optional[User]:
        """
        Returns authentication information associated with a given api_key.
        """
