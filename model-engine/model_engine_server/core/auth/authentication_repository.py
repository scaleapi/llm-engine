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

    @staticmethod
    @abstractmethod
    def is_allowed_team(team: str) -> bool:
        """
        Returns whether the provided team is an allowed team.
        """

    @abstractmethod
    def get_auth_from_username(self, username: str) -> Optional[User]:
        """
        Returns authentication information associated with a given Basic Auth username.
        """

    @abstractmethod
    async def get_auth_from_username_async(self, username: str) -> Optional[User]:
        """
        Returns authentication information associated with a given Basic Auth username.
        """
