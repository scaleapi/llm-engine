import hashlib
from typing import Dict, Optional

from model_engine_server.core.auth.authentication_repository import AuthenticationRepository, User


class FakeAuthenticationRepository(AuthenticationRepository):
    def __init__(self, user_team_override: Optional[Dict[str, str]] = None):
        if user_team_override is None:
            user_team_override = {}
        self.user_team_override = user_team_override

    @staticmethod
    def is_allowed_team(team: str) -> bool:
        return True

    @staticmethod
    def _stable_id(username: str) -> str:
        """
        Model-engine DB schemas often store created_by/owner as VARCHAR(24) (mongo-style ids).
        When using fake auth and bearer tokens are long, we must map them into 24 chars to
        avoid DB insert failures while keeping determinism.
        """
        if len(username) <= 24:
            return username
        return hashlib.sha1(username.encode("utf-8")).hexdigest()[:24]

    def get_auth_from_username(self, username: str) -> Optional[User]:
        user_id = self._stable_id(username)
        team_id_raw = self.user_team_override.get(username, username)
        team_id = self._stable_id(team_id_raw)
        return User(user_id=user_id, team_id=team_id, is_privileged_user=True)

    async def get_auth_from_username_async(self, username: str) -> Optional[User]:
        user_id = self._stable_id(username)
        team_id_raw = self.user_team_override.get(username, username)
        team_id = self._stable_id(team_id_raw)
        return User(user_id=user_id, team_id=team_id, is_privileged_user=True)
