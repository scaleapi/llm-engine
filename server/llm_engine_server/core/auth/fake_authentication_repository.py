from typing import Dict, Optional

from llm_engine_server.core.auth.authentication_repository import AuthenticationRepository, User


class FakeAuthenticationRepository(AuthenticationRepository):
    def __init__(self, user_team_override: Optional[Dict[str, str]] = None):
        if user_team_override is None:
            user_team_override = {}
        self.user_team_override = user_team_override

    def get_auth_from_user_id(self, user_id: str) -> Optional[User]:
        team_id = self.user_team_override.get(user_id, user_id)
        return User(user_id=user_id, team_id=team_id, is_privileged_user=True)

    async def get_auth_from_user_id_async(self, user_id: str) -> Optional[User]:
        team_id = self.user_team_override.get(user_id, user_id)
        return User(user_id=user_id, team_id=team_id, is_privileged_user=True)

    def get_auth_from_api_key(self, api_key: str) -> Optional[User]:
        return User(user_id=api_key, team_id=api_key, is_privileged_user=True)

    async def get_auth_from_api_key_async(self, api_key: str) -> Optional[User]:
        return User(user_id=api_key, team_id=api_key, is_privileged_user=True)
