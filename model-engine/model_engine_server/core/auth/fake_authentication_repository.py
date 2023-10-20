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

    def get_auth_from_username(self, username: str) -> Optional[User]:
        team_id = self.user_team_override.get(username, username)
        return User(user_id=username, team_id=team_id, is_privileged_user=True)

    async def get_auth_from_username_async(self, username: str) -> Optional[User]:
        team_id = self.user_team_override.get(username, username)
        return User(user_id=username, team_id=team_id, is_privileged_user=True)
