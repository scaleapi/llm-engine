from typing import Optional

import requests
from model_engine_server.core.loggers import logger_name, make_logger
from model_engine_server.infra.services.endpoint_gc_service import DigestGateway

logger = make_logger(logger_name())

_SLACK_POST_MESSAGE_URL = "https://slack.com/api/chat.postMessage"
_SLACK_TEXT_LIMIT = 39000  # chat.postMessage rejects text over 40,000 characters.


class SlackDigestGateway(DigestGateway):
    """Posts the digest to a Slack channel with a bot token (chat.postMessage)."""

    def __init__(self, bot_token: str, channel: str):
        self.bot_token = bot_token
        self.channel = channel

    def send_digest(self, text: str) -> None:
        if len(text) > _SLACK_TEXT_LIMIT:
            text = text[:_SLACK_TEXT_LIMIT] + "\n... truncated, see pod logs for the full digest"
        try:
            response = requests.post(
                _SLACK_POST_MESSAGE_URL,
                headers={"Authorization": f"Bearer {self.bot_token}"},
                json={"channel": self.channel, "text": text},
                timeout=10,
            )
            body = response.json()
        except (requests.RequestException, ValueError):
            logger.exception("Failed to post GC digest to Slack")
            return
        if not body.get("ok"):
            logger.error(f"Slack rejected GC digest: {body.get('error')}")


class LogDigestGateway(DigestGateway):
    def send_digest(self, text: str) -> None:
        logger.info(f"Endpoint GC digest:\n{text}")


def build_digest_gateway(bot_token: Optional[str], channel: Optional[str]) -> DigestGateway:
    if bot_token and channel:
        return SlackDigestGateway(bot_token=bot_token, channel=channel)
    return LogDigestGateway()
