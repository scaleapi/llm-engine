from abc import ABC, abstractmethod


class DigestGateway(ABC):
    """Delivers a human-readable run summary (for example to a Slack channel)."""

    @abstractmethod
    def send_digest(self, text: str) -> None:
        pass
