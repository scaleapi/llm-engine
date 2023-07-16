from abc import ABC, abstractmethod
from enum import Enum
from typing import List


class NotificationApp(str, Enum):
    SLACK = "slack"
    EMAIL = "email"


class NotificationGateway(ABC):
    """
    Abstract Base Class for interacting with a notification service.
    """

    @abstractmethod
    def send_notification(
        self,
        title: str,
        description: str,
        help_url: str,
        notification_apps: List[NotificationApp],
        users: List[str],
    ) -> bool:
        """
        Sends a notification to the given users.

        Args:
            title: The title of the notification.
            description: The description of the notification.
            help_url: The URL to the help page for the notification.
            notification_apps: The apps to send the notification to.
            users: The list of users to send the notification to.
        """
