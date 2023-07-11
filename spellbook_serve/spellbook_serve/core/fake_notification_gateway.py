from collections import defaultdict
from typing import List

from spellbook_serve.core.notification_gateway import NotificationApp, NotificationGateway


class FakeNotificationGateway(NotificationGateway):
    def __init__(self, notifications_should_succeed: bool = True):
        self.notifications_should_succeed = notifications_should_succeed
        self.notifications_sent = defaultdict(list)

    def send_notification(
        self,
        title: str,
        description: str,
        help_url: str,
        notification_apps: List[NotificationApp],
        users: List[str],
    ) -> bool:
        if self.notifications_should_succeed:
            for app in notification_apps:
                notification = {
                    "title": title,
                    "description": description,
                    "help_url": help_url,
                    "users": users,
                }
                self.notifications_sent[app].append(notification)

        return self.notifications_should_succeed
