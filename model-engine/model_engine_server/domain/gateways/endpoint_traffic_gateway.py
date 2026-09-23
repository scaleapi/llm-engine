from abc import ABC, abstractmethod
from datetime import datetime
from enum import Enum
from typing import Dict, Optional, Set


class TrafficKey(str, Enum):
    """How a traffic source names an endpoint."""

    DEPLOYMENT_NAME = "deployment_name"  # k8s workload name, e.g. launch-endpoint-id-end-...
    ENDPOINT_NAME = "endpoint_name"  # the model endpoint's name (DD_SERVICE on its pods)


class EndpointTrafficGateway(ABC):
    """Reports which endpoints received requests. None means the source could not answer."""

    key: TrafficKey

    @abstractmethod
    async def active_keys(self, since: datetime) -> Optional[Set[str]]:
        """Keys of endpoints with at least one request since ``since``."""

    async def last_active_at(self, since: datetime) -> Optional[Dict[str, datetime]]:
        """Most recent request time per key since ``since``; None when history is unavailable."""
        return None
