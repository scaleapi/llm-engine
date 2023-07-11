from abc import ABC, abstractmethod
from typing import Optional


class FeatureFlagRepository(ABC):
    @abstractmethod
    async def write_feature_flag_bool(
        self,
        key: str,
        value: bool,
    ):
        """
        Writes the boolean feature flag info to a cache
        Args:
            key: Key to store the feature flag
            value: Value of the feature flag
        Returns:
            None
        """
        pass

    @abstractmethod
    async def read_feature_flag_bool(
        self,
        key: str,
    ) -> Optional[bool]:
        """
        Reads the boolean feature flag from the cache
        Args:
            key: The key of the feature flag
        Returns:
            The value of the feature flag if it's available in the cache
        """
        pass
