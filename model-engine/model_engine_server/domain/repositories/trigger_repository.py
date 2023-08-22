from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Sequence

from model_engine_server.domain.entities.trigger_entity import Trigger


class TriggerRepository(ABC):
    @abstractmethod
    async def create_trigger(
        self,
        *,
        name: str,
        created_by: str,
        owner: str,
        cron_schedule: str,
        docker_image_batch_job_bundle_id: str,
        default_job_config: Optional[Dict[str, Any]],
        default_job_metadata: Optional[Dict[str, str]],
    ) -> Trigger:
        """
        Creates a trigger.
        Args:
            name: User-set name of trigger
            created_by: User creating trigger
            owner: Team owning trigger
            cron_schedule: Schedule of k8s CronJob
            docker_image_batch_job_bundle_id: ID of docker image batch job bundle used by trigger
            default_job_config: Optional config to specify parameters injected at runtime
            default_job_metadata: Optional metdata tags for k8s jobs spawned by trigger

        Returns:
            A trigger entity
        """
        pass

    @abstractmethod
    async def list_triggers(
        self,
        owner: str,
    ) -> Sequence[Trigger]:
        """
        Lists all triggers with a given owner
        Args:
            owner: Owner of trigger(s)

        Returns:
            Sequence of trigger entities
        """
        pass

    @abstractmethod
    async def get_trigger(
        self,
        trigger_id: str,
    ) -> Optional[Trigger]:
        """
        Retrieves a single trigger by ID
        Args:
            trigger_id: ID of trigger we want

        Returns:
            Associated trigger entity or None if we couldn't find it
        """
        pass

    @abstractmethod
    async def update_trigger(
        self,
        trigger_id: str,
        cron_schedule: str,
    ) -> bool:
        """
        Updates the specified trigger's cron schedule
        Args:
            trigger_id: ID of trigger we want
            cron_schedule: new cron schedule to replace the original

        Returns:
            True or False, whether the update of the trigger was successful or not
        """
        pass

    @abstractmethod
    async def delete_trigger(
        self,
        trigger_id: str,
    ) -> bool:
        """
        Deletes the specified trigger
        Args:
            trigger_id: ID of trigger we want to delete

        Returns:
            True or False, whether the deletion of the trigger was successful or not
        """
        pass
