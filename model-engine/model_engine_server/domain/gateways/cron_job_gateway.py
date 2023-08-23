from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from model_engine_server.domain.entities.batch_job_entity import DockerImageBatchJob


class CronJobGateway(ABC):
    """
    Base class for K8s CronJob Gateway
    """

    @abstractmethod
    async def create_cronjob(
        self,
        *,
        request_host: str,
        trigger_id: str,
        created_by: str,
        owner: str,
        cron_schedule: str,
        docker_image_batch_job_bundle_id: str,
        default_job_config: Optional[Dict[str, Any]],
        default_job_metadata: Dict[str, str],
    ) -> None:
        """
        Create a cron job from a bundle and trigger.

        Args:
            request_host: URL to forward the batch job creation request
            trigger_id: The ID of the trigger
            created_by: The user who created the trigger
            owner: The user who owns the trigger
            cron_schedule: Cron-formatted string representing the cron job's invocation schedule
            docker_image_batch_job_bundle_id: The ID of the docker image batch job bundle
            default_job_config: The user-specified input to the batch job. Exposed as a file mounted at mount_location to the batch job
            job_config: K8s team/product labels
            resource_requests: The resource requests for the batch job

        Returns:
            None
        """
        pass

    @abstractmethod
    async def list_jobs(
        self,
        *,
        owner: str,
        trigger_id: Optional[str],
    ) -> List[DockerImageBatchJob]:
        """
        Lists all docker image batch jobs spawned by the trigger with the given ID, otherwise by owner if trigger_id is None

        Args:
            trigger_id: the ID of the trigger pointing to the cron job

        Returns:
            List of docker image batch jobs spawned by the trigger with the given ID, otherwise by owner if trigger_id is None
        """
        pass

    @abstractmethod
    async def update_cronjob(
        self,
        *,
        trigger_id: str,
        cron_schedule: Optional[str],
        suspend: Optional[bool],
    ) -> None:
        """
        Partially updates the schedule field and/or the suspend field of the specified cron job.

        Args:
            trigger_id: the ID of the trigger pointing to the cron job
            cron_schedule: New cron schedule parameter representing the cron job's invocation schedule
            suspend: The active status of the trigger, False means paused and True means unpaused

        Returns:
            None
        """
        pass

    @abstractmethod
    async def delete_cronjob(
        self,
        *,
        trigger_id: str,
    ) -> None:
        """
        Deletes the specified cron job.

        Args:
            trigger_id: the ID of the trigger pointing to the cron job

        Returns:
            None
        """
        pass
