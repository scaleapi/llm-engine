from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Sequence

from llm_engine_server.common.dtos.model_bundles import ModelBundleOrderBy
from llm_engine_server.domain.entities import GpuType
from llm_engine_server.domain.entities.docker_image_batch_job_bundle_entity import (
    DockerImageBatchJobBundle,
)


class DockerImageBatchJobBundleRepository(ABC):
    @abstractmethod
    async def create_docker_image_batch_job_bundle(
        self,
        *,
        name: str,
        created_by: str,
        owner: str,
        image_repository: str,
        image_tag: str,
        command: List[str],
        env: Dict[str, str],
        mount_location: Optional[str],
        cpus: Optional[str],
        memory: Optional[str],
        storage: Optional[str],
        gpus: Optional[int],
        gpu_type: Optional[GpuType],
    ) -> DockerImageBatchJobBundle:
        """
        Creates a batch bundle.
        Args:
            name: User-set name of bundle
            created_by: User creating bundle
            owner: Team owning bundle
            image_repository: Docker Image repo (short), e.g. "hostedinference"
            image_tag: Tag of docker image
            command: Command to run inside of the batch bundle, e.g. `python script.py --arg1`
            env: List of env vars for the batch bundle
            mount_location: Optional, location of runtime-specifiable config that gets mounted on the filesystem
            cpus: Optional default # cpus for the underlying k8s job
            memory: Optional default amount of memory for the underlying k8s job
            storage: Optional default amount of storage for the underlying k8s job
            gpus: Optional default number of gpus
            gpu_type: Optional default gpu_type

        Returns:
            A batch bundle entity
        """
        pass

    @abstractmethod
    async def list_docker_image_batch_job_bundles(
        self, owner: str, name: Optional[str], order_by: Optional[ModelBundleOrderBy]
    ) -> Sequence[DockerImageBatchJobBundle]:
        """
        Lists all batch bundles with a given owner and name
        Args:
            owner: Owner of batch bundle(s)
            name: Name of batch bundle(s), if not specified we'll select all of the bundles
            order_by: Ordering to output bundle versions

        Returns:
            Sequence of batch bundle entities
        """
        pass

    @abstractmethod
    async def get_docker_image_batch_job_bundle(
        self, docker_image_batch_job_bundle_id: str
    ) -> Optional[DockerImageBatchJobBundle]:
        """
        Retrieves a single batch bundle by ID
        Args:
            docker_image_batch_job_bundle_id: Id of bundle we want

        Returns:
            Associated bundle entity or None if we couldn't find it
        """
        pass

    @abstractmethod
    async def get_latest_docker_image_batch_job_bundle(
        self, owner: str, name: str
    ) -> Optional[DockerImageBatchJobBundle]:
        """
        Retrieves the latest batch bundle by owner and name.
        Args:
            owner: Owner's name
            name: Bundle's name

        Returns:
            Associated bundle entity, or None if not found.
        """
        pass
