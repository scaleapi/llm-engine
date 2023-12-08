import re
from abc import ABC, abstractmethod
from typing import Optional

from model_engine_server.common.dtos.docker_repository import BuildImageRequest, BuildImageResponse


class DockerRepository(ABC):
    """
    Base class for Docker repositories.
    """

    @abstractmethod
    def image_exists(
        self, image_tag: str, repository_name: str, aws_profile: Optional[str] = None
    ) -> bool:
        """
        Returns whether a Docker image with the provided tag and repository name exists.

        Args:
            image_tag: the tag given to the Docker image.
            repository_name: the name of the repository containing the image.
            aws_profile: the aws profile to use for ECR.

        Returns: boolean of whether the image exists.
        """

    @abstractmethod
    def get_image_url(self, image_tag: str, repository_name: str) -> str:
        """
        Returns the image url given with the provided tag and repository name.

        Args:
            image_tag: the tag given to the Docker image.
            repository_name: the name of the repository containing the image.

        Returns: the image url.
        """

    @abstractmethod
    def build_image(self, image_params: BuildImageRequest) -> BuildImageResponse:
        """
        Builds a docker image.

        Args:
            image_params: Parameters to use for building the image.

        Returns: The status and logs of the image building.
        """
        pass

    @abstractmethod
    def get_latest_image_tag(self, repository_name: str) -> str:
        """
        Returns the Docker image tag of the most recently pushed image in the given repository

        Args:
            repository_name: the name of the repository containing the image.

        Returns: the tag of the latest Docker image.
        """

    def is_repo_name(self, repo_name: str):
        # We assume repository names must start with a letter and can only contain lowercase letters, numbers, hyphens, underscores, and forward slashes.
        # Based-off ECR naming standards
        # https://docs.aws.amazon.com/AmazonECR/latest/APIReference/API_CreateRepository.html#API_CreateRepository_RequestSyntax
        pattern = r"^[a-z0-9\-_/]*$"
        match = re.fullmatch(pattern, repo_name)
        return match is not None
