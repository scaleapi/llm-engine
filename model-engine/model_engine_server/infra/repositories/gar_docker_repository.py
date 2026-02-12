from typing import Optional

from google.api_core.exceptions import NotFound
from google.cloud import artifactregistry_v1
from model_engine_server.common.dtos.docker_repository import BuildImageRequest, BuildImageResponse
from model_engine_server.core.config import infra_config
from model_engine_server.core.loggers import logger_name, make_logger
from model_engine_server.domain.exceptions import DockerRepositoryNotFoundException
from model_engine_server.domain.repositories import DockerRepository

logger = make_logger(logger_name())


def _parse_ar_prefix():
    """Parse docker_repo_prefix (e.g. 'us-docker.pkg.dev/my-project/my-repo') into components.

    Returns:
        Tuple of (location, project, repository).
    """
    prefix = infra_config().docker_repo_prefix
    parts = prefix.split("/")
    location = parts[0].replace("-docker.pkg.dev", "")
    project = parts[1]
    repository = parts[2]
    return location, project, repository


class GARDockerRepository(DockerRepository):
    """Docker repository backed by Google Artifact Registry."""

    def image_exists(
        self, image_tag: str, repository_name: str, aws_profile: Optional[str] = None
    ) -> bool:
        client = artifactregistry_v1.ArtifactRegistryClient()
        location, project, repository = _parse_ar_prefix()
        name = (
            f"projects/{project}/locations/{location}/repositories/{repository}"
            f"/dockerImages/{repository_name}:{image_tag}"
        )
        try:
            client.get_docker_image(name=name)
            return True
        except NotFound:
            return False

    def get_image_url(self, image_tag: str, repository_name: str) -> str:
        return f"{infra_config().docker_repo_prefix}/{repository_name}:{image_tag}"

    def build_image(self, image_params: BuildImageRequest) -> BuildImageResponse:
        raise NotImplementedError("GCP Artifact Registry image build not supported yet")

    def get_latest_image_tag(self, repository_name: str) -> str:
        client = artifactregistry_v1.ArtifactRegistryClient()
        location, project, repository = _parse_ar_prefix()
        parent = f"projects/{project}/locations/{location}/repositories/{repository}"

        try:
            images = client.list_docker_images(parent=parent)
            matching = [
                img for img in images if f"dockerImages/{repository_name}" in img.name and img.tags
            ]
            if not matching:
                raise DockerRepositoryNotFoundException
            matching.sort(key=lambda img: img.update_time, reverse=True)
            return matching[0].tags[0]
        except DockerRepositoryNotFoundException:
            raise
        except Exception:
            raise DockerRepositoryNotFoundException
