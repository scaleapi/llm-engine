from typing import Optional

from google.api_core.exceptions import GoogleAPICallError, NotFound, PermissionDenied, Unauthenticated
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
        # GAR resource names use sha256 digests, not tags. Use the tags API
        # to check if a tag exists for the given image.
        parent = (
            f"projects/{project}/locations/{location}"
            f"/repositories/{repository}/packages/{repository_name}"
        )
        tag_name = f"{parent}/tags/{image_tag}"
        try:
            client.get_tag(name=tag_name)
            return True
        except NotFound:
            return False
        except (PermissionDenied, Unauthenticated):
            raise
        except GoogleAPICallError as e:
            logger.warning(f"GAR API error checking tag {tag_name} ({type(e).__name__}), assuming image does not exist")
            return False

    def get_image_url(self, image_tag: str, repository_name: str) -> str:
        return f"{infra_config().docker_repo_prefix}/{repository_name}:{image_tag}"

    def build_image(self, image_params: BuildImageRequest) -> BuildImageResponse:
        raise NotImplementedError("GCP Artifact Registry image build not supported yet")

    def get_latest_image_tag(self, repository_name: str) -> str:
        client = artifactregistry_v1.ArtifactRegistryClient()
        location, project, repository = _parse_ar_prefix()
        # In AR, each Docker image name is a "package"; scope to it for server-side filtering
        parent = (
            f"projects/{project}/locations/{location}"
            f"/repositories/{repository}/packages/{repository_name}"
        )
        try:
            tags = list(client.list_tags(parent=parent))
            if not tags:
                raise DockerRepositoryNotFoundException
            return tags[-1].name.rsplit("/tags/", 1)[-1]
        except DockerRepositoryNotFoundException:
            raise
        except Exception:
            raise DockerRepositoryNotFoundException
