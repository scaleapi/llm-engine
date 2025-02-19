from typing import Optional

from google.api_core.client_options import ClientOptions
from google.api_core.exceptions import NotFound
from google.cloud import artifactregistry_v1 as artifactregistry
from model_engine_server.common.dtos.docker_repository import BuildImageRequest, BuildImageResponse
from model_engine_server.core.config import infra_config
from model_engine_server.core.loggers import logger_name, make_logger
from model_engine_server.domain.exceptions import DockerRepositoryNotFoundException
from model_engine_server.domain.repositories import DockerRepository

logger = make_logger(logger_name())


class GCPArtifactRegistryDockerRepository(DockerRepository):
    def _get_client(self):
        client = artifactregistry.ArtifactRegistryClient(
            client_options=ClientOptions()
            # NOTE: uses default auth credentials for GCP. Read `google.auth.default` function for more details
        )
        return client

    def _get_repository_prefix(self) -> str:
        # GCP is verbose and so has a long prefix for the repository
        return f"projects/{infra_config().ml_account_id}/locations/{infra_config().default_region}/repositories"

    def image_exists(
        self, image_tag: str, repository_name: str, aws_profile: Optional[str] = None
    ) -> bool:
        client = self._get_client()

        try:
            client.get_docker_image(
                artifactregistry.GetDockerImageRequest(
                    # This is the google cloud naming convention: https://cloud.google.com/artifact-registry/docs/docker/names
                    name=f"{self._get_repository_prefix()}/{repository_name}/dockerImages/{image_tag}"
                )
            )
        except NotFound:
            return False
        return True

    def get_image_url(self, image_tag: str, repository_name: str) -> str:
        return f"{infra_config().docker_repo_prefix}/{repository_name}:{image_tag}"

    def build_image(self, image_params: BuildImageRequest) -> BuildImageResponse:
        raise NotImplementedError("GCP image build not supported yet")

    def get_latest_image_tag(self, repository_name: str) -> str:
        client = self._get_client()
        parent = f"{self._get_repository_prefix()}/{repository_name}"
        try:
            images_pager = client.list_docker_images(
                artifactregistry.ListDockerImagesRequest(
                    parent=parent,
                    order_by="update_time_desc",  # NOTE: we expect that the artifact registry is immutable, so there should not be any updates after upload
                    page_size=1,
                )
            )

            docker_image_page = next(images_pager.pages, None)
            if docker_image_page is None:
                raise DockerRepositoryNotFoundException
            if (
                len(docker_image_page.docker_images) == 0
            ):  # This condition shouldn't happen since we're asking for 1 image per page
                raise DockerRepositoryNotFoundException
            return docker_image_page.docker_images[
                0
            ].name  # TODO: is the return as expected? it's a big string, not just the tag
        except NotFound:
            raise DockerRepositoryNotFoundException
