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
        endpoint = f"https://{infra_config().docker_repo_prefix}"  # TODO: should this be HTTP since gcp uses GRPC
        credential = ...  # TODO!
        client = artifactregistry.ArtifactRegistryClient(
            client_options=ClientOptions(api_endpoint=endpoint), credentials=credential
        )  # TODO: should we use async?
        return client

    def image_exists(
        self, image_tag: str, repository_name: str, aws_profile: Optional[str] = None
    ) -> bool:
        client = self._get_client()

        try:
            # TODO: figure out the project_id and location
            client.get_docker_image(
                artifactregistry.GetDockerImageRequest(
                    name=f"projects/{infra_config().project_id}/locations/{infra_config().location}/repository/{repository_name}/dockerImages/{image_tag}"
                )
            )
        except NotFound:
            # TODO: check this is covered
            return False
        return True

    def get_image_url(
        self, image_tag: str, repository_name: str
    ) -> str:  # TODO: what should this look like for GCP? check ECR first
        return f"{infra_config().docker_repo_prefix}/{repository_name}:{image_tag}"

    def build_image(self, image_params: BuildImageRequest) -> BuildImageResponse:
        raise NotImplementedError("GCP image build not supported yet")
        # TODO: does this need to be implemented?

    def get_latest_image_tag(self, repository_name: str) -> str:
        client = self._get_client()
        parent = f"projects/{infra_config().project_id}/locations/{infra_config().location}/repository/{repository_name}"  # TODO: figure out the project_id and location
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
