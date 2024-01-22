import os
from typing import Optional

from azure.containerregistry import ContainerRegistryClient
from azure.core.exceptions import ResourceNotFoundError
from azure.identity import ManagedIdentityCredential
from model_engine_server.common.dtos.docker_repository import BuildImageRequest, BuildImageResponse
from model_engine_server.core.config import infra_config
from model_engine_server.core.loggers import logger_name, make_logger
from model_engine_server.domain.repositories import DockerRepository

logger = make_logger(logger_name())


class ACRDockerRepository(DockerRepository):
    def image_exists(
        self, image_tag: str, repository_name: str, aws_profile: Optional[str] = None
    ) -> bool:
        endpoint = f"https://{infra_config().docker_repo_prefix}"
        credential = ManagedIdentityCredential(
            client_id=os.getenv("AZURE_KUBERNETES_CLUSTER_CLIENT_ID")
        )
        client = ContainerRegistryClient(endpoint, credential)

        try:
            client.get_manifest_properties(repository_name, image_tag)
        except ResourceNotFoundError:
            return False
        return True

    def get_image_url(self, image_tag: str, repository_name: str) -> str:
        return f"{infra_config().docker_repo_prefix}/{repository_name}:{image_tag}"

    def build_image(self, image_params: BuildImageRequest) -> BuildImageResponse:
        raise NotImplementedError("ACR image build not supported yet")

    def get_latest_image_tag(self, repository_name: str) -> str:
        endpoint = f"https://{infra_config().docker_repo_prefix}"
        credential = ManagedIdentityCredential(
            client_id=os.getenv("AZURE_KUBERNETES_CLUSTER_CLIENT_ID")
        )
        client = ContainerRegistryClient(endpoint, credential)

        image = client.list_manifest_properties(
            repository_name, order_by="time_desc", results_per_page=1
        ).next()
        return image.tags[0]
