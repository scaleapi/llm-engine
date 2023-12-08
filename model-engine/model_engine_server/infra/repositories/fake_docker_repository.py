from typing import Optional

from model_engine_server.common.dtos.docker_repository import BuildImageRequest, BuildImageResponse
from model_engine_server.core.config import infra_config
from model_engine_server.core.loggers import logger_name, make_logger
from model_engine_server.domain.repositories import DockerRepository

logger = make_logger(logger_name())


class FakeDockerRepository(DockerRepository):
    def image_exists(
        self, image_tag: str, repository_name: str, aws_profile: Optional[str] = None
    ) -> bool:
        return True

    def get_image_url(self, image_tag: str, repository_name: str) -> str:
        return f"{infra_config().docker_repo_prefix}/{repository_name}:{image_tag}"

    def build_image(self, image_params: BuildImageRequest) -> BuildImageResponse:
        raise NotImplementedError("FakeDockerRepository build_image() not implemented")

    def get_latest_image_tag(self, repository_name: str) -> str:
        raise NotImplementedError("FakeDockerRepository get_latest_image_tag() not implemented")
