from typing import Optional

from model_engine_server.common.dtos.docker_repository import BuildImageRequest, BuildImageResponse
from model_engine_server.core.config import infra_config
from model_engine_server.core.loggers import logger_name, make_logger
from model_engine_server.domain.repositories import DockerRepository

logger = make_logger(logger_name())


class OnPremDockerRepository(DockerRepository):
    def image_exists(
        self, image_tag: str, repository_name: str, aws_profile: Optional[str] = None
    ) -> bool:
        if not repository_name:
            logger.warning(
                f"Direct image reference: {image_tag}, assuming exists. "
                f"Image validation skipped for on-prem deployments."
            )
            return True

        logger.warning(
            f"Registry image: {repository_name}:{image_tag}, assuming exists. "
            f"Image validation skipped for on-prem deployments. "
            f"Deployment will fail if image does not exist in registry."
        )
        return True

    def get_image_url(self, image_tag: str, repository_name: str) -> str:
        if not repository_name:
            logger.debug(f"Using direct image reference: {image_tag}")
            return image_tag

        image_url = f"{infra_config().docker_repo_prefix}/{repository_name}:{image_tag}"
        logger.debug(f"Constructed image URL: {image_url}")
        return image_url

    def build_image(self, image_params: BuildImageRequest) -> BuildImageResponse:
        raise NotImplementedError(
            "OnPremDockerRepository does not support building images. "
            "Images should be built via CI/CD and pushed to the on-prem registry."
        )

    def get_latest_image_tag(self, repository_name: str) -> str:
        raise NotImplementedError(
            "OnPremDockerRepository does not support querying latest image tags. "
            "Please specify explicit image tags in your deployment configuration."
        )
