from typing import Optional

from model_engine_server.common.dtos.docker_repository import BuildImageRequest, BuildImageResponse
from model_engine_server.core.config import infra_config
from model_engine_server.core.docker.ecr import image_exists as ecr_image_exists
from model_engine_server.core.docker.remote_build import build_remote_block
from model_engine_server.core.loggers import logger_name, make_logger
from model_engine_server.domain.repositories import DockerRepository

logger = make_logger(logger_name())


class ECRDockerRepository(DockerRepository):
    def image_exists(
        self, image_tag: str, repository_name: str, aws_profile: Optional[str] = None
    ) -> bool:
        logger.info(
            f"Called image_exists in ECRDockerRepository, returning {ecr_image_exists(image_tag=image_tag, repository_name=repository_name, aws_profile=aws_profile)}"
        )
        return ecr_image_exists(
            image_tag=image_tag,
            repository_name=repository_name,
            aws_profile=aws_profile,
        )

    def get_image_url(self, image_tag: str, repository_name: str) -> str:
        return f"{infra_config().docker_repo_prefix}/{repository_name}:{image_tag}"

    def build_image(self, image_params: BuildImageRequest) -> BuildImageResponse:
        logger.info(f"build_image args {locals()}")
        folders_to_include = ["model-engine"]
        if image_params.requirements_folder:
            folders_to_include.append(image_params.requirements_folder)

        dockerfile_root_folder = image_params.dockerfile.split("/")[0]
        if dockerfile_root_folder not in folders_to_include:
            folders_to_include.append(dockerfile_root_folder)

        build_args = {
            "BASE_IMAGE": image_params.base_image,
        }

        if image_params.substitution_args:
            build_args.update(image_params.substitution_args)

        build_result = build_remote_block(
            context=image_params.base_path,
            dockerfile=image_params.dockerfile,
            repotags=[f"{image_params.repo}:{image_params.image_tag}"],
            folders_to_include=folders_to_include,
            build_args=build_args,
        )
        return BuildImageResponse(status=build_result.status, logs=build_result.logs)
