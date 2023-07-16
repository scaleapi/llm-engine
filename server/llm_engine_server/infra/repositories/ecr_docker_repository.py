from typing import Optional

from llm_engine_server.common.dtos.docker_repository import BuildImageRequest, BuildImageResponse
from llm_engine_server.core.config import ml_infra_config
from llm_engine_server.core.docker.ecr import image_exists as ecr_image_exists
from llm_engine_server.core.docker.remote_build import build_remote_block
from llm_engine_server.domain.repositories import DockerRepository


class ECRDockerRepository(DockerRepository):
    def image_exists(
        self, image_tag: str, repository_name: str, aws_profile: Optional[str] = None
    ) -> bool:
        return ecr_image_exists(
            image_tag=image_tag,
            repository_name=repository_name,
            aws_profile=aws_profile,
        )

    def get_image_url(self, image_tag: str, repository_name: str) -> str:
        return f"{ml_infra_config().docker_repo_prefix}/{repository_name}:{image_tag}"

    def build_image(self, image_params: BuildImageRequest) -> BuildImageResponse:
        folders_to_include = [
            "llm_engine",
        ]
        if image_params.requirements_folder:
            folders_to_include.append(image_params.requirements_folder)

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
