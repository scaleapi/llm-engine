import pytest
from model_engine_server.infra.repositories.onprem_docker_repository import OnPremDockerRepository


@pytest.fixture
def onprem_docker_repo():
    return OnPremDockerRepository()


def test_image_exists_with_repository(onprem_docker_repo):
    result = onprem_docker_repo.image_exists(
        image_tag="v1.0.0",
        repository_name="my-registry/my-image",
    )
    assert result is True


def test_image_exists_without_repository(onprem_docker_repo):
    result = onprem_docker_repo.image_exists(
        image_tag="my-image:v1.0.0",
        repository_name="",
    )
    assert result is True


def test_image_exists_with_aws_profile(onprem_docker_repo):
    result = onprem_docker_repo.image_exists(
        image_tag="v1.0.0",
        repository_name="my-registry/my-image",
        aws_profile="some-profile",
    )
    assert result is True


def test_get_image_url_with_repository(onprem_docker_repo):
    result = onprem_docker_repo.get_image_url(
        image_tag="v1.0.0",
        repository_name="my-registry/my-image",
    )
    assert result == "my-registry/my-image:v1.0.0"


def test_get_image_url_without_repository(onprem_docker_repo):
    result = onprem_docker_repo.get_image_url(
        image_tag="my-full-image:v1.0.0",
        repository_name="",
    )
    assert result == "my-full-image:v1.0.0"


def test_build_image_raises_not_implemented(onprem_docker_repo):
    with pytest.raises(NotImplementedError) as exc_info:
        onprem_docker_repo.build_image(None)
    assert "does not support building images" in str(exc_info.value)


def test_get_latest_image_tag_raises_not_implemented(onprem_docker_repo):
    with pytest.raises(NotImplementedError) as exc_info:
        onprem_docker_repo.get_latest_image_tag("my-repo")
    assert "does not support querying latest image tags" in str(exc_info.value)

