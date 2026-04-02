from unittest import mock

import pytest
from google.api_core.exceptions import (
    GoogleAPICallError,
    NotFound,
    PermissionDenied,
    Unauthenticated,
)
from model_engine_server.domain.exceptions import DockerRepositoryNotFoundException
from model_engine_server.infra.repositories.gar_docker_repository import GARDockerRepository


@pytest.fixture
def gar_docker_repo():
    return GARDockerRepository()


@pytest.fixture
def mock_infra_config():
    with mock.patch(
        "model_engine_server.infra.repositories.gar_docker_repository.infra_config"
    ) as mock_config:
        mock_config.return_value.docker_repo_prefix = "us-east1-docker.pkg.dev/my-project/my-repo"
        yield mock_config


@pytest.fixture
def mock_ar_client():
    with mock.patch(
        "model_engine_server.infra.repositories.gar_docker_repository.artifactregistry_v1"
    ) as mock_ar:
        yield mock_ar.ArtifactRegistryClient.return_value


class TestImageExists:
    def test_returns_true_when_tag_found(self, gar_docker_repo, mock_infra_config, mock_ar_client):
        result = gar_docker_repo.image_exists("v1.0", "vllm")
        assert result is True
        mock_ar_client.get_tag.assert_called_once_with(
            name="projects/my-project/locations/us-east1/repositories/my-repo/packages/vllm/tags/v1.0"
        )

    def test_returns_false_when_not_found(self, gar_docker_repo, mock_infra_config, mock_ar_client):
        mock_ar_client.get_tag.side_effect = NotFound("not found")
        result = gar_docker_repo.image_exists("missing", "vllm")
        assert result is False

    def test_raises_on_permission_denied(self, gar_docker_repo, mock_infra_config, mock_ar_client):
        mock_ar_client.get_tag.side_effect = PermissionDenied("denied")
        with pytest.raises(PermissionDenied):
            gar_docker_repo.image_exists("v1.0", "vllm")

    def test_raises_on_unauthenticated(self, gar_docker_repo, mock_infra_config, mock_ar_client):
        mock_ar_client.get_tag.side_effect = Unauthenticated("unauth")
        with pytest.raises(Unauthenticated):
            gar_docker_repo.image_exists("v1.0", "vllm")

    def test_returns_false_on_other_api_error(
        self, gar_docker_repo, mock_infra_config, mock_ar_client
    ):
        mock_ar_client.get_tag.side_effect = GoogleAPICallError("server error")
        result = gar_docker_repo.image_exists("v1.0", "vllm")
        assert result is False


class TestGetImageUrl:
    def test_returns_full_url(self, gar_docker_repo, mock_infra_config):
        result = gar_docker_repo.get_image_url("v1.0", "vllm")
        assert result == "us-east1-docker.pkg.dev/my-project/my-repo/vllm:v1.0"


class TestBuildImage:
    def test_raises_not_implemented(self, gar_docker_repo):
        with pytest.raises(NotImplementedError):
            gar_docker_repo.build_image(None)


class TestGetLatestImageTag:
    def test_returns_last_tag(self, gar_docker_repo, mock_infra_config, mock_ar_client):
        tag1 = mock.Mock()
        tag1.name = "projects/p/locations/l/repositories/r/packages/vllm/tags/v1.0"
        tag2 = mock.Mock()
        tag2.name = "projects/p/locations/l/repositories/r/packages/vllm/tags/v2.0"
        mock_ar_client.list_tags.return_value = [tag1, tag2]

        result = gar_docker_repo.get_latest_image_tag("vllm")
        assert result == "v2.0"

    def test_raises_when_no_tags(self, gar_docker_repo, mock_infra_config, mock_ar_client):
        mock_ar_client.list_tags.return_value = []
        with pytest.raises(DockerRepositoryNotFoundException):
            gar_docker_repo.get_latest_image_tag("vllm")

    def test_raises_on_exception(self, gar_docker_repo, mock_infra_config, mock_ar_client):
        mock_ar_client.list_tags.side_effect = Exception("boom")
        with pytest.raises(DockerRepositoryNotFoundException):
            gar_docker_repo.get_latest_image_tag("vllm")
