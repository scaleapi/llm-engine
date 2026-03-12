from unittest import mock

import pytest
import requests
from model_engine_server.infra.repositories.generic_docker_repository import (
    GenericDockerRepository,
    _parse_www_authenticate,
)


@pytest.fixture
def generic_docker_repo():
    return GenericDockerRepository()


@pytest.fixture
def mock_infra_config():
    with mock.patch(
        "model_engine_server.infra.repositories.generic_docker_repository.infra_config"
    ) as mock_config:
        mock_config.return_value.docker_repo_prefix = "public.ecr.aws/b2z8n5q1"
        yield mock_config


class TestParseWwwAuthenticate:
    def test_parses_bearer_header(self):
        header = 'Bearer realm="https://auth.example.com/token",service="registry.example.com",scope="repository:myrepo:pull"'
        result = _parse_www_authenticate(header)
        assert result == {
            "realm": "https://auth.example.com/token",
            "service": "registry.example.com",
            "scope": "repository:myrepo:pull",
        }

    def test_returns_none_for_basic_auth(self):
        assert _parse_www_authenticate('Basic realm="registry"') is None

    def test_returns_none_for_missing_realm(self):
        assert _parse_www_authenticate('Bearer service="foo"') is None

    def test_returns_none_for_empty_string(self):
        assert _parse_www_authenticate("") is None


class TestImageExists:
    def test_returns_true_on_200(self, generic_docker_repo, mock_infra_config):
        with mock.patch(
            "model_engine_server.infra.repositories.generic_docker_repository.requests"
        ) as mock_requests:
            mock_resp = mock.Mock()
            mock_resp.status_code = 200
            mock_requests.head.return_value = mock_resp

            result = generic_docker_repo.image_exists("v0.4.0", "model-engine/vllm")

            assert result is True
            mock_requests.head.assert_called_once()
            call_url = mock_requests.head.call_args[0][0]
            assert (
                call_url == "https://public.ecr.aws/v2/b2z8n5q1/model-engine/vllm/manifests/v0.4.0"
            )

    def test_returns_false_on_404(self, generic_docker_repo, mock_infra_config):
        with mock.patch(
            "model_engine_server.infra.repositories.generic_docker_repository.requests"
        ) as mock_requests:
            mock_resp = mock.Mock()
            mock_resp.status_code = 404
            mock_requests.head.return_value = mock_resp

            result = generic_docker_repo.image_exists("nonexistent", "vllm")

            assert result is False

    def test_returns_false_on_connection_error(self, generic_docker_repo, mock_infra_config):
        with mock.patch(
            "model_engine_server.infra.repositories.generic_docker_repository.requests"
        ) as mock_requests:
            mock_requests.head.side_effect = requests.ConnectionError("unreachable")
            mock_requests.ConnectionError = requests.ConnectionError
            mock_requests.RequestException = requests.RequestException

            result = generic_docker_repo.image_exists("v1.0", "vllm")

            assert result is False

    def test_token_auth_on_401(self, generic_docker_repo, mock_infra_config):
        with mock.patch(
            "model_engine_server.infra.repositories.generic_docker_repository.requests"
        ) as mock_requests:
            mock_requests.RequestException = requests.RequestException

            # First HEAD returns 401 with Www-Authenticate
            unauthed_resp = mock.Mock()
            unauthed_resp.status_code = 401
            unauthed_resp.headers = {
                "Www-Authenticate": 'Bearer realm="https://public.ecr.aws/token",service="public.ecr.aws",scope="repository:b2z8n5q1/vllm:pull"'
            }

            # Second HEAD (with token) returns 200
            authed_resp = mock.Mock()
            authed_resp.status_code = 200

            mock_requests.head.side_effect = [unauthed_resp, authed_resp]

            # Token endpoint returns a token
            token_resp = mock.Mock()
            token_resp.status_code = 200
            token_resp.json.return_value = {"token": "test-token-123"}
            mock_requests.get.return_value = token_resp

            result = generic_docker_repo.image_exists("v0.4.0", "vllm")

            assert result is True
            assert mock_requests.head.call_count == 2
            # Verify the second HEAD had the Authorization header
            second_call_headers = mock_requests.head.call_args_list[1][1]["headers"]
            assert second_call_headers["Authorization"] == "Bearer test-token-123"

    def test_returns_false_on_401_without_www_authenticate(
        self, generic_docker_repo, mock_infra_config
    ):
        with mock.patch(
            "model_engine_server.infra.repositories.generic_docker_repository.requests"
        ) as mock_requests:
            mock_requests.RequestException = requests.RequestException

            mock_resp = mock.Mock()
            mock_resp.status_code = 401
            mock_resp.headers = {}
            mock_requests.head.return_value = mock_resp

            result = generic_docker_repo.image_exists("v1.0", "vllm")

            assert result is False


class TestGetImageUrl:
    def test_prepends_prefix_for_simple_repo_name(self, generic_docker_repo, mock_infra_config):
        result = generic_docker_repo.get_image_url("v1.0", "vllm")
        assert result == "public.ecr.aws/b2z8n5q1/vllm:v1.0"

    def test_no_prefix_for_full_url(self, generic_docker_repo, mock_infra_config):
        result = generic_docker_repo.get_image_url("v1.0", "docker.io/library/nginx")
        assert result == "docker.io/library/nginx:v1.0"


class TestNotImplemented:
    def test_build_image_raises(self, generic_docker_repo):
        with pytest.raises(NotImplementedError):
            generic_docker_repo.build_image(None)

    def test_get_latest_image_tag_raises(self, generic_docker_repo):
        with pytest.raises(NotImplementedError):
            generic_docker_repo.get_latest_image_tag("vllm")
