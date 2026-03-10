import re
from typing import Optional
from urllib.parse import urlencode

import requests
from model_engine_server.common.dtos.docker_repository import BuildImageRequest, BuildImageResponse
from model_engine_server.core.config import infra_config
from model_engine_server.core.loggers import logger_name, make_logger
from model_engine_server.domain.repositories import DockerRepository

logger = make_logger(logger_name())

_REQUEST_TIMEOUT = 10


def _parse_www_authenticate(header: str) -> Optional[dict]:
    """Parse a Www-Authenticate Bearer header into realm, service, and scope."""
    match = re.match(r'Bearer\s+(.*)', header, re.IGNORECASE)
    if not match:
        return None
    params = {}
    for m in re.finditer(r'(\w+)="([^"]*)"', match.group(1)):
        params[m.group(1)] = m.group(2)
    return params if "realm" in params else None


def _get_token(realm: str, service: Optional[str], scope: Optional[str]) -> Optional[str]:
    """Fetch a bearer token from the registry's token endpoint."""
    query = {}
    if service:
        query["service"] = service
    if scope:
        query["scope"] = scope
    url = f"{realm}?{urlencode(query)}" if query else realm
    try:
        resp = requests.get(url, timeout=_REQUEST_TIMEOUT)
        if resp.status_code == 200:
            data = resp.json()
            return data.get("token") or data.get("access_token")
    except (requests.RequestException, ValueError):
        pass
    return None


class GenericDockerRepository(DockerRepository):
    """Registry-agnostic Docker repository using the OCI Distribution / Docker Registry V2 HTTP API."""

    def image_exists(
        self, image_tag: str, repository_name: str, aws_profile: Optional[str] = None
    ) -> bool:
        prefix = infra_config().docker_repo_prefix.rstrip("/")
        parts = prefix.split("/", 1)
        registry_host = parts[0]
        path_prefix = parts[1] if len(parts) > 1 else ""
        full_repo = f"{path_prefix}/{repository_name}" if path_prefix else repository_name
        manifest_url = f"https://{registry_host}/v2/{full_repo}/manifests/{image_tag}"
        headers = {
            "Accept": ", ".join([
                "application/vnd.docker.distribution.manifest.v2+json",
                "application/vnd.oci.image.manifest.v1+json",
                "application/vnd.docker.distribution.manifest.list.v2+json",
                "application/vnd.oci.image.index.v1+json",
            ])
        }

        try:
            resp = requests.head(manifest_url, headers=headers, timeout=_REQUEST_TIMEOUT)

            if resp.status_code == 200:
                return True

            if resp.status_code == 401:
                www_auth = resp.headers.get("Www-Authenticate", "")
                auth_params = _parse_www_authenticate(www_auth)
                if auth_params:
                    token = _get_token(
                        realm=auth_params["realm"],
                        service=auth_params.get("service"),
                        scope=auth_params.get("scope"),
                    )
                    if token:
                        headers["Authorization"] = f"Bearer {token}"
                        resp = requests.head(
                            manifest_url, headers=headers, timeout=_REQUEST_TIMEOUT
                        )
                        return resp.status_code == 200

            return False
        except requests.RequestException as e:
            logger.warning(f"Failed to check image existence at {manifest_url}: {e}")
            return False

    def get_image_url(self, image_tag: str, repository_name: str) -> str:
        if self.is_repo_name(repository_name):
            return f"{infra_config().docker_repo_prefix}/{repository_name}:{image_tag}"
        return f"{repository_name}:{image_tag}"

    def build_image(self, image_params: BuildImageRequest) -> BuildImageResponse:
        raise NotImplementedError("GenericDockerRepository does not support building images")

    def get_latest_image_tag(self, repository_name: str) -> str:
        raise NotImplementedError("GenericDockerRepository does not support querying latest tags")
