import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, List

import pytest
import yaml

CHART_PATH = Path(__file__).resolve().parents[3] / "charts" / "model-engine"
VALUES_PATH = CHART_PATH / "values_circleci.yaml"

# values_circleci.yaml disables the autoscaler and the inference ServiceAccount; turn both on so
# every template that carries pull secrets is rendered.
BASE_ARGS = [
    "--set",
    "celery_autoscaler.enabled=true",
    "--set",
    "celery_autoscaler.num_shards=1",
    "--set",
    "serviceTemplate.createServiceAccount=true",
    "--set",
    "serviceTemplate.serviceAccountName=model-engine-inference",
    "--set",
    "serviceTemplate.serviceAccountAnnotations.example=annotation",
]
SERVICE_ACCOUNT_TEMPLATES = [
    "templates/service_account.yaml",
    "templates/service_account_inference.yaml",
]


def _render(templates: List[str], extra_args: List[str]) -> List[Dict[str, Any]]:
    if shutil.which("helm") is None:
        pytest.skip("helm is not installed")

    command = ["helm", "template", "test-release", str(CHART_PATH), "-f", str(VALUES_PATH)]
    for template in templates:
        command.extend(["--show-only", template])
    command.extend(BASE_ARGS + extra_args)
    rendered = subprocess.run(command, check=True, capture_output=True, text=True).stdout
    return [doc for doc in yaml.safe_load_all(rendered) if doc]


def _autoscaler_pod_spec(extra_args: List[str]) -> Dict[str, Any]:
    (stateful_set,) = _render(["templates/celery_autoscaler_stateful_set.yaml"], extra_args)
    return stateful_set["spec"]["template"]["spec"]


def _service_accounts(extra_args: List[str]) -> List[Dict[str, Any]]:
    service_accounts = _render(SERVICE_ACCOUNT_TEMPLATES, extra_args)
    assert {sa["metadata"]["name"] for sa in service_accounts} == {
        "model-engine",
        "model-engine-inference",
    }
    return service_accounts


def test_celery_autoscaler_renders_image_pull_secrets():
    pod_spec = _autoscaler_pod_spec(["--set", "imagePullSecrets[0].name=registry-cred"])

    assert pod_spec["imagePullSecrets"] == [{"name": "registry-cred"}]


def test_celery_autoscaler_omits_image_pull_secrets_when_unset():
    pod_spec = _autoscaler_pod_spec([])

    assert "imagePullSecrets" not in pod_spec


def test_service_accounts_render_image_pull_secrets():
    for service_account in _service_accounts(["--set", "imagePullSecrets[0].name=registry-cred"]):
        assert service_account["imagePullSecrets"] == [{"name": "registry-cred"}]


# Valid secret names that YAML would read as an int or a bool if left unquoted.
@pytest.mark.parametrize("secret_name", ["123", "true"])
def test_image_pull_secret_names_stay_strings(secret_name: str):
    set_args = ["--set-string", f"imagePullSecrets[0].name={secret_name}"]

    assert _autoscaler_pod_spec(set_args)["imagePullSecrets"] == [{"name": secret_name}]
    for service_account in _service_accounts(set_args):
        assert service_account["imagePullSecrets"] == [{"name": secret_name}]


def test_service_accounts_omit_image_pull_secrets_when_unset():
    for service_account in _service_accounts([]):
        assert "imagePullSecrets" not in service_account


def test_azure_service_accounts_keep_regcred_without_duplicating_it():
    for service_account in _service_accounts(
        [
            "--set",
            "azure.client_id=client-id",
            "--set",
            "imagePullSecrets[0].name=egp-ecr-regcred",
            "--set",
            "imagePullSecrets[1].name=registry-cred",
        ]
    ):
        assert service_account["imagePullSecrets"] == [
            {"name": "egp-ecr-regcred"},
            {"name": "registry-cred"},
        ]
