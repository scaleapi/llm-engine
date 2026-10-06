import re
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, Iterator, List

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
AZURE_ARGS = [
    "--set",
    "azure.client_id=client-id",
    "--set",
    "azure.object_id=object-id",
    "--set",
    "azure.servicebus_namespace=servicebus",
]
SERVICE_ACCOUNT_TEMPLATES = [
    "templates/service_account.yaml",
    "templates/service_account_inference.yaml",
]
GENERATED_POD_TEMPLATES = {
    "deployment-runnable-image-sync-cpu.yaml",
    "leader-worker-set-streaming-gpu.yaml",
    "batch-job-orchestration-job.yaml",
    "docker-image-batch-job-gpu.yaml",
    "image-cache-cpu.yaml",
    "cron-trigger.yaml",
}

# The endpoint delegate fills ${...} placeholders at runtime. Some stand in for a whole mapping
# entry (e.g. ${STORAGE_DICT}), so one alone on its line becomes a key to keep the YAML valid.
_LINE_PLACEHOLDER = re.compile(r"(?m)^(\s*)\$\{[A-Z0-9_]+\}\s*$")
_PLACEHOLDER = re.compile(r"\$\{[A-Z0-9_]+\}")


def _render(
    templates: List[str], extra_args: List[str], base_args: List[str] = BASE_ARGS
) -> List[Dict[str, Any]]:
    if shutil.which("helm") is None:
        pytest.skip("helm is not installed")

    command = ["helm", "template", "test-release", str(CHART_PATH), "-f", str(VALUES_PATH)]
    for template in templates:
        command.extend(["--show-only", template])
    command.extend(base_args + extra_args)
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


def _find_pod_specs(node: Any) -> Iterator[Dict[str, Any]]:
    if isinstance(node, dict):
        if "containers" in node:
            yield node
        for value in node.values():
            yield from _find_pod_specs(value)
    elif isinstance(node, list):
        for value in node:
            yield from _find_pod_specs(value)


def _generated_pod_specs(extra_args: List[str]) -> List[Dict[str, Any]]:
    # Plain values_circleci.yaml runs endpoint pods as the existing "default" ServiceAccount, which
    # the chart doesn't manage, so these pods only get pull secrets set on the pod itself.
    (config_map,) = _render(
        ["templates/service_template_config_map.yaml"], extra_args, base_args=[]
    )
    pod_specs_by_template = {}
    for name, template in config_map["data"].items():
        template = _LINE_PLACEHOLDER.sub(r"\1placeholder: placeholder", template)
        pod_specs = list(_find_pod_specs(yaml.safe_load(_PLACEHOLDER.sub("placeholder", template))))
        if pod_specs:
            pod_specs_by_template[name] = pod_specs
    assert GENERATED_POD_TEMPLATES <= pod_specs_by_template.keys()
    return [pod_spec for pod_specs in pod_specs_by_template.values() for pod_spec in pod_specs]


def test_celery_autoscaler_renders_image_pull_secrets():
    pod_spec = _autoscaler_pod_spec(["--set", "imagePullSecrets[0].name=registry-cred"])

    assert pod_spec["imagePullSecrets"] == [{"name": "registry-cred"}]


def test_celery_autoscaler_omits_image_pull_secrets_when_unset():
    pod_spec = _autoscaler_pod_spec([])

    assert "imagePullSecrets" not in pod_spec


def test_service_accounts_render_image_pull_secrets():
    for service_account in _service_accounts(["--set", "imagePullSecrets[0].name=registry-cred"]):
        assert service_account["imagePullSecrets"] == [{"name": "registry-cred"}]


def test_generated_pods_render_image_pull_secrets():
    for pod_spec in _generated_pod_specs(["--set", "imagePullSecrets[0].name=registry-cred"]):
        assert pod_spec["imagePullSecrets"] == [{"name": "registry-cred"}]


# Valid secret names that YAML would read as an int or a bool if left unquoted.
@pytest.mark.parametrize("secret_name", ["123", "true"])
def test_image_pull_secret_names_stay_strings(secret_name: str):
    set_args = ["--set-string", f"imagePullSecrets[0].name={secret_name}"]

    assert _autoscaler_pod_spec(set_args)["imagePullSecrets"] == [{"name": secret_name}]
    for service_account in _service_accounts(set_args):
        assert service_account["imagePullSecrets"] == [{"name": secret_name}]
    for pod_spec in _generated_pod_specs(set_args):
        assert pod_spec["imagePullSecrets"] == [{"name": secret_name}]


def test_service_accounts_omit_image_pull_secrets_when_unset():
    for service_account in _service_accounts([]):
        assert "imagePullSecrets" not in service_account


# Azure alone must not add pod-level secrets: they would replace the ServiceAccount's.
@pytest.mark.parametrize("extra_args", [[], AZURE_ARGS], ids=["default", "azure"])
def test_generated_pods_omit_image_pull_secrets_when_unset(extra_args: List[str]):
    for pod_spec in _generated_pod_specs(extra_args):
        assert "imagePullSecrets" not in pod_spec


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


def test_azure_generated_pods_keep_regcred():
    for pod_spec in _generated_pod_specs(
        AZURE_ARGS + ["--set", "imagePullSecrets[0].name=registry-cred"]
    ):
        assert pod_spec["imagePullSecrets"] == [
            {"name": "egp-ecr-regcred"},
            {"name": "registry-cred"},
        ]
