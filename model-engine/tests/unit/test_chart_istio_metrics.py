import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, List

import pytest
import yaml

CHART_PATH = Path(__file__).resolve().parents[3] / "charts" / "model-engine"
VALUES_PATH = CHART_PATH / "values_circleci.yaml"
RELEASE_NAMESPACE = "launch"


def _render() -> List[Dict[str, Any]]:
    if shutil.which("helm") is None:
        pytest.skip("helm is not installed")

    command = [
        "helm",
        "template",
        "test-release",
        str(CHART_PATH),
        "-f",
        str(VALUES_PATH),
        "--namespace",
        RELEASE_NAMESPACE,
        "--show-only",
        "templates/istio-metrics.yaml",
    ]
    rendered = subprocess.run(command, check=True, capture_output=True, text=True).stdout
    return [doc for doc in yaml.safe_load_all(rendered) if doc]


def test_request_operation_telemetry_is_scoped_to_attributegen_pods() -> None:
    docs = _render()
    telemetry = next(doc for doc in docs if doc["kind"] == "Telemetry")
    wasm_plugin = next(doc for doc in docs if doc["kind"] == "WasmPlugin")

    # A selector-less Telemetry in istio-system is mesh-wide, and Istio applies only the oldest one.
    assert telemetry["metadata"]["namespace"] == RELEASE_NAMESPACE
    selector = telemetry["spec"]["selector"]["matchLabels"]
    assert selector == wasm_plugin["spec"]["selector"]["matchLabels"]
