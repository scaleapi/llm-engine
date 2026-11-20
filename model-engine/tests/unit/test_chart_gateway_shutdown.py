import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, List

import pytest
import yaml

CHART_PATH = Path(__file__).resolve().parents[3] / "charts" / "model-engine"
VALUES_PATH = CHART_PATH / "values_circleci.yaml"


def _render(*extra_args: str) -> Dict[str, Any]:
    if shutil.which("helm") is None:
        pytest.skip("helm is not installed")

    command = [
        "helm",
        "template",
        "test-release",
        str(CHART_PATH),
        "-f",
        str(VALUES_PATH),
        "--show-only",
        "templates/gateway_deployment.yaml",
        *extra_args,
    ]
    rendered = subprocess.run(command, check=True, capture_output=True, text=True).stdout
    return yaml.safe_load(rendered)


def _shutdown_args(pre_stop_sleep_seconds: int, grace_period_seconds: int) -> List[str]:
    return [
        "--set",
        f"gateway.shutdown.preStopSleepSeconds={pre_stop_sleep_seconds}",
        "--set",
        f"gateway.shutdown.terminationGracePeriodSeconds={grace_period_seconds}",
    ]


def _gateway_container(deployment: Dict[str, Any]) -> Dict[str, Any]:
    return deployment["spec"]["template"]["spec"]["containers"][0]


def _proxy_metadata(deployment: Dict[str, Any]) -> Dict[str, str]:
    annotations = deployment["spec"]["template"]["metadata"]["annotations"]
    return yaml.safe_load(annotations["proxy.istio.io/config"])["proxyMetadata"]


def test_gateway_holds_before_sigterm_and_outlives_gunicorn_drain() -> None:
    deployment = _render()

    # The app keeps serving while the endpoint removal propagates, with python because the
    # Chainguard runtime image has no sleep binary or shell.
    assert _gateway_container(deployment)["lifecycle"]["preStop"]["exec"]["command"] == [
        "python",
        "-c",
        "import time; time.sleep(15)",
    ]
    # Hold + Gunicorn's 60s graceful timeout + headroom.
    assert deployment["spec"]["template"]["spec"]["terminationGracePeriodSeconds"] == 90
    # The sidecar outlives the hold, then exits once its connections close.
    assert _proxy_metadata(deployment) == {
        "EXIT_ON_ZERO_ACTIVE_CONNECTIONS": "true",
        "MINIMUM_DRAIN_DURATION": "20s",
    }


def test_shutdown_timings_are_configurable() -> None:
    deployment = _render(*_shutdown_args(40, 110))

    command = _gateway_container(deployment)["lifecycle"]["preStop"]["exec"]["command"]
    assert command[-1] == "import time; time.sleep(40)"
    assert deployment["spec"]["template"]["spec"]["terminationGracePeriodSeconds"] == 110
    assert _proxy_metadata(deployment)["MINIMUM_DRAIN_DURATION"] == "45s"


def test_zero_pre_stop_sleep_removes_the_hook() -> None:
    deployment = _render(*_shutdown_args(0, 60))

    assert "lifecycle" not in _gateway_container(deployment)


def test_grace_period_must_cover_the_hold_and_gunicorn_drain() -> None:
    with pytest.raises(subprocess.CalledProcessError) as error:
        _render(*_shutdown_args(40, 90))
    assert "terminationGracePeriodSeconds must be at least preStopSleepSeconds + 60" in (
        error.value.stderr
    )
