from datetime import datetime, timedelta, timezone
from typing import List

import pytest
from model_engine_server.infra.gateways.prometheus_endpoint_traffic_gateway import (
    PrometheusEndpointTrafficGateway,
)
from model_engine_server.infra.gateways.slack_digest_gateway import (
    _SLACK_TEXT_LIMIT,
    SlackDigestGateway,
)

SINCE = datetime.now(timezone.utc) - timedelta(hours=36)


@pytest.mark.parametrize(
    "coverage,activity,expected",
    [
        pytest.param([], [], None, id="no-series-at-all-is-unknown"),
        pytest.param([{"metric": {}, "value": [0, "0"]}], [], None, id="zero-series-is-unknown"),
        pytest.param([{"metric": {}, "value": [0, "12"]}], [], set(), id="covered-and-idle"),
        pytest.param(
            [{"metric": {}, "value": [0, "12"]}],
            [
                {"metric": {"destination_workload": "launch-endpoint-id-end-a"}, "value": [0, "3"]},
                {"metric": {"destination_workload": "launch-endpoint-id-end-b"}, "value": [0, "0"]},
            ],
            {"launch-endpoint-id-end-a"},
            id="active-workloads",
        ),
    ],
)
@pytest.mark.asyncio
async def test_prometheus_active_keys(coverage, activity, expected):
    gateway = PrometheusEndpointTrafficGateway("http://prom")
    answers = iter([coverage, activity])

    async def fake_query(query: str):
        return next(answers)

    gateway._query = fake_query  # type: ignore[method-assign]
    assert await gateway.active_keys(SINCE) == expected


@pytest.mark.asyncio
async def test_prometheus_query_failure_is_unknown():
    gateway = PrometheusEndpointTrafficGateway("http://prom")

    async def fake_query(query: str):
        return None

    gateway._query = fake_query  # type: ignore[method-assign]
    assert await gateway.active_keys(SINCE) is None


def test_slack_digest_is_split_and_every_part_must_deliver():
    posted: List[str] = []
    gateway = SlackDigestGateway("token", "#c")
    gateway._post = lambda text: posted.append(text) or True  # type: ignore[method-assign]
    lines = [f"line {i} " + "x" * 200 for i in range(300)]
    text = "\n".join(lines)

    assert gateway.send_digest(text) is True
    assert len(posted) > 1
    assert all(len(p) <= _SLACK_TEXT_LIMIT + 32 for p in posted)
    assert "\n".join(lines) in "\n".join(p.split("\n", 1)[1] for p in posted)

    gateway._post = lambda text: False  # type: ignore[method-assign]
    assert gateway.send_digest("short") is False


def _target(app: str, pod: str, health: str = "up", ready: str = "true", port: int = 15020):
    return {
        "health": health,
        "scrapeUrl": (
            f"http://10.0.0.1:{port}/stats/prometheus"
            if port in (15020, 15090)
            else f"http://10.0.0.1:{port}/metrics"
        ),
        "discoveredLabels": {
            "__meta_kubernetes_pod_label_app": app,
            "__meta_kubernetes_pod_name": pod,
            "__meta_kubernetes_pod_ready": ready,
        },
    }


A, B = "launch-endpoint-id-end-a", "launch-endpoint-id-end-b"


@pytest.mark.parametrize(
    "data,expected",
    [
        pytest.param(None, None, id="request-failed"),
        pytest.param({"activeTargets": []}, None, id="no-endpoint-sidecar-is-unknown"),
        pytest.param(
            {"activeTargets": [_target(A, "a-1"), _target("other", "o-1"), {"health": "up"}]},
            {A: 1},
            id="healthy-sidecar-pods-counted-per-deployment",
        ),
        pytest.param(
            {
                "activeTargets": [
                    _target(A, "a-1"),
                    _target(A, "a-1", port=15090),
                    _target(A, "a-2"),
                ]
            },
            {A: 2},
            id="pods-counted-once",
        ),
        pytest.param(
            {"activeTargets": [_target(A, "a-1"), _target(B, "b-1", health="down")]},
            {A: 1},
            id="sidecar-down-is-not-observed",
        ),
        pytest.param(
            {"activeTargets": [_target(A, "a-1"), _target(B, "b-1", port=5000)]},
            {A: 1},
            id="app-metrics-target-does-not-count",
        ),
        pytest.param(
            {"activeTargets": [_target(A, "a-1"), _target(A, "a-2", ready="false")]},
            {A: 1},
            id="not-ready-pod-not-counted",
        ),
    ],
)
@pytest.mark.asyncio
async def test_prometheus_observed_pod_counts(data, expected):
    gateway = PrometheusEndpointTrafficGateway("http://prom")

    async def fake_get(path: str, params: dict):
        assert path == "/api/v1/targets" and params == {"state": "active"}
        return data

    gateway._get = fake_get  # type: ignore[method-assign]
    assert await gateway.observed_pod_counts() == expected


@pytest.mark.asyncio
async def test_prometheus_never_vouches_for_history():
    gateway = PrometheusEndpointTrafficGateway("http://prom")

    async def fail(*args, **kwargs):  # pragma: no cover - must not be called
        raise AssertionError("no request expected")

    gateway._get = fail  # type: ignore[method-assign]
    assert await gateway.last_active_at(datetime.now(timezone.utc) - timedelta(days=180)) is None
