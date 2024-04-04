from unittest.mock import Mock

import pytest
from datadog import statsd
from model_engine_server.inference.infra.gateways.datadog_inference_monitoring_metrics_gateway import (
    DatadogInferenceMonitoringMetricsGateway,
)


@pytest.fixture(autouse=True)
def mock_statsd():
    # https://github.com/DataDog/datadogpy/issues/183 for how dd mocks statsd
    statsd.socket = Mock()
    # also mock the methods we use or may use, there might be more
    statsd.gauge = Mock()
    statsd.increment = Mock()
    statsd.decrement = Mock()
    statsd.histogram = Mock()
    statsd.distribution = Mock()


@pytest.fixture
def datadog_inference_monitoring_metrics_gateway():
    return DatadogInferenceMonitoringMetricsGateway()


def test_datadog_inference_monitoring_metrics_gateway_batch_completion_metrics(
    datadog_inference_monitoring_metrics_gateway,
):
    model = "test_model"
    use_tool = True
    num_prompt_tokens = 100
    num_completion_tokens = 200
    is_finetuned = True
    datadog_inference_monitoring_metrics_gateway.emit_batch_completions_metric(
        model, use_tool, num_prompt_tokens, num_completion_tokens, is_finetuned
    )
    statsd.increment.assert_called()
    statsd.increment.reset_mock()
