from unittest.mock import Mock

import pytest
from datadog import statsd
from model_engine_server.common.dtos.llms import TokenUsage
from model_engine_server.core.auth.authentication_repository import User
from model_engine_server.domain.gateways.monitoring_metrics_gateway import MetricMetadata
from model_engine_server.infra.gateways import DatadogMonitoringMetricsGateway


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
def sync_token_count():
    return TokenUsage(
        num_prompt_tokens=100,
        num_completion_tokens=200,
        total_duration=30,
    )


@pytest.fixture
def streaming_token_count():
    return TokenUsage(
        num_prompt_tokens=100,
        num_completion_tokens=200,
        total_duration=30,
        time_to_first_token=5,
    )


@pytest.fixture
def datadog_monitoring_metrics_gateway():
    gateway = DatadogMonitoringMetricsGateway(prefix="model_engine_unit_test")
    return gateway


def test_datadog_monitoring_metrics_gateway_build_metrics(datadog_monitoring_metrics_gateway):
    datadog_monitoring_metrics_gateway.emit_attempted_build_metric()
    statsd.increment.assert_called_once()
    statsd.increment.reset_mock()
    datadog_monitoring_metrics_gateway.emit_successful_build_metric()
    statsd.increment.assert_called_once()
    statsd.increment.reset_mock()
    datadog_monitoring_metrics_gateway.emit_build_time_metric(300)
    statsd.distribution.assert_called_once()
    statsd.distribution.reset_mock()
    datadog_monitoring_metrics_gateway.emit_image_build_cache_hit_metric("test_image")
    statsd.increment.assert_called_once()
    statsd.increment.reset_mock()
    datadog_monitoring_metrics_gateway.emit_image_build_cache_miss_metric("test_image_2")
    statsd.increment.assert_called_once()
    statsd.increment.reset_mock()
    datadog_monitoring_metrics_gateway.emit_docker_failed_build_metric()
    statsd.increment.assert_called_once()
    statsd.increment.reset_mock()


def test_datadog_monitoring_metrics_gateway_db_metrics(datadog_monitoring_metrics_gateway):
    datadog_monitoring_metrics_gateway.emit_database_cache_hit_metric()
    statsd.increment.assert_called_once()
    statsd.increment.reset_mock()
    datadog_monitoring_metrics_gateway.emit_database_cache_miss_metric()
    statsd.increment.assert_called_once()
    statsd.increment.reset_mock()


def test_datadog_monitoring_metrics_gateway_route_call_metrics(datadog_monitoring_metrics_gateway):
    metadata = MetricMetadata(
        user=User(user_id="test_user", team_id="test_team", email="test_email"),
        model_name="test_model",
    )
    datadog_monitoring_metrics_gateway.emit_route_call_metric("test_route", metadata)
    statsd.increment.assert_called_once()
    statsd.increment.reset_mock()


def test_datadog_monitoring_metrics_gateway_token_count_metrics(
    datadog_monitoring_metrics_gateway, sync_token_count, streaming_token_count
):
    metadata = MetricMetadata(
        user=User(user_id="test_user", team_id="test_team", email="test_email"),
        model_name="test_model",
    )
    datadog_monitoring_metrics_gateway.emit_token_count_metrics(sync_token_count, metadata)
    statsd.increment.assert_called()
    statsd.increment.reset_mock()
    statsd.histogram.assert_called()
    statsd.histogram.reset_mock()
    datadog_monitoring_metrics_gateway.emit_token_count_metrics(streaming_token_count, metadata)
    statsd.increment.assert_called()
    statsd.increment.reset_mock()
    statsd.histogram.assert_called()
    statsd.histogram.reset_mock()
    statsd.distribution.assert_called()
    statsd.distribution.reset_mock()


def test_datadog_monitoring_metrics_gateway_batch_completion_metrics(
    datadog_monitoring_metrics_gateway,
):
    model = "test_model"
    use_tool = True
    num_prompt_tokens = 100
    num_completion_tokens = 200
    is_finetuned = True
    datadog_monitoring_metrics_gateway.emit_batch_completions_metric(
        model, use_tool, num_prompt_tokens, num_completion_tokens, is_finetuned
    )
    statsd.increment.assert_called()
    statsd.increment.reset_mock()
