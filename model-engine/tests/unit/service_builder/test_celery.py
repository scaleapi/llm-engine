import pytest


def test_module_imports():
    """Test that the celery module can be imported"""
    from model_engine_server.service_builder.celery import (
        service_builder_broker_type,
        service_builder_service,
    )

    assert service_builder_broker_type in ["redis", "servicebus", "sqs"]
    assert service_builder_service is not None


def test_broker_type_is_string():
    """Test that broker type is set to a valid string"""
    from model_engine_server.service_builder.celery import service_builder_broker_type

    valid_types = ["redis", "servicebus", "sqs"]
    assert service_builder_broker_type in valid_types
    assert isinstance(service_builder_broker_type, str)


@pytest.mark.parametrize(
    "cloud_provider,is_ci,force_redis,expected",
    [
        ("aws", False, False, "sqs"),
        ("azure", False, False, "servicebus"),
        ("gcp", False, False, "redis"),
        ("aws", True, False, "redis"),
        ("azure", True, False, "redis"),
        ("gcp", True, False, "redis"),
        ("aws", False, True, "redis"),
        ("azure", False, True, "redis"),
        ("gcp", False, True, "redis"),
    ],
)
def test_get_broker_type(cloud_provider, is_ci, force_redis, expected):
    from model_engine_server.service_builder.celery import get_broker_type

    assert get_broker_type(cloud_provider, is_ci, force_redis) == expected
