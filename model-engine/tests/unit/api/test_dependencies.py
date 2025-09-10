from unittest.mock import MagicMock, patch

from model_engine_server.api.dependencies import get_external_interfaces


def test_redis_task_queue_selection_when_celery_broker_type_redis_enabled():
    """Test that redis task queue gateways are selected when celery_broker_type_redis is True"""
    with patch("model_engine_server.api.dependencies.infra_config") as mock_config:
        # Mock the configuration
        mock_config_instance = MagicMock()
        mock_config_instance.celery_broker_type_redis = True
        mock_config.return_value = mock_config_instance

        # Mock the gateway constructors
        with patch(
            "model_engine_server.api.dependencies.redis_task_queue_gateway"
        ) as mock_redis_gateway:
            # Call the function
            external_interfaces = get_external_interfaces()

            # Verify that redis gateways are used
            assert external_interfaces.inference_task_queue_gateway == mock_redis_gateway
            assert external_interfaces.infra_task_queue_gateway == mock_redis_gateway


def test_default_task_queue_selection_when_celery_broker_type_redis_disabled():
    """Test that default task queue gateways are selected when celery_broker_type_redis is False/None"""
    with patch("model_engine_server.api.dependencies.infra_config") as mock_config:
        # Mock the configuration
        mock_config_instance = MagicMock()
        mock_config_instance.celery_broker_type_redis = False  # or None
        mock_config.return_value = mock_config_instance

        # Mock the gateway constructors
        with patch(
            "model_engine_server.api.dependencies.servicebus_task_queue_gateway"
        ) as mock_servicebus_gateway:
            # Call the function
            external_interfaces = get_external_interfaces()

            # Verify that servicebus gateways are used
            assert external_interfaces.inference_task_queue_gateway == mock_servicebus_gateway
            assert external_interfaces.infra_task_queue_gateway == mock_servicebus_gateway
