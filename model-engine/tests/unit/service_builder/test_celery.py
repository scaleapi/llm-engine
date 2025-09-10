from unittest.mock import MagicMock, patch


class TestServiceBuilderCelery:
    """Test the service builder celery configuration logic"""

    def test_broker_type_redis_when_circleci(self):
        """Test that Redis broker is selected when CIRCLECI is True"""
        with (
            patch("model_engine_server.service_builder.celery.CIRCLECI", True),
            patch("model_engine_server.service_builder.celery.infra_config") as mock_config,
            patch("model_engine_server.service_builder.celery.BrokerType") as mock_broker_type,
        ):

            # Setup mocks
            mock_config_instance = MagicMock()
            mock_config_instance.celery_broker_type_redis = False
            mock_config.return_value = mock_config_instance

            mock_broker_type.REDIS.value = "redis"

            # Import the module to test (this will execute the logic)
            # Force re-execution by reloading
            import importlib

            import model_engine_server.service_builder.celery as celery_module

            importlib.reload(celery_module)

            # Verify Redis broker is selected
            assert celery_module.service_builder_broker_type == "redis"

    def test_broker_type_redis_when_config_enabled(self):
        """Test that Redis broker is selected when celery_broker_type_redis is True"""
        with (
            patch("model_engine_server.service_builder.celery.CIRCLECI", False),
            patch("model_engine_server.service_builder.celery.infra_config") as mock_config,
            patch("model_engine_server.service_builder.celery.BrokerType") as mock_broker_type,
        ):

            # Setup mocks
            mock_config_instance = MagicMock()
            mock_config_instance.celery_broker_type_redis = True
            mock_config.return_value = mock_config_instance

            mock_broker_type.REDIS.value = "redis"

            # Import and reload to test the logic
            import importlib

            import model_engine_server.service_builder.celery as celery_module

            importlib.reload(celery_module)

            # Verify Redis broker is selected
            assert celery_module.service_builder_broker_type == "redis"

    def test_broker_type_servicebus_for_azure(self):
        """Test that ServiceBus broker is selected for Azure"""
        with (
            patch("model_engine_server.service_builder.celery.CIRCLECI", False),
            patch("model_engine_server.service_builder.celery.infra_config") as mock_config,
            patch("model_engine_server.service_builder.celery.BrokerType") as mock_broker_type,
        ):

            # Setup mocks
            mock_config_instance = MagicMock()
            mock_config_instance.celery_broker_type_redis = False
            mock_config_instance.cloud_provider = "azure"
            mock_config.return_value = mock_config_instance

            mock_broker_type.SERVICEBUS.value = "servicebus"

            # Import and reload to test the logic
            import importlib

            import model_engine_server.service_builder.celery as celery_module

            importlib.reload(celery_module)

            # Verify ServiceBus broker is selected
            assert celery_module.service_builder_broker_type == "servicebus"

    def test_broker_type_sqs_default(self):
        """Test that SQS broker is selected as default"""
        with (
            patch("model_engine_server.service_builder.celery.CIRCLECI", False),
            patch("model_engine_server.service_builder.celery.infra_config") as mock_config,
            patch("model_engine_server.service_builder.celery.BrokerType") as mock_broker_type,
        ):

            # Setup mocks
            mock_config_instance = MagicMock()
            mock_config_instance.celery_broker_type_redis = False
            mock_config_instance.cloud_provider = "aws"
            mock_config.return_value = mock_config_instance

            mock_broker_type.SQS.value = "sqs"

            # Import and reload to test the logic
            import importlib

            import model_engine_server.service_builder.celery as celery_module

            importlib.reload(celery_module)

            # Verify SQS broker is selected as default
            assert celery_module.service_builder_broker_type == "sqs"
