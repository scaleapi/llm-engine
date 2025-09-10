from unittest.mock import MagicMock, patch


class TestServiceBuilderCelery:
    """Test the service builder celery configuration logic"""

    @patch("model_engine_server.service_builder.celery.CIRCLECI", True)
    @patch("model_engine_server.service_builder.celery.infra_config")
    @patch("model_engine_server.service_builder.celery.celery_app")
    def test_broker_type_redis_when_circleci(self, mock_celery_app, mock_config, mock_circleci):
        """Test that Redis broker is selected when CIRCLECI is True"""
        # Setup mocks
        mock_config_instance = MagicMock()
        mock_config_instance.s3_bucket = "test-bucket"
        mock_config_instance.cloud_provider = "aws"
        mock_config.return_value = mock_config_instance

        # Force module reload to test the logic
        import importlib

        import model_engine_server.service_builder.celery as celery_module

        importlib.reload(celery_module)

        # Verify Redis broker is selected
        assert celery_module.service_builder_broker_type == "redis"

    @patch("model_engine_server.service_builder.celery.CIRCLECI", False)
    @patch("model_engine_server.service_builder.celery.infra_config")
    @patch("model_engine_server.service_builder.celery.celery_app")
    def test_broker_type_redis_when_config_enabled(
        self, mock_celery_app, mock_config, mock_circleci
    ):
        """Test that Redis broker is selected when celery_broker_type_redis is True"""
        # Setup mocks
        mock_config_instance = MagicMock()
        mock_config_instance.celery_broker_type_redis = True
        mock_config_instance.s3_bucket = "test-bucket"
        mock_config_instance.cloud_provider = "aws"
        mock_config.return_value = mock_config_instance

        # Force module reload to test the logic
        import importlib

        import model_engine_server.service_builder.celery as celery_module

        importlib.reload(celery_module)

        # Verify Redis broker is selected
        assert celery_module.service_builder_broker_type == "redis"

    @patch("model_engine_server.service_builder.celery.CIRCLECI", False)
    @patch("model_engine_server.service_builder.celery.infra_config")
    @patch("model_engine_server.service_builder.celery.celery_app")
    def test_broker_type_servicebus_for_azure(self, mock_celery_app, mock_config, mock_circleci):
        """Test that ServiceBus broker is selected for Azure"""
        # Setup mocks
        mock_config_instance = MagicMock()
        mock_config_instance.celery_broker_type_redis = False
        mock_config_instance.cloud_provider = "azure"
        mock_config_instance.s3_bucket = "test-bucket"
        mock_config.return_value = mock_config_instance

        # Force module reload to test the logic
        import importlib

        import model_engine_server.service_builder.celery as celery_module

        importlib.reload(celery_module)

        # Verify ServiceBus broker is selected
        assert celery_module.service_builder_broker_type == "servicebus"

    @patch("model_engine_server.service_builder.celery.CIRCLECI", False)
    @patch("model_engine_server.service_builder.celery.infra_config")
    @patch("model_engine_server.service_builder.celery.celery_app")
    def test_broker_type_sqs_default(self, mock_celery_app, mock_config, mock_circleci):
        """Test that SQS broker is selected as default"""
        # Setup mocks
        mock_config_instance = MagicMock()
        mock_config_instance.celery_broker_type_redis = False
        mock_config_instance.cloud_provider = "aws"
        mock_config_instance.s3_bucket = "test-bucket"
        mock_config.return_value = mock_config_instance

        # Force module reload to test the logic
        import importlib

        import model_engine_server.service_builder.celery as celery_module

        importlib.reload(celery_module)

        # Verify SQS broker is selected as default
        assert celery_module.service_builder_broker_type == "sqs"
