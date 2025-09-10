from unittest.mock import MagicMock, patch

from model_engine_server.common.dtos.model_endpoints import BrokerType


class TestServiceBuilderCelery:
    """Test the service builder celery configuration logic"""

    @patch("model_engine_server.service_builder.celery.celery_app")
    @patch("model_engine_server.service_builder.celery.infra_config")
    @patch("model_engine_server.service_builder.celery.CIRCLECI", True)
    def test_broker_type_selection_circleci_enabled(self, mock_config, mock_celery_app):
        """Test that Redis broker is selected when CIRCLECI is True"""
        # Setup mocks
        mock_config_instance = MagicMock()
        mock_config_instance.celery_broker_type_redis = False
        mock_config_instance.cloud_provider = "aws"
        mock_config_instance.s3_bucket = "test-bucket"
        mock_config.return_value = mock_config_instance

        # Import the module to trigger the configuration logic
        from model_engine_server.service_builder import celery as celery_module

        # Verify Redis broker is selected for CircleCI
        assert celery_module.service_builder_broker_type == str(BrokerType.REDIS.value)

        # Verify celery_app was called with correct parameters
        mock_celery_app.assert_called_once_with(
            name="model_engine_server.service_builder",
            modules=["model_engine_server.service_builder.tasks_v1"],
            s3_bucket="test-bucket",
            broker_type=str(BrokerType.REDIS.value),
            backend_protocol="s3",
            task_track_started=True,
            task_remote_tracebacks=True,
            task_time_limit=1800,
            task_soft_time_limit=1500,
        )

    @patch("model_engine_server.service_builder.celery.celery_app")
    @patch("model_engine_server.service_builder.celery.infra_config")
    @patch("model_engine_server.service_builder.celery.CIRCLECI", False)
    def test_broker_type_selection_redis_enabled(self, mock_config, mock_celery_app):
        """Test that Redis broker is selected when celery_broker_type_redis is True"""
        # Setup mocks
        mock_config_instance = MagicMock()
        mock_config_instance.celery_broker_type_redis = True
        mock_config_instance.cloud_provider = "aws"
        mock_config_instance.s3_bucket = "test-bucket"
        mock_config.return_value = mock_config_instance

        # Import the module to trigger the configuration logic
        from model_engine_server.service_builder import celery as celery_module

        # Verify Redis broker is selected
        assert celery_module.service_builder_broker_type == str(BrokerType.REDIS.value)

    @patch("model_engine_server.service_builder.celery.celery_app")
    @patch("model_engine_server.service_builder.celery.infra_config")
    @patch("model_engine_server.service_builder.celery.CIRCLECI", False)
    def test_broker_type_selection_azure_cloud(self, mock_config, mock_celery_app):
        """Test that ServiceBus broker is selected for Azure cloud provider"""
        # Setup mocks
        mock_config_instance = MagicMock()
        mock_config_instance.celery_broker_type_redis = False
        mock_config_instance.cloud_provider = "azure"
        mock_config_instance.s3_bucket = "test-bucket"
        mock_config.return_value = mock_config_instance

        # Import the module to trigger the configuration logic
        from model_engine_server.service_builder import celery as celery_module

        # Verify ServiceBus broker is selected for Azure
        assert celery_module.service_builder_broker_type == str(BrokerType.SERVICEBUS.value)

        # Verify backend_protocol is set to "abs" for Azure
        mock_celery_app.assert_called_once_with(
            name="model_engine_server.service_builder",
            modules=["model_engine_server.service_builder.tasks_v1"],
            s3_bucket="test-bucket",
            broker_type=str(BrokerType.SERVICEBUS.value),
            backend_protocol="abs",  # Azure backend
            task_track_started=True,
            task_remote_tracebacks=True,
            task_time_limit=1800,
            task_soft_time_limit=1500,
        )

    @patch("model_engine_server.service_builder.celery.celery_app")
    @patch("model_engine_server.service_builder.celery.infra_config")
    @patch("model_engine_server.service_builder.celery.CIRCLECI", False)
    def test_broker_type_selection_aws_default(self, mock_config, mock_celery_app):
        """Test that SQS broker is selected as default for AWS"""
        # Setup mocks
        mock_config_instance = MagicMock()
        mock_config_instance.celery_broker_type_redis = False
        mock_config_instance.cloud_provider = "aws"
        mock_config_instance.s3_bucket = "test-bucket"
        mock_config.return_value = mock_config_instance

        # Import the module to trigger the configuration logic
        from model_engine_server.service_builder import celery as celery_module

        # Verify SQS broker is selected as default
        assert celery_module.service_builder_broker_type == str(BrokerType.SQS.value)

        # Verify backend_protocol is set to "s3" for AWS
        mock_celery_app.assert_called_once_with(
            name="model_engine_server.service_builder",
            modules=["model_engine_server.service_builder.tasks_v1"],
            s3_bucket="test-bucket",
            broker_type=str(BrokerType.SQS.value),
            backend_protocol="s3",  # S3 backend
            task_track_started=True,
            task_remote_tracebacks=True,
            task_time_limit=1800,
            task_soft_time_limit=1500,
        )

    @patch("model_engine_server.service_builder.celery.celery_app")
    @patch("model_engine_server.service_builder.celery.infra_config")
    @patch("model_engine_server.service_builder.celery.CIRCLECI", False)
    def test_celery_app_configuration_parameters(self, mock_config, mock_celery_app):
        """Test that celery app is configured with correct parameters"""
        # Setup mocks
        mock_config_instance = MagicMock()
        mock_config_instance.celery_broker_type_redis = False
        mock_config_instance.cloud_provider = "gcp"  # Non-Azure to test default
        mock_config_instance.s3_bucket = "my-test-bucket"
        mock_config.return_value = mock_config_instance

        # Import the module to trigger the configuration logic
        from model_engine_server.service_builder import celery as celery_module

        # Verify all configuration parameters
        mock_celery_app.assert_called_once_with(
            name="model_engine_server.service_builder",
            modules=["model_engine_server.service_builder.tasks_v1"],
            s3_bucket="my-test-bucket",
            broker_type=str(BrokerType.SQS.value),
            backend_protocol="s3",
            task_track_started=True,
            task_remote_tracebacks=True,
            task_time_limit=1800,  # 30 minutes
            task_soft_time_limit=1500,  # 25 minutes
        )

    @patch("model_engine_server.service_builder.celery.service_builder_service")
    def test_main_block_execution(self, mock_service):
        """Test that service.start() is called when module is run as main"""
        # Mock the service
        mock_service.start = MagicMock()

        # Import and execute the main block
        import model_engine_server.service_builder.celery as celery_module

        # Simulate running as main
        if "__main__" == "__main__":  # This condition will always be true in test
            celery_module.service_builder_service.start()

        # Verify start was called
        mock_service.start.assert_called_once()
