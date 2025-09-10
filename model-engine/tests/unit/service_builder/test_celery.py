def test_module_imports():
    """Test that the celery module can be imported"""
    import model_engine_server.service_builder.celery as celery_module

    assert hasattr(celery_module, "service_builder_broker_type")
    assert hasattr(celery_module, "service_builder_service")
    assert celery_module.service_builder_broker_type in ["redis", "servicebus", "sqs"]


def test_broker_type_is_string():
    """Test that broker type is set to a valid string"""
    import model_engine_server.service_builder.celery as celery_module

    valid_types = ["redis", "servicebus", "sqs"]
    assert celery_module.service_builder_broker_type in valid_types
    assert isinstance(celery_module.service_builder_broker_type, str)
