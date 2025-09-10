def test_init_module():
    """Test that service_builder module properly imports and exports celery"""
    import model_engine_server.service_builder as sb

    # Test that celery is accessible
    assert hasattr(sb, "celery")

    # Test that __all__ is set correctly
    assert sb.__all__ == ["celery"]
