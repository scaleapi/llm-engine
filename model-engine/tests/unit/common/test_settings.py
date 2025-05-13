from model_engine_server.common.settings import get_service_builder_queue


def test_get_service_builder_queue():
    assert (
        get_service_builder_queue(service_identifier=None, service_builder_queue_name="test")
        == "test"
    )
    assert (
        get_service_builder_queue(service_identifier=None, service_builder_queue_name=None)
        == "model-engine-service-builder"
    )
    assert (
        get_service_builder_queue(service_identifier="test", service_builder_queue_name=None)
        == "model-engine-test-service-builder"
    )
