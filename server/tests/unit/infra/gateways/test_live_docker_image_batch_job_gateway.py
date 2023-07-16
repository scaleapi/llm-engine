from spellbook_serve.infra.gateways.live_docker_image_batch_job_gateway import (
    K8sEnvDict,
    _add_list_values,
    _check_batch_job_id_valid,
    _get_job_id,
)


def test_valid_job_ids_are_valid():
    for _ in range(20):
        # _get_job_id() is nondeterministic
        job_id = _get_job_id()
        assert _check_batch_job_id_valid(job_id), f"job_id {job_id} apparently isn't valid"


def test_invalid_job_ids_are_invalid():
    assert not _check_batch_job_id_valid("spaces fail")
    assert not _check_batch_job_id_valid("punctuation'")
    assert not _check_batch_job_id_valid(".")


# test the adding list values
def test_add_list_values():
    default_values = [
        K8sEnvDict(name="default1", value="val1"),
        K8sEnvDict(name="default2", value="val2"),
        K8sEnvDict(name="default3", value="val3"),
    ]
    override_values = [
        K8sEnvDict(name="default1", value="override0"),
        K8sEnvDict(name="override1", value="override1"),
        K8sEnvDict(name="override2", value="override2"),
    ]
    expected_values = [
        K8sEnvDict(name="default1", value="val1"),
        K8sEnvDict(name="default2", value="val2"),
        K8sEnvDict(name="default3", value="val3"),
        K8sEnvDict(name="override1", value="override1"),
        K8sEnvDict(name="override2", value="override2"),
    ]

    actual_values = _add_list_values(default_values, override_values)
    actual_values.sort(key=lambda x: x["name"])
    assert expected_values == actual_values
