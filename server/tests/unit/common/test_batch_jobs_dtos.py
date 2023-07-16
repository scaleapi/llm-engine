from spellbook_serve.common.dtos.batch_jobs import CreateDockerImageBatchJobResourceRequests


def test_create_docker_image_batch_job_resource_requests_merge_requests():
    # Test default = None
    default = CreateDockerImageBatchJobResourceRequests()
    override = CreateDockerImageBatchJobResourceRequests(cpus=1, memory="100Mi")

    actual = CreateDockerImageBatchJobResourceRequests.merge_requests(default, override)
    assert override == actual

    # Test override = None
    override = CreateDockerImageBatchJobResourceRequests()
    default = CreateDockerImageBatchJobResourceRequests(cpus=1, memory="100Mi")
    actual = CreateDockerImageBatchJobResourceRequests.merge_requests(default, override)
    assert default == actual

    # Test overriding
    default = CreateDockerImageBatchJobResourceRequests(cpus=0.5, memory="200Mi")
    override = CreateDockerImageBatchJobResourceRequests(cpus=1, memory="100Mi")
    actual = CreateDockerImageBatchJobResourceRequests.merge_requests(default, override)
    assert override == actual

    # Test merging
    default = CreateDockerImageBatchJobResourceRequests(cpus=0.5)
    override = CreateDockerImageBatchJobResourceRequests(
        memory="100Mi", gpus=1, gpu_type="nvidia-a100", storage="10Gi"
    )
    expected = CreateDockerImageBatchJobResourceRequests(
        cpus=0.5, memory="100Mi", gpus=1, gpu_type="nvidia-a100", storage="10Gi"
    )
    actual = CreateDockerImageBatchJobResourceRequests.merge_requests(default, override)
    assert expected == actual


def test_create_docker_image_batch_job_resource_requests_common_requests():
    # Test no common
    default = CreateDockerImageBatchJobResourceRequests(storage="1Ti")
    override = CreateDockerImageBatchJobResourceRequests(cpus=1, memory="100Mi")
    expected = set()

    actual = CreateDockerImageBatchJobResourceRequests.common_requests(default, override)
    assert expected == actual

    # Test common
    default = CreateDockerImageBatchJobResourceRequests(cpus=0.5, memory="200Mi", storage="1Ti")
    override = CreateDockerImageBatchJobResourceRequests(
        cpus=1, memory="100Mi", gpus=1, gpu_type="nvidia-tesla-t4"
    )
    expected = {"cpus", "memory"}

    actual = CreateDockerImageBatchJobResourceRequests.common_requests(default, override)
    assert expected == actual
