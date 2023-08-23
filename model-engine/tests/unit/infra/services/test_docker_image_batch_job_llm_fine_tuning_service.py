import pytest
import pytest_asyncio
from model_engine_server.domain.entities.llm_fine_tune_entity import LLMFineTuneTemplate
from model_engine_server.infra.services import DockerImageBatchJobLLMFineTuningService


@pytest_asyncio.fixture
async def live_docker_image_batch_job_llm_fine_tuning_service(
    fake_docker_image_batch_job_gateway,
    fake_docker_image_batch_job_bundle_repository,
    fake_llm_fine_tune_repository,
):
    fake_bundle = (
        await fake_docker_image_batch_job_bundle_repository.create_docker_image_batch_job_bundle(
            name="fake_fine_tune_bundle",
            created_by="fake_egp_admin",
            owner="fake_egp_admin",
            image_repository="fake_image_repo",
            image_tag="fake_image_tag",
            command=["fake_command"],
            env={"fake_env": "fake_env"},
            mount_location="/fake_mount_location",
            cpus="1",
            memory="0.1Gi",
            storage="1Gi",
            gpus=0,
            gpu_type=None,
            public=True,
        )
    )
    await fake_llm_fine_tune_repository.write_job_template_for_model(
        model_name="fake_model_name",
        fine_tuning_method="fake_fine_tuning_method",
        job_template=LLMFineTuneTemplate(
            docker_image_batch_job_bundle_id=fake_bundle.id,
            launch_endpoint_config={},
            default_hparams={},
            required_params=[],
        ),
    )
    return DockerImageBatchJobLLMFineTuningService(
        docker_image_batch_job_gateway=fake_docker_image_batch_job_gateway,
        docker_image_batch_job_bundle_repo=fake_docker_image_batch_job_bundle_repository,
        llm_fine_tune_repository=fake_llm_fine_tune_repository,
    )


@pytest.mark.asyncio
async def test_create_fine_tune_success(
    live_docker_image_batch_job_llm_fine_tuning_service,
    fake_docker_image_batch_job_gateway,
):
    batch_job_id = await live_docker_image_batch_job_llm_fine_tuning_service.create_fine_tune(
        created_by="fake_user",
        owner="fake_user",
        model="fake_model_name",
        training_file="fake_training_file_path",
        validation_file="fake_validation_file_path",
        fine_tuning_method="fake_fine_tuning_method",
        hyperparameters={},
        fine_tuned_model="fake_fine_tuned_model_name",
        wandb_config=None,
    )
    assert batch_job_id is not None
    assert fake_docker_image_batch_job_gateway.get_docker_image_batch_job(batch_job_id) is not None
