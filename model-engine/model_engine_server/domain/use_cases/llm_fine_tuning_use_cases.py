import csv
import datetime
import json
import re
from typing import Optional

import smart_open
from model_engine_server.common.dtos.llms import (
    CancelFineTuneResponse,
    CreateFineTuneRequest,
    CreateFineTuneResponse,
    GetFineTuneEventsResponse,
    GetFineTuneResponse,
    ListFineTunesResponse,
)
from model_engine_server.core.auth.authentication_repository import User
from model_engine_server.core.loggers import logger_name, make_logger
from model_engine_server.domain.entities import BatchJobStatus
from model_engine_server.domain.exceptions import (
    InvalidRequestException,
    LLMFineTuningQuotaReached,
    ObjectNotFoundException,
)
from model_engine_server.domain.gateways import FileStorageGateway
from model_engine_server.domain.repositories import LLMFineTuneEventsRepository
from model_engine_server.domain.services import LLMFineTuningService, ModelEndpointService

DEFAULT_FINE_TUNING_METHOD = "lora"
REQUIRED_COLUMNS = [["prompt", "response"], ["input", "output"]]

MAX_LLM_ENDPOINTS_PER_EXTERNAL_USER = 5

MAX_SUFFIX_LENGTH = 28
# k8s labels need to be <= 62 characters, timestamp takes 13 characters, 2 characters for periods,
# model name is currently 17 long, but want to add a bit of buffer.

logger = make_logger(logger_name())


def is_model_name_suffix_valid(model_name: str):
    pattern = "^[A-Za-z0-9-]+$"  # TODO can we do spaces and underscores
    return bool(re.match(pattern, model_name)) and len(model_name) <= MAX_SUFFIX_LENGTH


def ensure_model_name_is_valid_k8s_label(model_name: str):
    """
    Ensure the model name is usable as a k8s label,
    since we will end up creating a deployment with the model name as a label.
    """
    return re.sub("[^-A-Za-z0-9_.]", "-", model_name).lstrip("-_.")[:62].rstrip("-_.")


def read_csv_headers(file_location: str):
    """
    Read the headers of a csv file.
    This will also parse for a JSONL file and will return the first row of the file split by comma.
    """
    with smart_open.open(file_location, transport_params=dict(buffer_size=1024)) as file:
        csv_reader = csv.DictReader(file)
        return csv_reader.fieldnames


def are_dataset_headers_valid(file_location: str):
    """
    Ensure the dataset headers are valid with required columns 'prompt' and 'response'.
    """
    current_headers = read_csv_headers(file_location)
    first_line = ",".join(current_headers)
    try:
        object = json.loads(first_line)  # JSONL file format
        current_headers = object.keys()
    except json.decoder.JSONDecodeError:  # CSV file format
        pass
    return any(
        [
            all(header in current_headers for header in header_group)
            for header_group in REQUIRED_COLUMNS
        ]
    )


def check_file_is_valid(file_name: Optional[str], file_type: str):
    """
    Ensure the file is valid with required columns 'prompt' and 'response', isn't malformatted, and exists.
    Accepts CSV and JSONL formats.
    file_type: 'training' or 'validation'
    """
    try:
        if file_name is not None and not are_dataset_headers_valid(file_name):
            raise InvalidRequestException(
                f"Required column headers (one subset of {REQUIRED_COLUMNS}) not found in {file_type} dataset"
            )
    except FileNotFoundError:
        raise InvalidRequestException(
            f"Cannot find the {file_type} file. Verify the path and file name are correct."
        )
    except csv.Error as exc:
        raise InvalidRequestException(
            f"Cannot parse the {file_type} dataset as CSV. Details: {exc}"
        )


class CreateFineTuneV1UseCase:
    def __init__(
        self,
        llm_fine_tuning_service: LLMFineTuningService,
        model_endpoint_service: ModelEndpointService,
        llm_fine_tune_events_repository: LLMFineTuneEventsRepository,
        file_storage_gateway: FileStorageGateway,
    ):
        self.llm_fine_tuning_service = llm_fine_tuning_service
        self.model_endpoint_service = model_endpoint_service
        self.llm_fine_tune_events_repository = llm_fine_tune_events_repository
        self.file_storage_gateway = file_storage_gateway

    async def execute(self, user: User, request: CreateFineTuneRequest) -> CreateFineTuneResponse:
        di_batch_jobs = await self.llm_fine_tuning_service.list_fine_tunes(
            owner=user.team_id,
        )
        in_progress_jobs = [
            job
            for job in di_batch_jobs
            if job.status in [BatchJobStatus.PENDING, BatchJobStatus.RUNNING]
        ]
        model_endpoints = await self.model_endpoint_service.list_model_endpoints(
            owner=user.team_id, name=None, order_by=None
        )

        current_jobs_and_endpoints = len(in_progress_jobs) + len(model_endpoints)

        if (
            not user.is_privileged_user
            and current_jobs_and_endpoints >= MAX_LLM_ENDPOINTS_PER_EXTERNAL_USER
        ):
            raise LLMFineTuningQuotaReached(
                f"Limit {MAX_LLM_ENDPOINTS_PER_EXTERNAL_USER} fine-tunes/fine-tuned endpoints per user. "
                f"Cancel/delete a total of "
                f"{current_jobs_and_endpoints - MAX_LLM_ENDPOINTS_PER_EXTERNAL_USER + 1} pending or "
                f"running fine-tune(s) or fine-tuned endpoints to run another fine-tune."
            )

        if request.suffix is not None and not is_model_name_suffix_valid(request.suffix):
            raise InvalidRequestException(
                f"User-provided suffix is invalid, must only contain alphanumeric characters and dashes and be at most {MAX_SUFFIX_LENGTH} characters"
            )
        time_now = datetime.datetime.utcnow().strftime("%y%m%d-%H%M%S")
        # Colons breaks our download command. Keep delimiters as `.`
        fine_tuned_model = (
            f"{request.model}.{request.suffix}.{time_now}"
            if request.suffix is not None
            else f"{request.model}.{time_now}"
        )

        # We need to ensure fine_tuned_model conforms to the k8s label spec
        # This is unfortunately a leaky abstraction. This likely goes away if we redo how we implement fine-tuned
        # models though
        fine_tuned_model = ensure_model_name_is_valid_k8s_label(fine_tuned_model)

        if request.training_file.startswith("file-"):
            training_file_url = await self.file_storage_gateway.get_url_from_id(
                user.team_id, request.training_file
            )
            if training_file_url is None:
                raise ObjectNotFoundException("Training file does not exist")
        else:
            training_file_url = request.training_file

        if request.validation_file is not None and request.validation_file.startswith("file-"):
            validation_file_url = await self.file_storage_gateway.get_url_from_id(
                user.team_id, request.validation_file
            )
            if validation_file_url is None:
                raise ObjectNotFoundException("Validation file does not exist")
        else:
            validation_file_url = request.validation_file

        check_file_is_valid(training_file_url, "training")
        check_file_is_valid(validation_file_url, "validation")

        await self.llm_fine_tune_events_repository.initialize_events(user.team_id, fine_tuned_model)
        fine_tune_id = await self.llm_fine_tuning_service.create_fine_tune(
            created_by=user.user_id,
            owner=user.team_id,
            model=request.model,
            training_file=request.training_file,  # for Files API, pass file ID rather than signed URL since the latter expires; fine-tuning script will get file content from Files API
            validation_file=request.validation_file,
            fine_tuning_method=DEFAULT_FINE_TUNING_METHOD,
            hyperparameters=request.hyperparameters,
            fine_tuned_model=fine_tuned_model,
            wandb_config=request.wandb_config,
        )
        return CreateFineTuneResponse(
            id=fine_tune_id,
        )


class GetFineTuneV1UseCase:
    def __init__(self, llm_fine_tuning_service: LLMFineTuningService):
        self.llm_fine_tuning_service = llm_fine_tuning_service

    async def execute(self, user: User, fine_tune_id: str) -> GetFineTuneResponse:
        di_batch_job = await self.llm_fine_tuning_service.get_fine_tune(
            owner=user.team_id,
            fine_tune_id=fine_tune_id,
        )
        if di_batch_job is None:
            raise ObjectNotFoundException
        if di_batch_job.annotations:
            fine_tuned_model = di_batch_job.annotations.get("fine_tuned_model")
        else:
            fine_tuned_model = None
            logger.warning(f"Fine-tune {di_batch_job.id} has no annotations. This is unexpected.")
        return GetFineTuneResponse(
            id=di_batch_job.id,
            fine_tuned_model=fine_tuned_model,
            status=di_batch_job.status,
        )


class ListFineTunesV1UseCase:
    def __init__(self, llm_fine_tuning_service: LLMFineTuningService):
        self.llm_fine_tuning_service = llm_fine_tuning_service

    async def execute(self, user: User) -> ListFineTunesResponse:
        di_batch_jobs = await self.llm_fine_tuning_service.list_fine_tunes(
            owner=user.team_id,
        )
        return ListFineTunesResponse(
            jobs=[
                GetFineTuneResponse(
                    id=job.id,
                    status=job.status,
                    fine_tuned_model=job.annotations.get("fine_tuned_model")
                    if job.annotations
                    else None,
                )
                for job in di_batch_jobs
            ]
        )


class CancelFineTuneV1UseCase:
    def __init__(self, llm_fine_tuning_service: LLMFineTuningService):
        self.llm_fine_tuning_service = llm_fine_tuning_service

    async def execute(self, user: User, fine_tune_id: str) -> CancelFineTuneResponse:
        success = await self.llm_fine_tuning_service.cancel_fine_tune(
            owner=user.team_id,
            fine_tune_id=fine_tune_id,
        )
        return CancelFineTuneResponse(
            success=success,
        )


class GetFineTuneEventsV1UseCase:
    def __init__(
        self,
        llm_fine_tune_events_repository: LLMFineTuneEventsRepository,
        llm_fine_tuning_service: LLMFineTuningService,
    ):
        self.llm_fine_tune_events_repository = llm_fine_tune_events_repository
        self.llm_fine_tuning_service = llm_fine_tuning_service

    async def execute(self, user: User, fine_tune_id: str) -> GetFineTuneEventsResponse:
        model_endpoint_name = await self.llm_fine_tuning_service.get_fine_tune_model_name_from_id(
            user.team_id, fine_tune_id
        )
        if model_endpoint_name is None:
            raise ObjectNotFoundException(f"Fine-tune with id {fine_tune_id} not found")
        events = await self.llm_fine_tune_events_repository.get_fine_tune_events(
            user_id=user.team_id, model_endpoint_name=model_endpoint_name
        )
        return GetFineTuneEventsResponse(events=events)
