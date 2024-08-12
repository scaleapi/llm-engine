from typing import Dict, List, Optional, Union

from model_engine_server.common.dtos.llms.chat_completion import (
    ChatCompletionV2Request,
    ChatCompletionV2Response,
)
from model_engine_server.common.dtos.llms.completion import (
    CompletionV2Request,
    CompletionV2Response,
)
from model_engine_server.common.pydantic_types import BaseModel, ConfigDict, Field
from typing_extensions import TypeAlias

CompletionRequest: TypeAlias = Union[CompletionV2Request, ChatCompletionV2Request]
CompletionOutput: TypeAlias = Union[CompletionV2Response, ChatCompletionV2Response]


class BatchCompletionJob:
    job_id: str
    input_data_path: str
    output_data_path: str
    model_config: CompletionRequest
    priority: int
    status: str
    created_at: str
    expires_at: str
    completed_at: Optional[str]
    metadata: Optional[Dict[str, str]]


class CreateBatchCompletionsModelConfig(BaseModel):
    model: str = Field(
        description="ID of the model to use.",
        examples=["mixtral-8x7b-instruct"],
    )

    checkpoint_path: Optional[str] = Field(
        default=None, description="Path to the checkpoint to load the model from."
    )
    labels: Dict[str, str] = Field(
        default={}, description="Labels to attach to the batch inference job."
    )
    seed: Optional[int] = Field(default=None, description="Random seed for the model.")


class CreateBatchCompletionsRequestContent(BaseModel):
    pass


class CreateBatchCompletionsV2Request(BaseModel):
    """
    Request object for batch completions.
    """

    model_config = ConfigDict(protected_namespaces=())

    input_data_path: Optional[str] = Field(
        default=None,
        description="Path to the input file. The input file should be a JSON file of type List[CreateBatchCompletionsRequestContent].",
    )
    output_data_path: str = Field(
        description="Path to the output file. The output file will be a JSON file of type List[CompletionOutput]."
    )

    content: Optional[CreateBatchCompletionsRequestContent] = Field(
        default=None,
        description="""
Either `input_data_path` or `content` needs to be provided.
When input_data_path is provided, the input file should be a JSON file of type List[CreateBatchCompletionsRequestConttent].
""",
    )

    # We rename model_config from api to model_cfg in engine since engine uses pydantic v2 which
    #  reserves model_config as a keyword.
    model_cfg: CreateBatchCompletionsModelConfig = Field(
        alias="model_config",
        description="""Model configuration for the batch inference. Hardware configurations are inferred.""",
    )

    data_parallelism: Optional[int] = Field(
        default=1,
        ge=1,
        le=64,
        description="Number of replicas to run the batch inference. More replicas are slower to schedule but faster to inference.",
    )

    max_runtime_sec: Optional[int] = Field(
        default=24 * 3600,
        ge=1,
        le=2 * 24 * 3600,
        description="Maximum runtime of the batch inference in seconds. Default to one day.",
    )

    priority: Optional[str] = Field(
        default=None, description="Priority of the batch inference job. Default to normal."
    )


class ListBatchCompletionV2Response(BaseModel):
    jobs: List[BatchCompletionJob]


class GetBatchCompletionV2Response(BaseModel):
    job: BatchCompletionJob
