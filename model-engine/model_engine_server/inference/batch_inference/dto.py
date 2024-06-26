# This is a copy of model_engine_server.common.dtos.llm
# This is done to decouple the pydantic requirements since vllm requires pydantic >2
# while model engine is on 1.x
from enum import Enum
from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class TokenOutput(BaseModel):
    token: str
    log_prob: float


class CompletionOutput(BaseModel):
    text: str
    num_prompt_tokens: int
    num_completion_tokens: int
    tokens: Optional[List[TokenOutput]] = None


class CreateBatchCompletionsRequestContent(BaseModel):
    prompts: List[str]
    max_new_tokens: int
    temperature: float = Field(ge=0.0, le=1.0)
    """
    Temperature of the sampling. Setting to 0 equals to greedy sampling.
    """
    stop_sequences: Optional[List[str]] = None
    """
    List of sequences to stop the completion at.
    """
    return_token_log_probs: Optional[bool] = False
    """
    Whether to return the log probabilities of the tokens.
    """
    presence_penalty: Optional[float] = Field(default=None, ge=0.0, le=2.0)
    """
    Only supported in vllm, lightllm
    Penalize new tokens based on whether they appear in the text so far. 0.0 means no penalty
    """
    frequency_penalty: Optional[float] = Field(default=None, ge=0.0, le=2.0)
    """
    Only supported in vllm, lightllm
    Penalize new tokens based on their existing frequency in the text so far. 0.0 means no penalty
    """
    top_k: Optional[int] = Field(default=None, ge=-1)
    """
    Controls the number of top tokens to consider. -1 means consider all tokens.
    """
    top_p: Optional[float] = Field(default=None, gt=0.0, le=1.0)
    """
    Controls the cumulative probability of the top tokens to consider. 1.0 means consider all tokens.
    """
    skip_special_tokens: Optional[bool] = True
    """
    Whether to skip special tokens in the output.
    """


class Quantization(str, Enum):
    BITSANDBYTES = "bitsandbytes"
    AWQ = "awq"


class CreateBatchCompletionsModelConfig(BaseModel):
    model: str
    checkpoint_path: Optional[str] = None
    """
    Path to the checkpoint to load the model from.
    """
    labels: Dict[str, str]
    """
    Labels to attach to the batch inference job.
    """
    num_shards: Optional[int] = 1
    """
    Suggested number of shards to distribute the model. When not specified, will infer the number of shards based on model config.
    System may decide to use a different number than the given value.
    """
    quantize: Optional[Quantization] = None
    """
    Whether to quantize the model.
    """
    seed: Optional[int] = None
    """
    Random seed for the model.
    """


class ToolConfig(BaseModel):
    """
    Configuration for tool use.
    NOTE: this config is highly experimental and signature will change significantly in future iterations.
    """

    name: str
    """
    Name of the tool to use for the batch inference.
    """
    max_iterations: Optional[int] = 10
    """
    Maximum number of iterations to run the tool.
    """
    execution_timeout_seconds: Optional[int] = 60
    """
    Maximum runtime of the tool in seconds.
    """
    should_retry_on_error: Optional[bool] = True
    """
    Whether to retry the tool on error.
    """


class CreateBatchCompletionsRequest(BaseModel):
    """
    Request object for batch completions.
    """

    input_data_path: Optional[str]
    output_data_path: str
    """
    Path to the output file. The output file will be a JSON file of type List[CompletionOutput].
    """
    content: Optional[CreateBatchCompletionsRequestContent] = None
    """
    Either `input_data_path` or `content` needs to be provided.
    When input_data_path is provided, the input file should be a JSON file of type BatchCompletionsRequestContent.
    """

    data_parallelism: Optional[int] = Field(default=1, ge=1, le=64)
    """
    Number of replicas to run the batch inference. More replicas are slower to schedule but faster to inference.
    """
    max_runtime_sec: Optional[int] = Field(default=24 * 3600, ge=1, le=2 * 24 * 3600)
    """
    Maximum runtime of the batch inference in seconds. Default to one day.
    """
    tool_config: Optional[ToolConfig] = None
    """
    Configuration for tool use.
    NOTE: this config is highly experimental and signature will change significantly in future iterations.
    """


class CreateBatchCompletionsEngineRequest(CreateBatchCompletionsRequest):
    """
    Internal model for representing request to the llm engine. This contains additional fields that we want
    hidden from the DTO exposed to the client.
    """

    model_cfg: CreateBatchCompletionsModelConfig = Field(alias="model_config")
    """
    Model configuration for the batch inference. Hardware configurations are inferred.
    
    We rename model_config from api to model_cfg in engine since engine uses pydantic v2 which
    reserves model_config as a keyword.

    We alias `model_config` for deserialization for backwards compatibility.
    """

    max_gpu_memory_utilization: Optional[float] = Field(default=0.9, le=1.0)
    """
    Maximum GPU memory utilization for the batch inference. Default to 90%.
    """
