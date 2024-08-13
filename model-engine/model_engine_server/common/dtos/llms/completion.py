from typing import Any, Dict, List, Optional

from model_engine_server.common.pydantic_types import BaseModel, Field
from model_engine_server.common.types.gen.openai import (
    CreateCompletionRequest,
    CreateCompletionResponse,
)
from typing_extensions import Annotated

# Fields that are a part of OpenAI spec but are not supported by model engine
UNSUPPORTED_FIELDS = ["service_tier"]


class CompletionSyncV1Request(BaseModel):
    """
    Request object for a synchronous prompt completion task.
    """

    prompt: str
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
    include_stop_str_in_output: Optional[bool] = None
    """
    Whether to include the stop strings in output text.
    """
    guided_json: Optional[Dict[str, Any]] = None
    """
    JSON schema for guided decoding. Only supported in vllm.
    """
    guided_regex: Optional[str] = None
    """
    Regex for guided decoding. Only supported in vllm.
    """
    guided_choice: Optional[List[str]] = None
    """
    Choices for guided decoding. Only supported in vllm.
    """
    guided_grammar: Optional[str] = None
    """
    Context-free grammar for guided decoding. Only supported in vllm.
    """
    skip_special_tokens: Optional[bool] = True
    """
    Whether to skip special tokens in the output. Only supported in vllm.
    """


class TokenOutput(BaseModel):
    token: str
    log_prob: float


class CompletionOutput(BaseModel):
    text: str
    num_prompt_tokens: int
    num_completion_tokens: int
    tokens: Optional[List[TokenOutput]] = None


class CompletionSyncV1Response(BaseModel):
    """
    Response object for a synchronous prompt completion task.
    """

    request_id: Optional[str] = None
    output: Optional[CompletionOutput] = None


class CompletionStreamV1Request(BaseModel):
    """
    Request object for a stream prompt completion task.
    """

    prompt: str
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
    include_stop_str_in_output: Optional[bool] = None
    """
    Whether to include the stop strings in output text.
    """
    guided_json: Optional[Dict[str, Any]] = None
    """
    JSON schema for guided decoding. Only supported in vllm.
    """
    guided_regex: Optional[str] = None
    """
    Regex for guided decoding. Only supported in vllm.
    """
    guided_choice: Optional[List[str]] = None
    """
    Choices for guided decoding. Only supported in vllm.
    """
    guided_grammar: Optional[str] = None
    """
    Context-free grammar for guided decoding. Only supported in vllm.
    """
    skip_special_tokens: Optional[bool] = True
    """
    Whether to skip special tokens in the output. Only supported in vllm.
    """


class CompletionStreamOutput(BaseModel):
    text: str
    finished: bool
    num_prompt_tokens: Optional[int] = None
    num_completion_tokens: Optional[int] = None
    token: Optional[TokenOutput] = None


class StreamErrorContent(BaseModel):
    error: str
    """Error message."""
    timestamp: str
    """Timestamp of the error."""


class StreamError(BaseModel):
    """
    Error object for a stream prompt completion task.
    """

    status_code: int
    """The HTTP status code of the error."""
    content: StreamErrorContent
    """The error content."""


class CompletionStreamV1Response(BaseModel):
    """
    Response object for a stream prompt completion task.
    """

    request_id: Optional[str] = None
    output: Optional[CompletionStreamOutput] = None
    error: Optional[StreamError] = None
    """Error of the response (if any)."""


class TokenUsage(BaseModel):
    """
    Token usage for a prompt completion task.
    """

    num_prompt_tokens: Optional[int] = 0
    num_completion_tokens: Optional[int] = 0
    total_duration: Optional[float] = None
    """Includes time spent waiting for the model to be ready."""

    time_to_first_token: Optional[float] = None  # Only for streaming requests

    @property
    def num_total_tokens(self) -> int:
        return (self.num_prompt_tokens or 0) + (self.num_completion_tokens or 0)

    @property
    def total_tokens_per_second(self) -> float:
        return (
            self.num_total_tokens / self.total_duration
            if self.total_duration and self.total_duration > 0
            else 0.0
        )

    @property
    def inter_token_latency(self) -> Optional[float]:  # Only for streaming requests
        # Note: we calculate a single inter-token latency for the entire request.
        # Calculating latency between each token seems a bit heavyweight, although we can do this if we wanted
        if (
            self.time_to_first_token is None
            or self.num_completion_tokens is None
            or self.total_duration is None
        ):
            return None
        if self.num_completion_tokens < 2:
            return None
        return (self.total_duration - self.time_to_first_token) / (self.num_completion_tokens - 1)


class CompletionV2Request(CreateCompletionRequest):
    model: Annotated[
        str,
        Field(
            description="ID of the model to use.",
            examples=["mixtral-8x7b-instruct"],
        ),
    ]

    stream: Annotated[
        Optional[bool],
        Field(
            False,
            description="If set, partial message deltas will be sent. Tokens will be sent as data-only [server-sent events](https://developer.mozilla.org/en-US/docs/Web/API/Server-sent_events/Using_server-sent_events#Event_stream_format) as they become available, with the stream terminated by a `data: [DONE]` message. [Example Python code](https://cookbook.openai.com/examples/how_to_stream_completions).\n",
        ),
    ]

    top_k: Annotated[
        Optional[int],
        Field(
            None,
            ge=-1,
            description="Controls the number of top tokens to consider. -1 means consider all tokens.",
        ),
    ]

    include_stop_str_in_output: Annotated[
        Optional[bool],
        Field(None, description="Whether to include the stop strings in output text."),
    ]


class CompletionV2Response(CreateCompletionResponse):
    pass
