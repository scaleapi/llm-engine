from typing import Optional, TypeAlias

from .core import StreamError
from .gen.openai import (
    CreateChatCompletionRequest,
    CreateChatCompletionResponse,
    CreateChatCompletionStreamResponse,
)
from .pydantic_types import BaseModel, Field
from .vllm import VLLMChatCompletionAdditionalParams

# Fields that are a part of OpenAI spec but are not supported by model engine
UNSUPPORTED_FIELDS = ["service_tier"]


class ChatCompletionV2Request(CreateChatCompletionRequest, VLLMChatCompletionAdditionalParams):
    model: str = Field(
        description="ID of the model to use.",
        examples=["mixtral-8x7b-instruct"],
    )

    stream: Optional[bool] = Field(
        False,
        description="If set, partial message deltas will be sent. Tokens will be sent as data-only [server-sent events](https://developer.mozilla.org/en-US/docs/Web/API/Server-sent_events/Using_server-sent_events#Event_stream_format) as they become available, with the stream terminated by a `data: [DONE]` message. [Example Python code](https://cookbook.openai.com/examples/how_to_stream_completions).\n",
    )

    top_k: Optional[int] = Field(
        None,
        ge=-1,
        description="Controls the number of top tokens to consider. -1 means consider all tokens.",
    )

    include_stop_str_in_output: Optional[bool] = Field(
        None, description="Whether to include the stop strings in output text."
    )


ChatCompletionV2SyncResponse: TypeAlias = CreateChatCompletionResponse
ChatCompletionV2StreamSuccessChunk: TypeAlias = CreateChatCompletionStreamResponse


class ChatCompletionV2StreamErrorChunk(BaseModel):
    error: StreamError


ChatCompletionV2Chunk: TypeAlias = (
    ChatCompletionV2StreamSuccessChunk | ChatCompletionV2StreamErrorChunk
)

ChatCompletionV2Response: TypeAlias = ChatCompletionV2SyncResponse
