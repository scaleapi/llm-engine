from typing import Optional

from model_engine_server.common.dtos.llms.vllm import VLLMChatCompletionAdditionalParams
from model_engine_server.common.pydantic_types import Field
from model_engine_server.common.types.gen.openai import (
    CreateChatCompletionRequest,
    CreateChatCompletionResponse,
    CreateChatCompletionStreamResponse,
)
from typing_extensions import Annotated

# Fields that are a part of OpenAI spec but are not supported by model engine
UNSUPPORTED_FIELDS = ["service_tier"]


class ChatCompletionV2Request(CreateChatCompletionRequest, VLLMChatCompletionAdditionalParams):
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


class ChatCompletionV2Response(CreateChatCompletionResponse):
    pass


class ChatCompletionV2Chunk(CreateChatCompletionStreamResponse):
    pass
