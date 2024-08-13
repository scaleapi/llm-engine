from typing import Optional

from model_engine_server.common.pydantic_types import Field
from model_engine_server.common.types.gen.openai import (
    CreateCompletionRequest,
    CreateCompletionResponse,
)
from typing_extensions import Annotated

# Fields that are a part of OpenAI spec but are not supported by model engine
UNSUPPORTED_FIELDS = ["service_tier"]


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
