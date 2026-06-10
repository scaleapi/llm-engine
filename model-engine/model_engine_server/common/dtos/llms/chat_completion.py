from typing import List, Optional, Union

from model_engine_server.common.dtos.llms.completion import StreamError
from model_engine_server.common.dtos.llms.vllm import VLLMChatCompletionAdditionalParams
from model_engine_server.common.pydantic_types import AliasChoices, BaseModel, Field, RootModel
from model_engine_server.common.types.gen.openai import (
    ChatCompletionRequestAssistantMessage,
    ChatCompletionRequestDeveloperMessage,
    ChatCompletionRequestFunctionMessage,
    ChatCompletionRequestSystemMessage,
    ChatCompletionRequestToolMessage,
    ChatCompletionRequestUserMessage,
    ChatCompletionResponseMessage,
    ChatCompletionStreamResponseDelta,
    Choice,
    Choice1,
    CreateChatCompletionRequest,
    CreateChatCompletionResponse,
    CreateChatCompletionStreamResponse,
)
from sse_starlette import EventSourceResponse
from typing_extensions import Annotated, TypeAlias

# Fields that are a part of OpenAI spec but are not supported by model engine
UNSUPPORTED_FIELDS = ["service_tier"]

# The OpenAI spec has no field for the reasoning (thinking) traces emitted by reasoning
# models, so the generated spec models silently drop them. The subclasses below extend the
# spec so reasoning content survives a round trip through model engine:
#   - Responses surface it as `reasoning_content` (the LiteLLM/DeepSeek convention),
#     accepting either `reasoning` (vLLM) or `reasoning_content` from the inference
#     framework.
#   - Request assistant messages accept either name and forward it to the inference
#     framework as `reasoning` (the field vLLM exposes to chat templates), so chat
#     templates that render prior-turn reasoning keep working.
# Both fields are optional. Requests are forwarded and stream chunks serialized with
# `exclude_none=True`, so those payloads are unchanged for non-reasoning models; sync
# responses gain a `reasoning_content: null` key alongside the other nullable spec
# fields (`refusal`, `audio`, ...) that FastAPI already serializes.


class ChatCompletionRequestAssistantMessageWithReasoning(ChatCompletionRequestAssistantMessage):
    reasoning: Annotated[
        Optional[str],
        Field(
            validation_alias=AliasChoices("reasoning", "reasoning_content"),
            description="The reasoning (thinking) trace of a previous assistant turn, used to"
            " round trip reasoning back to the model's chat template. Accepts `reasoning`"
            " (vLLM convention) or `reasoning_content` (LiteLLM/DeepSeek convention) on"
            " input; forwarded to the inference framework as `reasoning`.\n",
        ),
    ] = None


class ChatCompletionRequestMessageWithReasoning(
    RootModel[
        Union[
            ChatCompletionRequestDeveloperMessage,
            ChatCompletionRequestSystemMessage,
            ChatCompletionRequestUserMessage,
            ChatCompletionRequestAssistantMessageWithReasoning,
            ChatCompletionRequestToolMessage,
            ChatCompletionRequestFunctionMessage,
        ]
    ]
):
    root: Union[
        ChatCompletionRequestDeveloperMessage,
        ChatCompletionRequestSystemMessage,
        ChatCompletionRequestUserMessage,
        ChatCompletionRequestAssistantMessageWithReasoning,
        ChatCompletionRequestToolMessage,
        ChatCompletionRequestFunctionMessage,
    ]


class ChatCompletionV2Request(CreateChatCompletionRequest, VLLMChatCompletionAdditionalParams):
    messages: Annotated[  # type: ignore[assignment]
        List[ChatCompletionRequestMessageWithReasoning],
        Field(
            description="A list of messages comprising the conversation so far. Depending on the\n[model](/docs/models) you use, different message types (modalities) are\nsupported, like [text](/docs/guides/text-generation),\n[images](/docs/guides/vision), and [audio](/docs/guides/audio).\n",
            min_length=1,
        ),
    ]

    model: Annotated[  # type: ignore[assignment]
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


class ChatCompletionResponseMessageWithReasoning(ChatCompletionResponseMessage):
    reasoning_content: Annotated[
        Optional[str],
        Field(
            validation_alias=AliasChoices("reasoning_content", "reasoning"),
            description="The reasoning (thinking) trace produced by reasoning models, when the"
            " inference framework returns one (e.g. a vLLM reasoning parser). Accepted from"
            " the framework as `reasoning` or `reasoning_content`; always serialized as"
            " `reasoning_content`.\n",
        ),
    ] = None


class ChatCompletionChoiceWithReasoning(Choice):
    message: ChatCompletionResponseMessageWithReasoning  # type: ignore[assignment]


class ChatCompletionV2SyncResponse(CreateChatCompletionResponse):
    choices: Annotated[  # type: ignore[assignment]
        List[ChatCompletionChoiceWithReasoning],
        Field(
            description="A list of chat completion choices. Can be more than one if `n` is greater than 1."
        ),
    ]


class ChatCompletionStreamResponseDeltaWithReasoning(ChatCompletionStreamResponseDelta):
    reasoning_content: Annotated[
        Optional[str],
        Field(
            validation_alias=AliasChoices("reasoning_content", "reasoning"),
            description="The reasoning (thinking) trace of the chunk message, when the inference"
            " framework returns one (e.g. a vLLM reasoning parser). Accepted from the"
            " framework as `reasoning` or `reasoning_content`; always serialized as"
            " `reasoning_content`.\n",
        ),
    ] = None


class ChatCompletionStreamChoiceWithReasoning(Choice1):
    delta: ChatCompletionStreamResponseDeltaWithReasoning  # type: ignore[assignment]


class ChatCompletionV2StreamSuccessChunk(CreateChatCompletionStreamResponse):
    choices: Annotated[  # type: ignore[assignment]
        List[ChatCompletionStreamChoiceWithReasoning],
        Field(
            description='A list of chat completion choices. Can contain more than one elements if `n` is greater than 1. Can also be empty for the\nlast chunk if you set `stream_options: {"include_usage": true}`.\n'
        ),
    ]


class ChatCompletionV2StreamErrorChunk(BaseModel):
    error: StreamError


ChatCompletionV2Chunk: TypeAlias = (
    ChatCompletionV2StreamSuccessChunk | ChatCompletionV2StreamErrorChunk
)
ChatCompletionV2StreamResponse: TypeAlias = (
    EventSourceResponse  # EventSourceResponse[ChatCompletionV2Chunk]
)

ChatCompletionV2Response: TypeAlias = ChatCompletionV2SyncResponse | ChatCompletionV2StreamResponse

# This is a version of ChatCompletionV2Response that is used by pydantic to determine the response model
# Since EventSourceResponse isn't a pydantic model, we need to use a Union of the two response types
ChatCompletionV2ResponseItem: TypeAlias = ChatCompletionV2SyncResponse | ChatCompletionV2Chunk
