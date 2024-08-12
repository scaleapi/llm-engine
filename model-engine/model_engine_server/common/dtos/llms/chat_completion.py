from typing import Any, Dict, List, Optional

from model_engine_server.common.types.gen.openai import (
    CreateChatCompletionRequest,
    CreateChatCompletionResponse,
)
from pydantic import Field
from typing_extensions import Annotated

# Fields that are a part of OpenAI spec but are not supported by model engine
UNSUPPORTED_FIELDS = ["service_tier"]


class VLLMAdditionalFields:
    chat_template: Optional[str] = Field(
        default=None,
        description=(
            "A Jinja template to use for this conversion. "
            "As of transformers v4.44, default chat template is no longer "
            "allowed, so you must provide a chat template if the tokenizer "
            "does not define one."
        ),
    )
    chat_template_kwargs: Optional[Dict[str, Any]] = Field(
        default=None,
        description=(
            "Additional kwargs to pass to the template renderer. "
            "Will be accessible by the chat template."
        ),
    )

    guided_json: Annotated[
        Optional[Dict[str, Any]],
        Field(default=None, description="JSON schema for guided decoding. Only supported in vllm."),
    ]

    guided_regex: Annotated[
        Optional[str],
        Field(default=None, description="Regex for guided decoding. Only supported in vllm."),
    ]
    guided_choice: Annotated[
        Optional[List[str]],
        Field(default=None, description="Choices for guided decoding. Only supported in vllm."),
    ]

    guided_grammar: Annotated[
        Optional[str],
        Field(
            default=None,
            description="Context-free grammar for guided decoding. Only supported in vllm.",
        ),
    ]

    guided_decoding_backend: Annotated[
        Optional[str],
        Field(
            default=None,
            description=(
                "If specified, will override the default guided decoding backend "
                "of the server for this specific request. If set, must be either "
                "'outlines' / 'lm-format-enforcer'"
            ),
        ),
    ]

    guided_whitespace_pattern: Annotated[
        Optional[str],
        Field(
            default=None,
            description=(
                "If specified, will override the default whitespace pattern "
                "for guided json decoding."
            ),
        ),
    ]

    skip_special_tokens: Annotated[
        Optional[bool],
        Field(True, "Whether to skip special tokens in the output. Only supported in vllm."),
    ]


class ChatCompletionV2Request(CreateChatCompletionRequest, VLLMAdditionalFields):
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
        Optional[bool], Field(None, "Whether to include the stop strings in output text.")
    ]


class ChatCompletionV2Response(CreateChatCompletionResponse):
    pass
