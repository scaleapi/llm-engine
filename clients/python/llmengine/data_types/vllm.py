from typing import Any, Dict, List, Optional, Union

from typing_extensions import Annotated

from .gen.openai import ResponseFormatJsonObject, ResponseFormatJsonSchema, ResponseFormatText
from .pydantic_types import BaseModel, Field

# This was last synced w/ vLLM v0.5.5 on 2024-09-03


class VLLMModelConfig(BaseModel):
    """Model configuration for VLLM"""

    max_model_len: Optional[int] = Field(
        None,
        description="""Model context length, If unspecified, will be automatically derived from the model config""",
    )

    max_num_seqs: Optional[int] = Field(
        None,
        description="""Maximum number of sequences per iteration""",
    )

    enforce_eager: Optional[bool] = Field(
        None,
        description="""Always use eager-mode PyTorch. If False, will use eager mode and CUDA graph in hybrid for maximal perforamnce and flexibility""",
    )

    gpu_memory_utilization: Optional[float] = Field(
        None,
        description="Maximum GPU memory utilization for the batch inference. Default to 90%.",
    )

    trust_remote_code: Optional[bool] = Field(
        default=False,
        description="Whether to trust remote code from Hugging face hub. This is only applicable to models whose code is not supported natively by the transformers library (e.g. deepseek). Default to False.",
    )


class VLLMEngineAdditionalArgs(BaseModel):
    """Additional arguments to configure for vLLM that are not direct inputs to the vLLM engine"""

    max_gpu_memory_utilization: Optional[float] = Field(
        None,
        description="Maximum GPU memory utilization for the batch inference. Default to 90%. Deprecated in favor of specifying this in VLLMModelConfig",
    )

    attention_backend: Optional[str] = Field(
        default=None,
        description="Attention backend to use for vLLM. Default to None.",
    )


class VLLMEndpointAdditionalArgs(VLLMModelConfig, VLLMEngineAdditionalArgs, BaseModel):
    pass


class VLLMSamplingParams(BaseModel):
    best_of: Optional[int] = Field(
        None,
        description="""Number of output sequences that are generated from the prompt.
            From these `best_of` sequences, the top `n` sequences are returned.
            `best_of` must be greater than or equal to `n`. This is treated as
            the beam width when `use_beam_search` is True. By default, `best_of`
            is set to `n`.""",
    )
    top_k: Annotated[
        Optional[int],
        Field(
            None,
            ge=-1,
            description="Controls the number of top tokens to consider. -1 means consider all tokens.",
        ),
    ]
    min_p: Optional[float] = Field(
        None,
        description="""Float that represents the minimum probability for a token to be
            considered, relative to the probability of the most likely token.
            Must be in [0, 1]. Set to 0 to disable this.""",
    )
    use_beam_search: Optional[bool] = Field(
        None,
        description="""Whether to use beam search for sampling.""",
    )
    length_penalty: Optional[float] = Field(
        default=None,
        description="""Float that penalizes sequences based on their length.
            Used in beam search.""",
    )
    repetition_penalty: Optional[float] = Field(
        default=None,
        description="""Float that penalizes new tokens based on whether
            they appear in the prompt and the generated text so far. Values > 1
            encourage the model to use new tokens, while values < 1 encourage
            the model to repeat tokens.""",
    )
    early_stopping: Optional[bool] = Field(
        None,
        description="""Controls the stopping condition for beam search. It
            accepts the following values: `True`, where the generation stops as
            soon as there are `best_of` complete candidates; `False`, where an
            heuristic is applied and the generation stops when is it very
            unlikely to find better candidates; `"never"`, where the beam search
            procedure only stops when there cannot be better candidates
            (canonical beam search algorithm).""",
    )
    stop_token_ids: Optional[List[int]] = Field(
        default_factory=list,
        description="""List of tokens that stop the generation when they are
            generated. The returned output will contain the stop tokens unless
            the stop tokens are special tokens.""",
    )
    include_stop_str_in_output: Annotated[
        Optional[bool],
        Field(
            None,
            description="""Whether to include the stop strings in
            output text. Defaults to False.""",
        ),
    ]
    ignore_eos: Optional[bool] = Field(
        None,
        description="""Whether to ignore the EOS token and continue generating
            tokens after the EOS token is generated.""",
    )
    min_tokens: Optional[int] = Field(
        None,
        description="""Minimum number of tokens to generate per output sequence
            before EOS or stop_token_ids can be generated""",
    )

    skip_special_tokens: Optional[bool] = Field(
        True,
        description="Whether to skip special tokens in the output. Only supported in vllm.",
    )

    spaces_between_special_tokens: Optional[bool] = Field(
        True,
        description="Whether to add spaces between special tokens in the output. Only supported in vllm.",
    )


class VLLMChatCompletionAdditionalParams(VLLMSamplingParams):
    chat_template: Optional[str] = Field(
        default=None,
        description=(
            "A Jinja template to use for this conversion. "
            "As of transformers v4.44, default chat template is no longer "
            "allowed, so you must provide a chat template if the model's tokenizer "
            "does not define one and no override template is given"
        ),
    )
    chat_template_kwargs: Optional[Dict[str, Any]] = Field(
        default=None,
        description=(
            "Additional kwargs to pass to the template renderer. "
            "Will be accessible by the chat template."
        ),
    )

    guided_json: Optional[Dict[str, Any]] = Field(
        default=None,
        description="JSON schema for guided decoding. Only supported in vllm.",
    )

    guided_regex: Optional[str] = Field(
        default=None,
        description="Regex for guided decoding. Only supported in vllm.",
    )
    guided_choice: Optional[List[str]] = Field(
        default=None,
        description="Choices for guided decoding. Only supported in vllm.",
    )

    guided_grammar: Optional[str] = Field(
        default=None,
        description="Context-free grammar for guided decoding. Only supported in vllm.",
    )

    guided_decoding_backend: Optional[str] = Field(
        default=None,
        description=(
            "If specified, will override the default guided decoding backend "
            "of the server for this specific request. If set, must be either "
            "'outlines' / 'lm-format-enforcer'"
        ),
    )

    guided_whitespace_pattern: Optional[str] = Field(
        default=None,
        description=(
            "If specified, will override the default whitespace pattern "
            "for guided json decoding."
        ),
    )


class VLLMCompletionAdditionalParams(VLLMSamplingParams):
    add_special_tokens: Optional[bool] = Field(
        default=None,
        description=(
            "If true (the default), special tokens (e.g. BOS) will be added to " "the prompt."
        ),
    )

    response_format: Optional[
        Union[ResponseFormatText, ResponseFormatJsonObject, ResponseFormatJsonSchema]
    ] = Field(
        default=None,
        description=(
            "Similar to chat completion, this parameter specifies the format of "
            "output. Only {'type': 'json_object'} or {'type': 'text' } is "
            "supported."
        ),
    )

    guided_json: Optional[Dict[str, Any]] = Field(
        default=None,
        description="JSON schema for guided decoding. Only supported in vllm.",
    )

    guided_regex: Optional[str] = Field(
        default=None,
        description="Regex for guided decoding. Only supported in vllm.",
    )
    guided_choice: Optional[List[str]] = Field(
        default=None,
        description="Choices for guided decoding. Only supported in vllm.",
    )

    guided_grammar: Optional[str] = Field(
        default=None,
        description="Context-free grammar for guided decoding. Only supported in vllm.",
    )

    guided_decoding_backend: Optional[str] = Field(
        default=None,
        description=(
            "If specified, will override the default guided decoding backend "
            "of the server for this specific request. If set, must be either "
            "'outlines' / 'lm-format-enforcer'"
        ),
    )

    guided_whitespace_pattern: Optional[str] = Field(
        default=None,
        description=(
            "If specified, will override the default whitespace pattern "
            "for guided json decoding."
        ),
    )
