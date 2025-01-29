from typing import Any, Dict, List, Optional, Union

from model_engine_server.common.pydantic_types import BaseModel, Field
from model_engine_server.common.types.gen.openai import (
    ResponseFormatJsonObject,
    ResponseFormatJsonSchema,
    ResponseFormatText,
)

# This was last synced w/ vLLM v0.6.4.post1 on 2024-12-10


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

    trust_remote_code: Optional[bool] = Field(
        default=False,
        description="Whether to trust remote code from Hugging face hub. This is only applicable to models whose code is not supported natively by the transformers library (e.g. deepseek). Default to False.",
    )

    pipeline_parallel_size: Optional[int] = Field(
        None,
        description="Number of pipeline stages. Default to None.",
    )

    tensor_parallel_size: Optional[int] = Field(
        None,
        description="Number of tensor parallel replicas. Default to None.",
    )

    quantization: Optional[str] = Field(
        None,
        description="Method used to quantize the weights. If "
        "None, we first check the `quantization_config` "
        "attribute in the model config file. If that is "
        "None, we assume the model weights are not "
        "quantized and use `dtype` to determine the data "
        "type of the weights.",
    )

    disable_log_requests: Optional[bool] = Field(
        None,
        description="Disable logging requests. Default to None.",
    )

    chat_template: Optional[str] = Field(
        None,
        description="A Jinja template to use for this endpoint. If not provided, will use the chat template from the checkpoint",
    )

    tool_call_parser: Optional[str] = Field(
        None,
        description="Tool call parser",
    )

    enable_auto_tool_choice: Optional[bool] = Field(
        None,
        description="Enable auto tool choice",
    )

    load_format: Optional[str] = Field(
        None,
        description="The format of the model weights to load.\n\n"
        '* "auto" will try to load the weights in the safetensors format '
        "and fall back to the pytorch bin format if safetensors format "
        "is not available.\n"
        '* "pt" will load the weights in the pytorch bin format.\n'
        '* "safetensors" will load the weights in the safetensors format.\n'
        '* "npcache" will load the weights in pytorch format and store '
        "a numpy cache to speed up the loading.\n"
        '* "dummy" will initialize the weights with random values, '
        "which is mainly for profiling.\n"
        '* "tensorizer" will load the weights using tensorizer from '
        "CoreWeave. See the Tensorize vLLM Model script in the Examples "
        "section for more information.\n"
        '* "bitsandbytes" will load the weights using bitsandbytes '
        "quantization.\n",
    )

    config_format: Optional[str] = Field(
        None,
        description="The config format which shall be loaded.  Defaults to 'auto' which defaults to 'hf'.",
    )

    tokenizer_mode: Optional[str] = Field(
        None,
        description="Tokenizer mode. 'auto' will use the fast tokenizer if"
        "available, 'slow' will always use the slow tokenizer, and"
        "'mistral' will always use the tokenizer from `mistral_common`.",
    )

    limit_mm_per_prompt: Optional[str] = Field(
        None,
        description="Maximum number of data instances per modality per prompt. Only applicable for multimodal models.",
    )

    max_num_batched_tokens: Optional[int] = Field(
        None, description="Maximum number of batched tokens per iteration"
    )

    tokenizer: Optional[str] = Field(
        None,
        description="Name or path of the huggingface tokenizer to use.",
    )

    dtype: Optional[str] = Field(
        None,
        description="Data type for model weights and activations. The 'auto' option will use FP16 precision for FP32 and FP16 models, and BF16 precision for BF16 models.",
    )

    seed: Optional[int] = Field(
        None,
        description="Random seed for reproducibility.",
    )

    revision: Optional[str] = Field(
        None,
        description="The specific model version to use. It can be a branch name, a tag name, or a commit id. If unspecified, will use the default version.",
    )

    code_revision: Optional[str] = Field(
        None,
        description="The specific revision to use for the model code on Hugging Face Hub. It can be a branch name, a tag name, or a commit id. If unspecified, will use the default version.",
    )

    rope_scaling: Optional[Dict[str, Any]] = Field(
        None,
        description="Dictionary containing the scaling configuration for the RoPE embeddings. When using this flag, don't update `max_position_embeddings` to the expected new maximum.",
    )

    tokenizer_revision: Optional[str] = Field(
        None,
        description="The specific tokenizer version to use. It can be a branch name, a tag name, or a commit id. If unspecified, will use the default version.",
    )

    quantization_param_path: Optional[str] = Field(
        None,
        description="Path to JSON file containing scaling factors. Used to load KV cache scaling factors into the model when KV cache type is FP8_E4M3 on ROCm (AMD GPU). In the future these will also be used to load activation and weight scaling factors when the model dtype is FP8_E4M3 on ROCm.",
    )

    max_seq_len_to_capture: Optional[int] = Field(
        None,
        description="Maximum sequence len covered by CUDA graphs. When a sequence has context length larger than this, we fall back to eager mode. Additionally for encoder-decoder models, if the sequence length of the encoder input is larger than this, we fall back to the eager mode.",
    )

    disable_sliding_window: Optional[bool] = Field(
        None,
        description="Whether to disable sliding window. If True, we will disable the sliding window functionality of the model. If the model does not support sliding window, this argument is ignored.",
    )

    skip_tokenizer_init: Optional[bool] = Field(
        None,
        description="If true, skip initialization of tokenizer and detokenizer.",
    )

    served_model_name: Optional[str] = Field(
        None,
        description="The model name used in metrics tag `model_name`, matches the model name exposed via the APIs. If multiple model names provided, the first name will be used. If not specified, the model name will be the same as `model`.",
    )

    override_neuron_config: Optional[Dict[str, Any]] = Field(
        None,
        description="Initialize non default neuron config or override default neuron config that are specific to Neuron devices, this argument will be used to configure the neuron config that can not be gathered from the vllm arguments.",
    )

    mm_processor_kwargs: Optional[Dict[str, Any]] = Field(
        None,
        description="Arguments to be forwarded to the model's processor for multi-modal data, e.g., image processor.",
    )

    # cache configs
    block_size: Optional[int] = Field(
        None,
        description="Size of a cache block in number of tokens.",
    )
    gpu_memory_utilization: Optional[float] = Field(
        None,
        description="Fraction of GPU memory to use for the vLLM execution.",
    )
    swap_space: Optional[float] = Field(
        None,
        description="Size of the CPU swap space per GPU (in GiB).",
    )
    cache_dtype: Optional[str] = Field(
        None,
        description="Data type for kv cache storage.",
    )
    num_gpu_blocks_override: Optional[int] = Field(
        None,
        description="Number of GPU blocks to use. This overrides the profiled num_gpu_blocks if specified. Does nothing if None.",
    )
    enable_prefix_caching: Optional[bool] = Field(
        None,
        description="Enables automatic prefix caching.",
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
    top_k: Optional[int] = Field(
        None,
        ge=-1,
        description="Controls the number of top tokens to consider. -1 means consider all tokens.",
    )
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
    include_stop_str_in_output: Optional[bool] = Field(
        None,
        description="""Whether to include the stop strings in
            output text. Defaults to False.""",
    )
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
    echo: Optional[bool] = Field(
        default=None,
        description=(
            "If true, the new message will be prepended with the last message "
            "if they belong to the same role."
        ),
    )
    add_generation_prompt: Optional[bool] = Field(
        default=None,
        description=(
            "If true, the generation prompt will be added to the chat template. "
            "This is a parameter used by chat template in tokenizer config of the "
            "model."
        ),
    )
    continue_final_message: Optional[bool] = Field(
        default=None,
        description=(
            "If this is set, the chat will be formatted so that the final "
            "message in the chat is open-ended, without any EOS tokens. The "
            "model will continue this message rather than starting a new one. "
            'This allows you to "prefill" part of the model\'s response for it. '
            "Cannot be used at the same time as `add_generation_prompt`."
        ),
    )
    add_special_tokens: Optional[bool] = Field(
        default=None,
        description=(
            "If true, special tokens (e.g. BOS) will be added to the prompt "
            "on top of what is added by the chat template. "
            "For most models, the chat template takes care of adding the "
            "special tokens so this should be set to false (as is the "
            "default)."
        ),
    )
    documents: Optional[List[Dict[str, str]]] = Field(
        default=None,
        description=(
            "A list of dicts representing documents that will be accessible to "
            "the model if it is performing RAG (retrieval-augmented generation)."
            " If the template does not support RAG, this argument will have no "
            "effect. We recommend that each document should be a dict containing "
            '"title" and "text" keys.'
        ),
    )
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
    priority: Optional[int] = Field(
        default=None,
        description=(
            "The priority of the request (lower means earlier handling; "
            "default: 0). Any priority other than 0 will raise an error "
            "if the served model does not use priority scheduling."
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
