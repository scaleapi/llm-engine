from typing import List, Optional

from model_engine_server.common.pydantic_types import BaseModel, Field


# Last synced 01/26/2025 from https://github.com/sgl-project/sglang/blob/9472e69963283160b51a617431e936f23910443a/python/sglang/srt/server_args.py
class SGLangModelConfig(BaseModel):
    trust_remote_code: Optional[bool] = Field(
        default=False,
        description="Whether to trust remote code from Hugging face hub. This is only applicable to models whose code is not supported natively by the transformers library (e.g. deepseek). Default to False.",
    )
    tp_size: Optional[int] = Field(
        default=None,
        description="The tensor parallel size.",
    )
    skip_tokenizer_init: Optional[bool] = Field(
        default=None,
        description="If set, skip init tokenizer and pass input_ids in generate request",
    )
    load_format: Optional[str] = Field(
        default=None,
        description="The format of the model weights to load.",
    )
    dtype: Optional[str] = Field(
        default=None,
        description="Data type for model weights and activations.",
    )
    kv_cache_dtype: Optional[str] = Field(
        default=None,
        description='Data type for kv cache storage. "auto" will use model data type.',
    )
    quantization_param_path: Optional[str] = Field(
        default=None,
        description="Path to the JSON file containing the KV cache scaling factors.",
    )
    quantization: Optional[str] = Field(
        default=None,
        description="The quantization method.",
    )
    context_length: Optional[int] = Field(
        default=None,
        description="The model's maximum context length.",
    )
    device: Optional[str] = Field(
        default=None,
        description="The device type.",
    )
    served_model_name: Optional[str] = Field(
        default=None,
        description="Override the model name returned by the v1/models endpoint in OpenAI API server.",
    )
    chat_template: Optional[str] = Field(
        default=None,
        description="The builtin chat template name or path of the chat template file.",
    )
    is_embedding: Optional[bool] = Field(
        default=None,
        description="Whether to use a CausalLM as an embedding model.",
    )
    revision: Optional[str] = Field(
        default=None,
        description="The specific model version to use.",
    )
    mem_fraction_static: Optional[float] = Field(
        default=None,
        description="The fraction of the memory used for static allocation.",
    )
    max_running_requests: Optional[int] = Field(
        default=None,
        description="The maximum number of running requests.",
    )
    max_total_tokens: Optional[int] = Field(
        default=None,
        description="The maximum number of tokens in the memory pool.",
    )
    chunked_prefill_size: Optional[int] = Field(
        default=None,
        description="The maximum number of tokens in a chunk for the chunked prefill.",
    )
    max_prefill_tokens: Optional[int] = Field(
        default=None,
        description="The maximum number of tokens in a prefill batch.",
    )
    schedule_policy: Optional[str] = Field(
        default=None,
        description="The scheduling policy of the requests.",
    )
    schedule_conservativeness: Optional[float] = Field(
        default=None,
        description="How conservative the schedule policy is.",
    )
    cpu_offload_gb: Optional[int] = Field(
        default=None,
        description="How many GBs of RAM to reserve for CPU offloading",
    )
    prefill_only_one_req: Optional[bool] = Field(
        default=None,
        description="If true, we only prefill one request at one prefill batch",
    )
    stream_interval: Optional[int] = Field(
        default=None,
        description="The interval for streaming in terms of the token length.",
    )
    random_seed: Optional[int] = Field(
        default=None,
        description="The random seed.",
    )
    constrained_json_whitespace_pattern: Optional[str] = Field(
        default=None,
        description="Regex pattern for syntactic whitespaces allowed in JSON constrained output.",
    )
    watchdog_timeout: Optional[float] = Field(
        default=None,
        description="Set watchdog timeout in seconds.",
    )
    download_dir: Optional[str] = Field(
        default=None,
        description="Model download directory.",
    )
    base_gpu_id: Optional[int] = Field(
        default=None,
        description="The base GPU ID to start allocating GPUs from.",
    )
    log_level: Optional[str] = Field(
        default=None,
        description="The logging level of all loggers.",
    )
    log_level_http: Optional[str] = Field(
        default=None,
        description="The logging level of HTTP server.",
    )
    log_requests: Optional[bool] = Field(
        default=None,
        description="Log the inputs and outputs of all requests.",
    )
    show_time_cost: Optional[bool] = Field(
        default=None,
        description="Show time cost of custom marks.",
    )
    enable_metrics: Optional[bool] = Field(
        default=None,
        description="Enable log prometheus metrics.",
    )
    decode_log_interval: Optional[int] = Field(
        default=None,
        description="The log interval of decode batch.",
    )
    api_key: Optional[str] = Field(
        default=None,
        description="Set API key of the server.",
    )
    file_storage_pth: Optional[str] = Field(
        default=None,
        description="The path of the file storage in backend.",
    )
    enable_cache_report: Optional[bool] = Field(
        default=None,
        description="Return number of cached tokens in usage.prompt_tokens_details.",
    )
    data_parallel_size: Optional[int] = Field(
        default=None,
        description="The data parallelism size.",
    )
    load_balance_method: Optional[str] = Field(
        default=None,
        description="The load balancing strategy for data parallelism.",
    )
    expert_parallel_size: Optional[int] = Field(
        default=None,
        description="The expert parallelism size.",
    )
    dist_init_addr: Optional[str] = Field(
        default=None,
        description="The host address for initializing distributed backend.",
    )
    nnodes: Optional[int] = Field(
        default=None,
        description="The number of nodes.",
    )
    node_rank: Optional[int] = Field(
        default=None,
        description="The node rank.",
    )
    json_model_override_args: Optional[str] = Field(
        default=None,
        description="A dictionary in JSON string format used to override default model configurations.",
    )
    lora_paths: Optional[List[str]] = Field(
        default=None,
        description="The list of LoRA adapters.",
    )
    max_loras_per_batch: Optional[int] = Field(
        default=None,
        description="Maximum number of adapters for a running batch.",
    )
    attention_backend: Optional[str] = Field(
        default=None,
        description="Choose the kernels for attention layers.",
    )
    sampling_backend: Optional[str] = Field(
        default=None,
        description="Choose the kernels for sampling layers.",
    )
    grammar_backend: Optional[str] = Field(
        default=None,
        description="Choose the backend for grammar-guided decoding.",
    )
    speculative_algorithm: Optional[str] = Field(
        default=None,
        description="Speculative algorithm.",
    )
    speculative_draft_model_path: Optional[str] = Field(
        default=None,
        description="The path of the draft model weights.",
    )
    speculative_num_steps: Optional[int] = Field(
        default=None,
        description="The number of steps sampled from draft model in Speculative Decoding.",
    )
    speculative_num_draft_tokens: Optional[int] = Field(
        default=None,
        description="The number of token sampled from draft model in Speculative Decoding.",
    )
    speculative_eagle_topk: Optional[int] = Field(
        default=None,
        description="The number of token sampled from draft model in eagle2 each step.",
    )
    enable_double_sparsity: Optional[bool] = Field(
        default=None,
        description="Enable double sparsity attention",
    )
    ds_channel_config_path: Optional[str] = Field(
        default=None,
        description="The path of the double sparsity channel config",
    )
    ds_heavy_channel_num: Optional[int] = Field(
        default=None,
        description="The number of heavy channels in double sparsity attention",
    )
    ds_heavy_token_num: Optional[int] = Field(
        default=None,
        description="The number of heavy tokens in double sparsity attention",
    )
    ds_heavy_channel_type: Optional[str] = Field(
        default=None,
        description="The type of heavy channels in double sparsity attention",
    )
    ds_sparse_decode_threshold: Optional[int] = Field(
        default=None,
        description="The threshold for sparse decoding in double sparsity attention",
    )
    disable_radix_cache: Optional[bool] = Field(
        default=None,
        description="Disable RadixAttention for prefix caching.",
    )
    disable_jump_forward: Optional[bool] = Field(
        default=None,
        description="Disable jump-forward for grammar-guided decoding.",
    )
    disable_cuda_graph: Optional[bool] = Field(
        default=None,
        description="Disable cuda graph.",
    )
    disable_cuda_graph_padding: Optional[bool] = Field(
        default=None,
        description="Disable cuda graph when padding is needed.",
    )
    disable_outlines_disk_cache: Optional[bool] = Field(
        default=None,
        description="Disable disk cache of outlines.",
    )
    disable_custom_all_reduce: Optional[bool] = Field(
        default=None,
        description="Disable the custom all-reduce kernel.",
    )
    disable_mla: Optional[bool] = Field(
        default=None,
        description="Disable Multi-head Latent Attention (MLA) for DeepSeek-V2.",
    )
    disable_overlap_schedule: Optional[bool] = Field(
        default=None,
        description="Disable the overlap scheduler.",
    )
    enable_mixed_chunk: Optional[bool] = Field(
        default=None,
        description="Enable mixing prefill and decode in a batch when using chunked prefill.",
    )
    enable_dp_attention: Optional[bool] = Field(
        default=None,
        description="Enable data parallelism for attention and tensor parallelism for FFN.",
    )
    enable_ep_moe: Optional[bool] = Field(
        default=None,
        description="Enable expert parallelism for moe.",
    )
    enable_torch_compile: Optional[bool] = Field(
        default=None,
        description="Optimize the model with torch.compile.",
    )
    torch_compile_max_bs: Optional[int] = Field(
        default=None,
        description="Set the maximum batch size when using torch compile.",
    )
    cuda_graph_max_bs: Optional[int] = Field(
        default=None,
        description="Set the maximum batch size for cuda graph.",
    )
    cuda_graph_bs: Optional[List[int]] = Field(
        default=None,
        description="Set the list of batch sizes for cuda graph.",
    )
    torchao_config: Optional[str] = Field(
        default=None,
        description="Optimize the model with torchao.",
    )
    enable_nan_detection: Optional[bool] = Field(
        default=None,
        description="Enable the NaN detection for debugging purposes.",
    )
    enable_p2p_check: Optional[bool] = Field(
        default=None,
        description="Enable P2P check for GPU access.",
    )
    triton_attention_reduce_in_fp32: Optional[bool] = Field(
        default=None,
        description="Cast the intermediate attention results to fp32.",
    )
    triton_attention_num_kv_splits: Optional[int] = Field(
        default=None,
        description="The number of KV splits in flash decoding Triton kernel.",
    )
    num_continuous_decode_steps: Optional[int] = Field(
        default=None,
        description="Run multiple continuous decoding steps to reduce scheduling overhead.",
    )
    delete_ckpt_after_loading: Optional[bool] = Field(
        default=None,
        description="Delete the model checkpoint after loading the model.",
    )
    enable_memory_saver: Optional[bool] = Field(
        default=None,
        description="Allow saving memory using release_memory_occupation and resume_memory_occupation",
    )
    allow_auto_truncate: Optional[bool] = Field(
        default=None,
        description="Allow automatically truncating requests that exceed the maximum input length.",
    )
    enable_custom_logit_processor: Optional[bool] = Field(
        default=None,
        description="Enable users to pass custom logit processors to the server.",
    )
    tool_call_parser: Optional[str] = Field(
        default=None,
        description="Specify the parser for handling tool-call interactions.",
    )


class SGLangEndpointAdditionalArgs(SGLangModelConfig, BaseModel):
    pass
