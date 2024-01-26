# 🐍 Python Client Data Type Reference

::: llmengine.CompletionOutput
    options:
        members:
            - text
            - num_prompt_tokens
            - num_completion_tokens

::: llmengine.CompletionStreamOutput
    options:
        members:
            - text
            - finished
            - num_prompt_tokens
            - num_completion_tokens

::: llmengine.CompletionSyncResponse
    options:
        members:
            - request_id
            - output

::: llmengine.CompletionStreamResponse
    options:
        members:
            - request_id
            - output

::: llmengine.CreateFineTuneResponse
    options:
        members:
            - id

::: llmengine.GetFineTuneResponse
    options:
        members:
            - id
            - fine_tuned_model

::: llmengine.ListFineTunesResponse
    options:
        members:
            - jobs

::: llmengine.CancelFineTuneResponse
    options:
        members:
            - success

::: llmengine.GetLLMEndpointResponse
    options:
        members:
            - name
            - source
            - inference_framework
            - id
            - model_name
            - status
            - inference_framework_tag
            - num_shards
            - quantize
            - spec

::: llmengine.ListLLMEndpointsResponse
    options:
        members:
            - model_endpoints

::: llmengine.DeleteLLMEndpointResponse
    options:
        members:
            - deleted

::: llmengine.ModelDownloadRequest
    options:
        members:
            - model_name
            - download_format

::: llmengine.ModelDownloadResponse
    options:
        members:
            - urls

::: llmengine.UploadFileResponse
    options:
        members:
            - id

::: llmengine.GetFileResponse
    options:
        members:
            - id
            - filename
            - size

::: llmengine.GetFileContentResponse
    options:
        members:
            - id
            - content

::: llmengine.ListFilesResponse
    options:
        members:
            - files

::: llmengine.DeleteFileResponse
    options:
        members:
            - deleted

::: llmengine.CreateBatchCompletionsRequestContent
    options:
        members:
            - prompts
            - max_new_tokens
            - temperature
            - stop_sequences
            - return_token_log_probs
            - presence_penalty
            - frequency_penalty
            - top_k
            - top_p

::: llmengine.CreateBatchCompletionsModelConfig
    options:
        members:
            - model
            - checkpoint_path
            - labels
            - num_shards
            - quantize
            - seed

::: llmengine.CreateBatchCompletionsRequest
    options:
        members:
            - input_data_path
            - output_data_path
            - content
            - model_config
            - data_parallelism
            - max_runtime_sec

::: llmengine.CreateBatchCompletionsResponse
    options:
        members:
            - job_id
