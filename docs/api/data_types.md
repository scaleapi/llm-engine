# 🐍 Python Client Data Type Reference

::: llmengine.CompletionOutput
    selection:
        members:
            - text
            - num_prompt_tokens
            - num_completion_tokens

::: llmengine.CompletionStreamOutput
    selection:
        members:
            - text
            - finished
            - num_prompt_tokens
            - num_completion_tokens

::: llmengine.CompletionSyncResponse
    selection:
        members:
            - request_id
            - output

::: llmengine.CompletionStreamResponse
    selection:
        members:
            - request_id
            - output

::: llmengine.CreateFineTuneResponse
    selection:
        members:
            - id

::: llmengine.GetFineTuneResponse
    selection:
        members:
            - id
            - fine_tuned_model

::: llmengine.ListFineTunesResponse
    selection:
        members:
            - jobs

::: llmengine.CancelFineTuneResponse
    selection:
        members:
            - success

::: llmengine.GetLLMEndpointResponse
    selection:
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
    selection:
        members:
            - model_endpoints

::: llmengine.DeleteLLMEndpointResponse
    selection:
        members:
            - deleted

::: llmengine.ModelDownloadRequest
    selection:
        members:
            - model_name
            - download_format

::: llmengine.ModelDownloadResponse
    selection:
        members:
            - urls

::: llmengine.UploadFileResponse
    selection:
        members:
            - id

::: llmengine.GetFileResponse
    selection:
        members:
            - id
            - filename
            - size

::: llmengine.GetFileContentResponse
    selection:
        members:
            - id
            - content

::: llmengine.ListFilesResponse
    selection:
        members:
            - files

::: llmengine.DeleteFileResponse
    selection:
        members:
            - deleted
