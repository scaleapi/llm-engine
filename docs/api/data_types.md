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

::: llmengine.CompletionStreamResponse

::: llmengine.CreateFineTuneRequest
    selection:
        members:
            - model
            - training_file
            - validation_file
            - hyperparameters
            - suffix

::: llmengine.CreateFineTuneResponse

::: llmengine.GetFineTuneResponse

::: llmengine.ListFineTunesResponse

::: llmengine.CancelFineTuneResponse

::: llmengine.GetLLMEndpointResponse
    selection:
        members:
            - name
            - source
            - inference_framework
            - id
            - model_name
            - inference_framework_tag
            - num_shards
            - quantize
            - spec

::: llmengine.ListLLMEndpointsResponse

::: llmengine.DeleteLLMEndpointResponse
