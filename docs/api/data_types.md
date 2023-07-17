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

::: llmengine.CompletionSyncV1Response

::: llmengine.CompletionStreamV1Response

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
            - id
            - name
            - model_name
            - source
            - inference_framework
            - num_shards

::: llmengine.ListLLMEndpointsResponse

::: llmengine.DeleteLLMEndpointResponse
