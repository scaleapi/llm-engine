# 🐍 Python Client API Reference

::: llmengine.Completion
    selection:
        members:
            - create
            - acreate

::: llmengine.CompletionOutput
    selection:
        members:
            - text
            - num_completion_tokens

::: llmengine.CompletionSyncV1Response

::: llmengine.CompletionStreamOutput
    selection:
        members:
            - text
            - finished
            - num_completion_tokens


::: llmengine.CompletionStreamV1Response

::: llmengine.FineTune
    selection:
        members:
            - create
            - list
            - retrieve
            - cancel

::: llmengine.CreateFineTuneResponse

::: llmengine.GetFineTuneResponse

::: llmengine.ListFineTunesResponse

::: llmengine.CancelFineTuneResponse
