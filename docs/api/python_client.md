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
            - num_prompt_tokens
            - num_completion_tokens

::: llmengine.CompletionSyncV1Response
    selection:
        members:
            - status
            - outputs

::: llmengine.CompletionStreamOutput
    selection:
        members:
            - text
            - finished
            - num_prompt_tokens
            - num_completion_tokens


::: llmengine.CompletionStreamV1Response
    selection:
        members:
            - status
            - output

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

::: llmengine.TaskStatus

