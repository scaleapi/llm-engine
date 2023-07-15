# 🐍 Python Client API Reference

::: llmengine.Completion
	selection:
        members:
            - create
            - acreate

::: llmengine.FineTune
    selection:
        members:
            - create
            - list
            - retrieve
            - cancel

::: llmengine.Model

::: llmengine.CompletionSyncV1Response

::: llmengine.CompletionStreamV1Response

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

::: llmengine.TaskStatus

::: llmengine.CreateFineTuneJobRequest
    selection:
        members:
            - base_model
            - training_file
            - validation_file
            - fine_tuning_method
            - hyperparameters
            - model_name

::: llmengine.CreateFineTuneJobResponse

::: llmengine.GetFineTuneJobResponse

::: llmengine.ListFineTuneJobResponse

::: llmengine.CancelFineTuneJobResponse

