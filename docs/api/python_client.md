# 🐍 Python Client API Reference

::: spellbook_serve_client.Completion
    selection:
        members:
            - create
            - acreate

::: spellbook_serve_client.FineTune
    selection:
        members:
            - create
            - list
            - retrieve
            - cancel

::: spellbook_serve_client.Model

::: spellbook_serve_client.CompletionSyncV1Response

::: spellbook_serve_client.CompletionStreamV1Response

::: spellbook_serve_client.CompletionOutput
    selection:
        members:
            - text
            - num_prompt_tokens
            - num_completion_tokens

::: spellbook_serve_client.CompletionStreamOutput
    selection:
        members:
            - text
            - finished
            - num_prompt_tokens
            - num_completion_tokens

::: spellbook_serve_client.TaskStatus

::: spellbook_serve_client.CreateFineTuneJobRequest
    selection:
        members:
            - base_model
            - training_file
            - validation_file
            - fine_tuning_method
            - hyperparameters
            - model_name

::: spellbook_serve_client.CreateFineTuneJobResponse

::: spellbook_serve_client.GetFineTuneJobResponse

::: spellbook_serve_client.ListFineTuneJobResponse

::: spellbook_serve_client.CancelFineTuneJobResponse

