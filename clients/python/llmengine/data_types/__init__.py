"""
DTOs for LLM APIs.
"""

from typing_extensions import TypeAlias

from .batch_completion import *  # noqa: F403
from .chat_completion import *  # noqa: F403
from .completion import *  # noqa: F403
from .rest import *  # noqa: F403

# Alias for backwards compatibility
CreateBatchCompletionsRequestContent: TypeAlias = (
    CreateBatchCompletionsV1RequestContent  # noqa: F405
)
CreateBatchCompletionsRequest: TypeAlias = CreateBatchCompletionsV1Request  # noqa: F405
CreateBatchCompletionsResponse: TypeAlias = CreateBatchCompletionsV1Response  # noqa: F405
CreateBatchCompletionsModelConfig: TypeAlias = CreateBatchCompletionsV1ModelConfig  # noqa: F405

CompletionSyncRequest: TypeAlias = CompletionSyncV1Request  # noqa: F405
CompletionSyncResponse: TypeAlias = CompletionSyncV1Response  # noqa: F405
CompletionStreamRequest: TypeAlias = CompletionStreamV1Request  # noqa: F405
CompletionStreamResponse: TypeAlias = CompletionStreamV1Response  # noqa: F405
