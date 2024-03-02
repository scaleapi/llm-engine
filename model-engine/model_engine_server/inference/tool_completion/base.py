from typing import Optional, Tuple


class BaseTool:
    """
    Base class for third-party tools.
    """

    tool_context_start = ""
    tool_call_token = ""
    tool_context_end = ""

    def __call__(self, expression: str, past_context: Optional[str]) -> Tuple[str, int]:
        """
        Call method to be overridden by child classes.
        """
        raise NotImplementedError("The evaluate method must be implemented by child classes.")
