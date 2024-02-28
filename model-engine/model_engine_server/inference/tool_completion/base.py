class BaseTool:
    """
    Base class for third-party tools.
    """

    def __call__(self, expression: str) -> str:
        """
        Call method to be overridden by child classes.
        """
        raise NotImplementedError("The evaluate method must be implemented by child classes.")
