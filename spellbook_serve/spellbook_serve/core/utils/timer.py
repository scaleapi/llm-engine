"""Utilities for timing code blocks."""
import inspect
import time
from datetime import timedelta
from logging import Logger
from types import FrameType
from typing import Optional


class timer:  # pylint: disable=invalid-name
    """Context manager for timing a block of code.

    Example use case -- consider timing a function `f`:

    >>> def f():
    >>>     time.sleep(1) # simulate "long computation"

    Directly use the timer and grab `duration` after the context block has finished:

    >>> with timer() as block_timer:
    >>>     f()
    >>> print(f"{block_timer.duration:0.4f}s")

    Note that the duration is in seconds. It is a `float`, so it can represent fractional seconds.

    The other use case is to pass in a `name` and a `logger`. The timing will be recorded
    when the context block is exited:

    >>> from spellbook_serve.core.loggers import make_logger
    >>>
    >>> log = make_logger("my-main-program")
    >>>
    >>> with timer(logger=log, name="timing-func-f"):
    >>>     f()
    """

    __slots__ = ("logger", "name", "_duration", "start")

    def __init__(self, logger: Optional[Logger] = None, name: str = "") -> None:
        self.logger = logger
        self.name = name
        self._duration: Optional[float] = None
        # for start, -1 is the uninitialized value
        # it is set at the context-block entering method: __enter__
        self.start: float = -1.0

    def __enter__(self) -> "timer":
        """Records start time: context-block entering function."""
        self.start = time.monotonic()
        # avoid any code execution *after* recording start
        return self

    def __exit__(self, *args) -> None:
        """Records end time: context-block exiting function."""
        # calculate the duration *first*
        # CRITICAL: do not introduce any additional latency in this timing measurement
        self._duration = time.monotonic() - self.start
        # i.e. validation occurs after
        if self.start == -1:
            raise ValueError(
                "Cannot use context-block exit method if context-block enter method has not been "
                "called!"
            )
        self._maybe_log_end_time()

    def _maybe_log_end_time(self) -> None:
        if self.logger is not None:
            caller_namespace = "<unknown_caller_namespace>"
            frame: Optional[FrameType] = inspect.currentframe()
            if frame is not None:
                frame = frame.f_back
                if frame is not None:
                    caller_namespace = frame.f_globals["__name__"]
            metric_name = f"timer.{caller_namespace}"
            if self.name:
                metric_name = f"{metric_name}.{self.name}"
            msg = f"{metric_name} - {self._duration:5.2f}s"
            self.logger.info(msg, stacklevel=2)

    @property
    def duration(self) -> float:
        """The number of seconds from when the context block was entered until it was exited.

        Raises ValueError if the context block was either:
            - never entered
            - entered but not exited
        """
        if self._duration is None:
            raise ValueError("Cannot get duration if timer has not exited context block!")
        return self._duration

    @property
    def timedelta(self) -> timedelta:
        """Formats the duration as a datetime timedelta object.

        Raises ValueError on same conditions as `duration`.
        """
        return timedelta(seconds=self.duration)

    def __format__(self, format_spec: str) -> str:
        """String representation of the timer's duration, using the supplied formatting
        specification.
        """
        return f"{self.duration:{format_spec}}"

    #
    # It is often necessary to compute statistics on a collection of timers.
    # e.g. "What's the median time? Average time?" etc.
    #

    def __float__(self) -> float:
        """Alias to the timer's duration. See :func:`duration` for specification."""
        return self.duration

    def __int__(self) -> int:
        """Rounds the duration to the nearest second."""
        return int(round(float(self)))

    #
    # Implementations for builtins numeric operations.
    #

    def __eq__(self, other) -> bool:
        return float(self) == float(other)

    def __lt__(self, other) -> bool:
        return float(self) < float(other)

    def __le__(self, other) -> bool:
        return float(self) <= float(other)

    def __gt__(self, other) -> bool:
        return float(self) > float(other)

    def __ge__(self, other) -> bool:
        return float(self) >= float(other)

    def __abs__(self) -> float:
        return abs(float(self))

    def __add__(self, other) -> float:
        return float(self) + float(other)

    def __sub__(self, other) -> float:
        return float(self) - float(other)

    def __mul__(self, other) -> float:
        return float(self) * float(other)

    def __floordiv__(self, other) -> int:
        return int(self) // int(other)

    def __truediv__(self, other) -> float:
        return float(self) / float(other)
