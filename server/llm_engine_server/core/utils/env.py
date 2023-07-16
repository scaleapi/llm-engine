"""Utilities for working with environment variables."""
import os
from typing import ContextManager, Dict, Optional, Sequence, Union

__all__: Sequence[str] = ("environment",)


class environment(ContextManager):
    """Context manager for temporarily setting environment variables. Resets env var state on completion.

    **WARNING**: Mutates the `os.environ` environment variable mapping in the :func:`__enter__`
                 and :func:`__exit__` methods. Every `__enter__` **MUST** be followed-up by an `__exit__`.

    NOTE: The suggested use pattern is to make a new :class:`environment` for each time one
          requires the functionality. It is possible, however, to create one :class:`environment`
          instance and re-use to temporarily set the same environment variable state.

    For example, this is the most common use case:
    >>> with environment(ENV_VAR='your-value'):
    >>>     # do something that needs this ENV_VAR set to 'your-value'
    >>>     ...
    >>> # Environment variable "ENV_VAR" is reset to its prior value.
    >>> # If there was not a previously set value for "ENV_VAR", then it is unset now.

    You can also use `environment` to temporarily ensure that an environment variable is not set:
    >>>> with environment(ENV_VAR=None):
    >>>>   # There is no longer any environment variable value for "ENV_VAR"
    >>>>   ...
    >>>> # If there was one beforehand, the environment variable "ENV_VAR" is reset.
    >>>> # Otherwise, it is still unset.

    """

    def __init__(self, **env_vars) -> None:
        """Keep track of the temporary values one will assign to a set of environment variables.

        NOTE: Either supply string-valued keyword arguments or a dictionary of env var names to their values.
              Environment variables must be either `str` or `int` valued.
              Raises :class:`ValueError` if either of these conditions are invalid.
        """
        # used in __enter__ and __exit__: keeps track of prior existing environment
        # variable settings s.t. they can be reset when the context block is finished
        self.__prior_env_var_setting: Dict[str, Optional[Union[str, int]]] = {}

        # we store the user's set of desired temporary values for environment variables
        # we also validate these settings to a minimum bar of correctness
        self.__new_env_var_settings: Dict[str, Optional[str]] = {}
        for env, val in env_vars.items():
            if not isinstance(env, str) or len(env) == 0:
                raise ValueError(f"Need valid env var name, not ({type(env)}) '{env}'")
            self.__new_env_var_settings[env] = val if val is None else str(val)

    def __enter__(self) -> "environment":
        """Temporarily sets user's supplied environment variables.

        Records the previous values of all environment variables to be set.

        WARNING: Mutates internal state.
        """
        # get the existing values for all environment variables to be temporarily set
        # set the env vars to their desired values too
        for env, val in self.__new_env_var_settings.items():
            # track prior value and set new for environment variable
            prior: Optional[str] = os.environ.get(env)
            self.__prior_env_var_setting[env] = prior
            if val is not None:
                # we're setting a new value for the environment variable, env
                os.environ[env] = val
            elif env in os.environ:
                # otherwise, if env=None, then we want to remove env from the
                # internal environment variable state
                del os.environ[env]
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Restores the previous values of all temporarily set environment variables.

        WARNING: Mutates internal state.

        NOTE: Ignores all input arguments.
        """
        # restore all env vars
        for env, prior in self.__prior_env_var_setting.items():
            # If there _was_ a prior value, we'll always set it here. This is the same if
            # we were setting it to some other string or if we wanted the env var
            # temporarily gone (env=None in the __init__).
            if prior is not None:
                # restore previous environment variable value
                os.environ[env] = prior
            else:
                # If there was no prior value for the env var, then we need to determine
                # if we had set it to something new in the context. Or, if the env var
                # didn't exist in the first place.
                if env in os.environ:
                    # If there's a current env value, then it must be the one that we set
                    # in __enter__. Therefore, we want to remove it here to restore the
                    # previous env var state (in which env was never set).
                    del os.environ[env]
                # If the env var isn't currently in the environment variables state,
                # then we had requested it temporarily unset in the context without
                # the caller realizing that env was never set beforehand.
                # Thus, this "reset" action for this case is equivalent to a no-op.
        # forget about previous context setting
        # could be used for another __enter__ --> __exit__ cycle, if desired
        self.__prior_env_var_setting.clear()
