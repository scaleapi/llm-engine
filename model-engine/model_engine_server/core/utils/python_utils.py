"""Python-language-based utility functions."""
import builtins
from importlib import import_module
from typing import Any, Optional

from model_engine_server.core.utils.format import split_module_value, strip_non_empty


def dynamic_load(module_name: str, value_name: Optional[str], validate: bool = True) -> Any:
    """Dynamically loads the Python module and (optionally) a value from it.

    Loads the module :param:`module_name`.  If :param:`value_name` is not None,
    then this function also loads the :param:`value_name` defined in the loaded module.
    """
    if validate:
        module_name = strip_non_empty(module_name, "module name")
    module = import_module(module_name)
    if value_name:
        if validate:
            value_name = strip_non_empty(value_name, "value name")
        return getattr(module, value_name)
    return module


def import_by_name(full_name: str, validate: bool = True) -> Any:
    """Dynamically load a Python value by its fully-qualified name.

    E.g. For a class `Foo` in the module `bar.baz.qux`, supplying
         `"bar.baz.qux.Foo"` will yield a reference to the Python class.
         Notably, with such a reference, we could construct a class instance
         by calling its `__init__` method. So, the following would be valid:
         ```
            foo_init = import_by_name("bar.baz.qux.Foo")
            foo_instance = foo_init()
         ```
    """
    if validate:
        full_name = strip_non_empty(full_name, "complete path name")

    maybe_builtin_type = builtins.__dict__.get(full_name, None)
    if maybe_builtin_type:
        return maybe_builtin_type

    try:
        module_name, value_name = split_module_value(full_name, validate=False)
        return dynamic_load(module_name, value_name, validate=False)
    except ValueError:
        # no '.' separator
        return import_module(full_name)
