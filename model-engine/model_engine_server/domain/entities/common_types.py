from typing import Any, Dict, Union

CpuSpecificationType = Union[str, int, float]
StorageSpecificationType = Union[str, int, float]  # TODO(phil): we can make this more specific.
FineTuneHparamValueType = Union[
    str, int, float, Dict[str, Any]
]  # should just make this Any if we need to add more
