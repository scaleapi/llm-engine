from typing import Union

CpuSpecificationType = Union[str, int, float]
StorageSpecificationType = Union[str, int, float]  # TODO(phil): we can make this more specific.
FineTuneHparamValueType = Union[str, int, float]  # should suffice for now
