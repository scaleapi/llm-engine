from typing import Dict


def dict_not_none(**kwargs) -> Dict:
    return {k: v for k, v in kwargs.items() if v is not None}
