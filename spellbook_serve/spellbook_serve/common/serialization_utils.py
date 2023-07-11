import base64
import json
from typing import Any, Dict, List, Optional, Union

JSON = Union[List[str], Dict[str, Any], str, float, bool, int]


def python_json_to_b64(python_json: Optional[JSON]) -> str:
    return str_to_b64(json.dumps(python_json))


def b64_to_python_json(b64text: str) -> Optional[Dict[str, Any]]:
    return json.loads(b64_to_str(b64text))


def str_to_b64(raw_str: str) -> str:
    return base64.b64encode(raw_str.encode("utf-8")).decode("utf-8")


def b64_to_str(b64text: str) -> str:
    return base64.b64decode(b64text.encode("utf-8")).decode("utf-8")


def str_to_bool(bool_text: Optional[str]) -> Optional[bool]:
    if bool_text is None:
        return None
    return bool_text.lower() in ("yes", "y", "true", "t", "1")
    pass


def bool_to_str(bool_val: Optional[bool]) -> Optional[str]:
    if bool_val is None:
        return None
    return "true" if bool_val else "false"
