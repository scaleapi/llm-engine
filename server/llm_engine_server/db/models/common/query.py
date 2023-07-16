from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict


@dataclass
class Query:
    def to_sqlalchemy_query(self) -> Dict[str, Any]:
        """
        Convert the query to a dictionary of kwargs that can be used in a sqlalchemy query
        """
        return {key: value for key, value in vars(self).items() if value is not None}
