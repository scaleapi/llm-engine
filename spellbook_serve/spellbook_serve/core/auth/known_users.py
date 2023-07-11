from typing import Mapping

TEAM_TO_USER_ID_MAPPING: Mapping[str, str] = {
    "catalog": "63bf16970c6cc48a1cc1a260",
    "content_understanding": "63bf16bdc65a8abb84ad8615",
    "cvml": "63bf16d877db8a663d044aa3",
    "document": "63bf16e7d175bec460c602a7",
    "enterprise_ai": "63e6ba88dbe7b93adf245a7a",
    "federal": "63bf16fbc65a8abb84ad8619",
    "infra": "63bf1714faad99edd6e93f39",
    "instantml": "63bf172377db8a663d044aa7",
    "nucleus": "63bf1753342ca13a2479c11f",
    "rapid": "63bf1768de9df5183d1740b4",
    "spellbook": "63bf1777e487096169d50214",
    "studio": "63bf1785311cdb2853df2c10",
    "egp": "64962c7fa6d0ff5ff7261ef4",
}

USER_ID_TO_TEAM_MAPPING: Mapping[str, str] = {v: k for k, v in TEAM_TO_USER_ID_MAPPING.items()}

ALLOWED_TEAMS = set(TEAM_TO_USER_ID_MAPPING.keys())
