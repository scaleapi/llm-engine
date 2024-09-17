from typing import Sequence

from fastapi import APIRouter

from .batch_completion import batch_completions_router_v2
from .chat_completion import chat_router_v2

llm_router_v2 = APIRouter(prefix="/v2")
llm_router_v2.include_router(batch_completions_router_v2)
llm_router_v2.include_router(chat_router_v2)

__all__: Sequence[str] = ("llm_router_v2",)
