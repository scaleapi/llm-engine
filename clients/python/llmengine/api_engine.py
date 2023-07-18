# NOTICE - per Apache 2.0 license:
# This file was copied and modified from the OpenAI Python client library: https://github.com/openai/openai-python
import json
import os
from functools import wraps
from typing import Any, AsyncIterable, Dict, Iterator, Optional

import requests
from aiohttp import ClientSession, ClientTimeout
from llmengine.errors import parse_error

SPELLBOOK_API_URL = "https://api.spellbook.scale.com"
LLM_ENGINE_BASE_PATH = os.getenv("LLM_ENGINE_BASE_PATH", SPELLBOOK_API_URL)
DEFAULT_TIMEOUT: int = 10

api_key = os.getenv("SCALE_API_KEY")


def get_api_key() -> str:
    print(f"noglobal api_key={api_key}")
    return api_key


def assert_self_hosted(func):
    @wraps(func)
    def inner(*args, **kwargs):
        if SPELLBOOK_API_URL == LLM_ENGINE_BASE_PATH:
            raise ValueError("This feature is only available for self-hosted users.")
        return func(*args, **kwargs)

    return inner


class APIEngine:
    @classmethod
    def validate_api_key(cls):
        if SPELLBOOK_API_URL == LLM_ENGINE_BASE_PATH and not SCALE_API_KEY:
            raise ValueError(
                "You must set SCALE_API_KEY in your environment to to use the LLM Engine API."
            )

    @classmethod
    def _get(cls, resource_name: str, timeout: int) -> Dict[str, Any]:
        api_key = get_api_key()
        response = requests.get(
            os.path.join(LLM_ENGINE_BASE_PATH, resource_name),
            timeout=timeout,
            headers={"x-api-key": api_key},
        )
        if response.status_code != 200:
            raise parse_error(response.status_code, response.content)
        payload = response.json()
        return payload

    @classmethod
    def put(
        cls, resource_name: str, data: Optional[Dict[str, Any]], timeout: int
    ) -> Dict[str, Any]:
        api_key = get_api_key()
        response = requests.put(
            os.path.join(LLM_ENGINE_BASE_PATH, resource_name),
            json=data,
            timeout=timeout,
            headers={"x-api-key": api_key},
        )
        if response.status_code != 200:
            raise parse_error(response.status_code, response.content)
        payload = response.json()
        return payload

    @classmethod
    def _delete(cls, resource_name: str, timeout: int) -> Dict[str, Any]:
        api_key = get_api_key()
        response = requests.delete(
            os.path.join(LLM_ENGINE_BASE_PATH, resource_name),
            timeout=timeout,
            headers={"x-api-key": api_key},
        )
        if response.status_code != 200:
            raise parse_error(response.status_code, response.content)
        payload = response.json()
        return payload

    @classmethod
    def post_sync(cls, resource_name: str, data: Dict[str, Any], timeout: int) -> Dict[str, Any]:
        api_key = get_api_key()
        response = requests.post(
            os.path.join(LLM_ENGINE_BASE_PATH, resource_name),
            json=data,
            timeout=timeout,
            headers={"x-api-key": api_key},
        )
        if response.status_code != 200:
            raise parse_error(response.status_code, response.content)
        payload = response.json()
        return payload

    @classmethod
    def post_stream(
        cls, resource_name: str, data: Dict[str, Any], timeout: int
    ) -> Iterator[Dict[str, Any]]:
        api_key = get_api_key()
        response = requests.post(
            os.path.join(LLM_ENGINE_BASE_PATH, resource_name),
            json=data,
            timeout=timeout,
            headers={"x-api-key": api_key},
            stream=True,
        )
        if response.status_code != 200:
            raise parse_error(response.status_code, response.content)
        for byte_payload in response.iter_lines():
            # Skip line
            if byte_payload == b"\n":
                continue

            payload = byte_payload.decode("utf-8")

            # Event data
            if payload.startswith("data:"):
                # Decode payload
                payload_data = payload.lstrip("data:").rstrip("/n")
                try:
                    payload_json = json.loads(payload_data)
                    yield payload_json
                except json.JSONDecodeError:
                    raise ValueError(f"Invalid JSON payload: {payload_data}")

    @classmethod
    async def apost_sync(
        cls, resource_name: str, data: Dict[str, Any], timeout: int
    ) -> Dict[str, Any]:
        api_key = get_api_key()
        async with ClientSession(
            timeout=ClientTimeout(timeout), headers={"x-api-key": api_key}
        ) as session:
            async with session.post(
                os.path.join(LLM_ENGINE_BASE_PATH, resource_name), json=data
            ) as resp:
                if resp.status != 200:
                    raise parse_error(resp.status, await resp.read())
                payload = await resp.json()
                return payload

    @classmethod
    async def apost_stream(
        cls, resource_name: str, data: Dict[str, Any], timeout: int
    ) -> AsyncIterable[Dict[str, Any]]:
        api_key = get_api_key()
        async with ClientSession(
            timeout=ClientTimeout(timeout), headers={"x-api-key": api_key}
        ) as session:
            async with session.post(
                os.path.join(LLM_ENGINE_BASE_PATH, resource_name), json=data
            ) as resp:
                if resp.status != 200:
                    raise parse_error(resp.status, await resp.read())
                async for byte_payload in resp.content:
                    # Skip line
                    if byte_payload == b"\n":
                        continue

                    payload = byte_payload.decode("utf-8")

                    # Event data
                    if payload.startswith("data:"):
                        # Decode payload
                        payload_data = payload.lstrip("data:").rstrip("/n")
                        try:
                            response = json.loads(payload_data)
                            yield response
                        except json.JSONDecodeError:
                            raise ValueError(f"Invalid JSON payload: {payload_data}")
