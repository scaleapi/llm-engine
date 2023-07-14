import json
import os
from functools import wraps
from typing import Any, AsyncIterable, Dict, Iterator, Optional

import requests
from aiohttp import BasicAuth, ClientSession, ClientTimeout
from spellbook_serve_client.errors import parse_error

SCALE_API_KEY = os.getenv("SCALE_API_KEY")
SPELLBOOK_API_URL = "https://api.spellbook.scale.com"
SPELLBOOK_SERVE_BASE_PATH = os.getenv("SPELLBOOK_SERVE_BASE_PATH", SPELLBOOK_API_URL)
DEFAULT_TIMEOUT: int = 10


def get_api_key() -> str:
    return SCALE_API_KEY or "root"


def assert_self_hosted(func):
    @wraps(func)
    def inner(*args, **kwargs):
        if SPELLBOOK_API_URL == SPELLBOOK_SERVE_BASE_PATH:
            raise ValueError(
                "This feature is only available for self-hosted users."
            )
        return func(*args, **kwargs)

    return inner


class APIEngine:
    @classmethod
    def validate_api_key(cls):
        if SPELLBOOK_API_URL == SPELLBOOK_SERVE_BASE_PATH and not SCALE_API_KEY:
            raise ValueError(
                "You must set SCALE_API_KEY in your environment to to use the Spellbook Serve API."
            )

    @classmethod
    def get(cls, resource_name: str, timeout: int) -> Dict[str, Any]:
        api_key = get_api_key()
        response = requests.get(
            os.path.join(SPELLBOOK_SERVE_BASE_PATH, resource_name),
            timeout=timeout,
            auth=(api_key, ""),
        )
        payload = response.json()
        if response.status_code != 200:
            raise parse_error(response.status_code, payload)
        return payload

    @classmethod
    def put(
        cls, resource_name: str, data: Optional[Dict[str, Any]], timeout: int
    ) -> Dict[str, Any]:
        api_key = get_api_key()
        response = requests.put(
            os.path.join(SPELLBOOK_SERVE_BASE_PATH, resource_name),
            json=data,
            timeout=timeout,
            auth=(api_key, ""),
        )
        payload = response.json()
        if response.status_code != 200:
            raise parse_error(response.status_code, payload)
        return payload

    @classmethod
    def post_sync(cls, resource_name: str, data: Dict[str, Any], timeout: int) -> Dict[str, Any]:
        api_key = get_api_key()
        response = requests.post(
            os.path.join(SPELLBOOK_SERVE_BASE_PATH, resource_name),
            json=data,
            timeout=timeout,
            auth=(api_key, ""),
        )
        payload = response.json()
        if response.status_code != 200:
            raise parse_error(response.status_code, payload)
        return payload

    @classmethod
    def post_stream(
        cls, resource_name: str, data: Dict[str, Any], timeout: int
    ) -> Iterator[Dict[str, Any]]:
        api_key = get_api_key()
        response = requests.post(
            os.path.join(SPELLBOOK_SERVE_BASE_PATH, resource_name),
            json=data,
            timeout=timeout,
            auth=(api_key, ""),
            stream=True,
        )
        if response.status_code != 200:
            raise parse_error(response.status_code, response.json())
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
            timeout=ClientTimeout(timeout), auth=BasicAuth(login=api_key)
        ) as session:
            async with session.post(
                os.path.join(SPELLBOOK_SERVE_BASE_PATH, resource_name), json=data
            ) as resp:
                payload = await resp.json()

                if resp.status != 200:
                    raise parse_error(resp.status, payload)
                return payload

    @classmethod
    async def apost_stream(
        cls, resource_name: str, data: Dict[str, Any], timeout: int
    ) -> AsyncIterable[Dict[str, Any]]:
        api_key = get_api_key()
        async with ClientSession(
            timeout=ClientTimeout(timeout), auth=BasicAuth(login=api_key)
        ) as session:
            async with session.post(
                os.path.join(SPELLBOOK_SERVE_BASE_PATH, resource_name), json=data
            ) as resp:
                if resp.status != 200:
                    raise parse_error(resp.status, await resp.json())
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
