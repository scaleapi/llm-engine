import json
import os
from typing import AsyncIterator, Iterator, Optional

import requests
from aiohttp import BasicAuth, ClientSession
from pydantic import ValidationError
from spellbook_serve_client.errors import parse_error
from spellbook_serve_client.types import (
    CompletionStreamV1Request,
    CompletionStreamV1Response,
    CompletionSyncV1Request,
    CompletionSyncV1Response,
)


class Client:
    """Client to make calls to a spellbook-serve-client instance

     Example:

     ```python
     >>> from spellbook_serve_client import Client

     >>> client = Client("https://api.spellbook.scale.com", "flan-t5-xxl-deepspeed-sync")
     >>> client.generate("Why is the sky blue?").outputs[0].text
     ' Rayleigh scattering'

     >>> result = ""
     >>> for response in client.generate_stream("Why is the sky blue?"):
     >>>     if response.output:
     >>>         result += response.output.text
     >>> result
    ' Rayleigh scattering'
     ```
    """

    def __init__(
        self,
        base_url: str,
        model_name: str,
        api_key: Optional[str] = None,
        timeout: int = 10,
    ):
        """
        Args:
            base_url (`str`):
                spellbook-serve-client instance base url
            model_name (`str`):
                Model name to use for inference
            api_key (`str`):
                API key to use for authentication
            timeout (`int`):
                Timeout in seconds
        """
        self.base_url = base_url
        self.model_name = model_name
        self.sync_inference_url = os.path.join(
            self.base_url, f"v1/llm/completions-sync?model_endpoint_name={self.model_name}"
        )
        self.stream_inference_url = os.path.join(
            self.base_url, f"v1/llm/completions-stream?model_endpoint_name={self.model_name}"
        )
        self.timeout = timeout
        if api_key is not None:
            self.api_key = api_key
        else:
            env_api_key = os.getenv("SCALE_API_KEY")
            if env_api_key is None:
                self.api_key = "root"
            else:
                self.api_key = env_api_key

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 20,
        temperature: float = 0.2,
    ) -> CompletionSyncV1Response:
        """
        Given a prompt, generate the following text

        Args:
            prompt (`str`):
                Input text
            max_new_tokens (`int`):
                Maximum number of generated tokens
            temperature (`float`):
                The value used to module the logits distribution.

        Returns:
            CompletionSyncV1Response: generated response
        """
        # Validate parameters
        request = CompletionSyncV1Request(
            prompts=[prompt], max_new_tokens=max_new_tokens, temperature=temperature
        )

        resp = requests.post(
            self.sync_inference_url,
            json=request.dict(),
            auth=(self.api_key, ""),
            timeout=self.timeout,
        )
        payload = resp.json()
        if resp.status_code != 200:
            raise parse_error(resp.status_code, payload)
        return CompletionSyncV1Response.parse_obj(payload)

    def generate_stream(
        self,
        prompt: str,
        max_new_tokens: int = 20,
        temperature: float = 0.2,
    ) -> Iterator[CompletionStreamV1Response]:
        """
        Given a prompt, generate the following stream of tokens

        Args:
            prompt (`str`):
                Input text
            max_new_tokens (`int`):
                Maximum number of generated tokens
            temperature (`float`):
                The value used to module the logits distribution.

        Returns:
            Iterator[CompletionStreamV1Response]: stream of generated tokens
        """
        request = CompletionStreamV1Request(
            prompt=prompt, max_new_tokens=max_new_tokens, temperature=temperature
        )

        resp = requests.post(
            self.stream_inference_url,
            json=request.dict(),
            auth=(self.api_key, ""),
            timeout=self.timeout,
            stream=True,
        )

        if resp.status_code != 200:
            raise parse_error(resp.status_code, resp.json())

        # Parse ServerSentEvents
        for byte_payload in resp.iter_lines():
            # Skip line
            if byte_payload == b"\n":
                continue

            payload = byte_payload.decode("utf-8")

            # Event data
            if payload.startswith("data:"):
                # Decode payload
                payload_data = payload.lstrip("data:").rstrip("/n")
                try:
                    # Parse payload
                    response = CompletionStreamV1Response.parse_raw(payload_data)
                except ValidationError:
                    # If we failed to parse the payload, then it is an error payload
                    raise parse_error(resp.status_code, json.loads(payload_data))
                yield response


class AsyncClient:
    """Asynchronous Client to make calls to a spellbook-serve-client instance

     Example:

     ```python
     >>> from spellbook_serve_client import AsyncClient

     >>> client = AsyncClient("https://api.spellbook.scale.com", "flan-t5-xxl-deepspeed-sync")
     >>> response = await client.generate("Why is the sky blue?")
     >>> response.outputs[0].text
     ' Rayleigh scattering'

     >>> result = ""
     >>> async for response in client.generate_stream("Why is the sky blue?"):
     >>>     if response.output:
     >>>         result += response.output.text
     >>> result
    ' Rayleigh scattering'
     ```
    """

    def __init__(
        self,
        base_url: str,
        model_name: str,
        api_key: Optional[str] = None,
        timeout: int = 10,
    ):
        """
        Args:
            base_url (`str`):
                spellbook-serve-client instance base url
            model_name (`str`):
                Model name to use for inference
            api_key (`str`):
                API key to use for authentication
            timeout (`int`):
                Timeout in seconds
        """
        self.base_url = base_url
        self.model_name = model_name
        self.sync_inference_url = os.path.join(
            self.base_url, f"v1/llm/completions-sync?model_endpoint_name={self.model_name}"
        )
        self.stream_inference_url = os.path.join(
            self.base_url, f"v1/llm/completions-stream?model_endpoint_name={self.model_name}"
        )
        self.timeout = timeout
        if api_key is not None:
            self.api_key = api_key
        else:
            env_api_key = os.getenv("SCALE_API_KEY")
            if env_api_key is None:
                self.api_key = "root"
            else:
                self.api_key = env_api_key

    async def generate(
        self,
        prompt: str,
        max_new_tokens: int = 20,
        temperature: float = 0.2,
    ) -> CompletionSyncV1Response:
        """
        Given a prompt, generate the following text asynchronously

        Args:
            prompt (`str`):
                Input text
            max_new_tokens (`int`):
                Maximum number of generated tokens
            temperature (`float`):
                The value used to module the logits distribution.

        Returns:
            CompletionSyncV1Response: generated response
        """
        request = CompletionSyncV1Request(
            prompts=[prompt], max_new_tokens=max_new_tokens, temperature=temperature
        )

        async with ClientSession(
            timeout=self.timeout, auth=BasicAuth(login=self.api_key)
        ) as session:
            async with session.post(self.sync_inference_url, json=request.dict()) as resp:
                payload = await resp.json()

                if resp.status != 200:
                    raise parse_error(resp.status, payload)
                return CompletionSyncV1Response(**payload[0])

    async def generate_stream(
        self,
        prompt: str,
        max_new_tokens: int = 20,
        temperature: float = 0.2,
    ) -> AsyncIterator[CompletionStreamV1Response]:
        """
        Given a prompt, generate the following stream of tokens asynchronously

        Args:
            prompt (`str`):
                Input text
            max_new_tokens (`int`):
                Maximum number of generated tokens
            temperature (`float`):
                The value used to module the logits distribution.

        Returns:
            AsyncIterator[CompletionStreamV1Response]: stream of generated tokens
        """
        request = CompletionStreamV1Request(
            prompt=prompt, max_new_tokens=max_new_tokens, temperature=temperature
        )

        async with ClientSession(
            timeout=self.timeout, auth=BasicAuth(login=self.api_key)
        ) as session:
            async with session.post(self.stream_inference_url, json=request.dict()) as resp:

                if resp.status != 200:
                    raise parse_error(resp.status, await resp.json())

                # Parse ServerSentEvents
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
                            # Parse payload
                            response = CompletionStreamV1Response.parse_raw(payload_data)
                        except ValidationError:
                            # If we failed to parse the payload, then it is an error payload
                            raise parse_error(resp.status, json.loads(payload_data))
                        yield response
