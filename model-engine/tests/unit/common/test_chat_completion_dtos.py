import json

import pytest
from model_engine_server.common.dtos.llms.chat_completion import (
    ChatCompletionV2Request,
    ChatCompletionV2StreamSuccessChunk,
    ChatCompletionV2SyncResponse,
)
from model_engine_server.common.pydantic_types import ValidationError

REASONING_TRACE = "The user wants 17 * 23. 17 * 23 = 391."


def _sync_response_payload(message: dict) -> dict:
    return {
        "id": "chatcmpl-123",
        "object": "chat.completion",
        "created": 1700000000,
        "model": "gemma-4-31b-it",
        "choices": [
            {
                "index": 0,
                "finish_reason": "stop",
                "logprobs": None,
                "message": message,
            }
        ],
        "usage": {"prompt_tokens": 30, "completion_tokens": 83, "total_tokens": 113},
    }


def _stream_chunk_payload(delta: dict) -> dict:
    return {
        "id": "chatcmpl-123",
        "object": "chat.completion.chunk",
        "created": 1700000000,
        "model": "gemma-4-31b-it",
        "choices": [{"index": 0, "finish_reason": None, "delta": delta}],
    }


def test_sync_response_maps_vllm_reasoning_to_reasoning_content():
    """vLLM reasoning parsers return the trace under `reasoning`."""
    payload = _sync_response_payload(
        {"role": "assistant", "content": "391", "reasoning": REASONING_TRACE}
    )
    response = ChatCompletionV2SyncResponse.model_validate(payload)

    assert response.choices[0].message.reasoning_content == REASONING_TRACE

    dumped = json.loads(response.model_dump_json(exclude_none=True))
    assert dumped["choices"][0]["message"]["reasoning_content"] == REASONING_TRACE
    assert "reasoning" not in dumped["choices"][0]["message"]


def test_sync_response_accepts_reasoning_content_key():
    payload = _sync_response_payload(
        {"role": "assistant", "content": "391", "reasoning_content": REASONING_TRACE}
    )
    response = ChatCompletionV2SyncResponse.model_validate(payload)

    assert response.choices[0].message.reasoning_content == REASONING_TRACE


def test_sync_response_without_reasoning_omits_key_with_exclude_none():
    payload = _sync_response_payload({"role": "assistant", "content": "391"})
    response = ChatCompletionV2SyncResponse.model_validate(payload)

    assert response.choices[0].message.reasoning_content is None
    dumped = json.loads(response.model_dump_json(exclude_none=True))
    assert "reasoning_content" not in dumped["choices"][0]["message"]
    assert "reasoning" not in dumped["choices"][0]["message"]


def test_sync_response_without_reasoning_serializes_stable_null():
    """The sync route returns the model through FastAPI without exclude_none, so
    non-reasoning models serialize `reasoning_content: null` alongside the other
    nullable spec fields (`refusal`, `audio`, ...) the route already emits."""
    payload = _sync_response_payload({"role": "assistant", "content": "391"})
    response = ChatCompletionV2SyncResponse.model_validate(payload)

    message = json.loads(response.model_dump_json())["choices"][0]["message"]
    assert "reasoning_content" in message
    assert message["reasoning_content"] is None
    # Pre-existing nullable spec field, serialized the same way.
    assert "refusal" in message
    assert message["refusal"] is None


def test_stream_chunk_maps_vllm_reasoning_to_reasoning_content():
    chunk = ChatCompletionV2StreamSuccessChunk.model_validate(
        _stream_chunk_payload({"role": "assistant", "reasoning": REASONING_TRACE})
    )

    assert chunk.choices[0].delta.reasoning_content == REASONING_TRACE

    dumped = json.loads(chunk.model_dump_json(exclude_none=True))
    assert dumped["choices"][0]["delta"]["reasoning_content"] == REASONING_TRACE
    assert "reasoning" not in dumped["choices"][0]["delta"]


def test_stream_chunk_without_reasoning_is_unchanged():
    chunk = ChatCompletionV2StreamSuccessChunk.model_validate(
        _stream_chunk_payload({"role": "assistant", "content": "391"})
    )

    assert chunk.choices[0].delta.reasoning_content is None
    dumped = json.loads(chunk.model_dump_json(exclude_none=True))
    assert "reasoning_content" not in dumped["choices"][0]["delta"]


def test_request_assistant_message_forwards_reasoning_content_as_reasoning():
    """Clients send `reasoning_content` back; the inference framework expects `reasoning`."""
    request = ChatCompletionV2Request.model_validate(
        {
            "model": "gemma-4-31b-it",
            "messages": [
                {"role": "user", "content": "What is 17*23?"},
                {
                    "role": "assistant",
                    "content": "391",
                    "reasoning_content": REASONING_TRACE,
                },
                {"role": "user", "content": "Are you sure?"},
            ],
        }
    )

    dumped = request.model_dump(exclude_none=True)
    assistant_message = dumped["messages"][1]
    assert assistant_message["reasoning"] == REASONING_TRACE
    assert "reasoning_content" not in assistant_message


def test_request_assistant_message_accepts_reasoning_key():
    request = ChatCompletionV2Request.model_validate(
        {
            "model": "gemma-4-31b-it",
            "messages": [
                {"role": "assistant", "content": "391", "reasoning": REASONING_TRACE},
            ],
        }
    )

    dumped = request.model_dump(exclude_none=True)
    assert dumped["messages"][0]["reasoning"] == REASONING_TRACE


def test_request_without_reasoning_is_unchanged():
    request = ChatCompletionV2Request.model_validate(
        {
            "model": "gemma-4-31b-it",
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "What is 17*23?"},
                {"role": "assistant", "content": "391"},
            ],
        }
    )

    dumped = request.model_dump(exclude_none=True)
    for message in dumped["messages"]:
        assert "reasoning" not in message
        assert "reasoning_content" not in message


def test_request_requires_at_least_one_message():
    """The `messages` override re-declares the spec's `min_length=1`: pydantic field
    overrides replace the parent FieldInfo rather than merging with it, so the
    constraint must be restated or it is silently lost."""
    with pytest.raises(ValidationError):
        ChatCompletionV2Request.model_validate({"model": "gemma-4-31b-it", "messages": []})


def test_request_tool_call_round_trip_with_reasoning():
    """Assistant tool-call turns are where reasoning round-tripping matters most."""
    request = ChatCompletionV2Request.model_validate(
        {
            "model": "gemma-4-31b-it",
            "messages": [
                {"role": "user", "content": "Edit the doc."},
                {
                    "role": "assistant",
                    "reasoning_content": REASONING_TRACE,
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "apply_mutations", "arguments": "{}"},
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": "call_1", "content": "ok"},
            ],
        }
    )

    dumped = request.model_dump(exclude_none=True)
    assistant_message = dumped["messages"][1]
    assert assistant_message["reasoning"] == REASONING_TRACE
    assert assistant_message["tool_calls"][0]["function"]["name"] == "apply_mutations"
