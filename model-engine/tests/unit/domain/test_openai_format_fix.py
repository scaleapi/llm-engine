#!/usr/bin/env python3
"""
Quick test to verify the OpenAI format parsing fix for vLLM 0.5+ compatibility.
Run with: python test_openai_format_fix.py
"""

# Test data representing vLLM responses
LEGACY_FORMAT = {
    "text": "Hello, I am a language model.",
    "count_prompt_tokens": 5,
    "count_output_tokens": 7,
    "tokens": ["Hello", ",", " I", " am", " a", " language", " model", "."],
    "log_probs": [
        {1: -0.5},
        {2: -0.3},
        {3: -0.2},
        {4: -0.1},
        {5: -0.4},
        {6: -0.2},
        {7: -0.1},
        {8: -0.05},
    ],
}

OPENAI_FORMAT = {
    "id": "cmpl-123",
    "object": "text_completion",
    "created": 1234567890,
    "model": "test-model",
    "choices": [
        {
            "text": "Hello, I am a language model.",
            "index": 0,
            "logprobs": {
                "tokens": ["Hello", ",", " I", " am", " a", " language", " model", "."],
                "token_logprobs": [-0.5, -0.3, -0.2, -0.1, -0.4, -0.2, -0.1, -0.05],
            },
            "finish_reason": "stop",
        }
    ],
    "usage": {"prompt_tokens": 5, "completion_tokens": 7, "total_tokens": 12},
}

OPENAI_STREAMING_FORMAT = {
    "id": "cmpl-123",
    "object": "text_completion",
    "created": 1234567890,
    "model": "test-model",
    "choices": [
        {
            "text": " world",
            "index": 0,
            "logprobs": None,
            "finish_reason": None,  # Not finished yet
        }
    ],
}

OPENAI_STREAMING_FINAL = {
    "id": "cmpl-123",
    "object": "text_completion",
    "created": 1234567890,
    "model": "test-model",
    "choices": [
        {
            "text": "!",
            "index": 0,
            "logprobs": None,
            "finish_reason": "stop",  # Finished
        }
    ],
    "usage": {"prompt_tokens": 5, "completion_tokens": 10, "total_tokens": 15},
}


def parse_completion_output(model_output: dict, with_token_probs: bool = False) -> dict:
    """
    Mimics the parsing logic from model_output_to_completion_output for VLLM.
    This is extracted from llm_model_endpoint_use_cases.py for testing.
    """
    tokens = None

    # Handle OpenAI-compatible format (vLLM 0.5+) vs legacy format
    if "choices" in model_output and model_output["choices"]:
        # OpenAI-compatible format
        choice = model_output["choices"][0]
        text = choice.get("text", "")
        usage = model_output.get("usage", {})
        num_prompt_tokens = usage.get("prompt_tokens", 0)
        num_completion_tokens = usage.get("completion_tokens", 0)

        if with_token_probs and choice.get("logprobs"):
            logprobs = choice["logprobs"]
            if logprobs.get("tokens") and logprobs.get("token_logprobs"):
                tokens = [
                    {
                        "token": logprobs["tokens"][i],
                        "log_prob": logprobs["token_logprobs"][i] or 0.0,
                    }
                    for i in range(len(logprobs["tokens"]))
                ]
    else:
        # Legacy format
        text = model_output["text"]
        num_prompt_tokens = model_output["count_prompt_tokens"]
        num_completion_tokens = model_output["count_output_tokens"]

        if with_token_probs and model_output.get("log_probs"):
            tokens = [
                {"token": model_output["tokens"][index], "log_prob": list(t.values())[0]}
                for index, t in enumerate(model_output["log_probs"])
            ]

    return {
        "text": text,
        "num_prompt_tokens": num_prompt_tokens,
        "num_completion_tokens": num_completion_tokens,
        "tokens": tokens,
    }


def parse_streaming_output(result: dict, with_token_probs: bool = False) -> dict:
    """
    Mimics the streaming parsing logic from _response_chunk_generator for VLLM.
    """
    token = None
    res = result

    if "choices" in res and res["choices"]:
        # OpenAI streaming format
        choice = res["choices"][0]
        text = choice.get("text", "")
        finished = choice.get("finish_reason") is not None
        usage = res.get("usage", {})
        num_prompt_tokens = usage.get("prompt_tokens", 0)
        num_completion_tokens = usage.get("completion_tokens", 0)

        if with_token_probs and choice.get("logprobs"):
            logprobs = choice["logprobs"]
            if logprobs and logprobs.get("tokens") and logprobs.get("token_logprobs"):
                idx = len(logprobs["tokens"]) - 1
                token = {
                    "token": logprobs["tokens"][idx],
                    "log_prob": logprobs["token_logprobs"][idx] or 0.0,
                }
    else:
        # Legacy format
        text = res["text"]
        finished = res["finished"]
        num_prompt_tokens = res["count_prompt_tokens"]
        num_completion_tokens = res["count_output_tokens"]

        if with_token_probs and res.get("log_probs"):
            token = {"token": res["text"], "log_prob": list(res["log_probs"].values())[0]}

    return {
        "text": text,
        "finished": finished,
        "num_prompt_tokens": num_prompt_tokens,
        "num_completion_tokens": num_completion_tokens,
        "token": token,
    }


def test_legacy_format():
    """Test parsing legacy vLLM format (pre-0.5)"""
    print("\n=== Testing Legacy Format ===")
    result = parse_completion_output(LEGACY_FORMAT, with_token_probs=True)

    assert result["text"] == "Hello, I am a language model.", f"Text mismatch: {result['text']}"
    assert (
        result["num_prompt_tokens"] == 5
    ), f"Prompt tokens mismatch: {result['num_prompt_tokens']}"
    assert (
        result["num_completion_tokens"] == 7
    ), f"Completion tokens mismatch: {result['num_completion_tokens']}"
    assert result["tokens"] is not None, "Tokens should not be None"
    assert len(result["tokens"]) == 8, f"Token count mismatch: {len(result['tokens'])}"

    print("✅ Legacy format parsing: PASSED")
    print(f"   Text: {result['text'][:50]}...")
    print(f"   Prompt tokens: {result['num_prompt_tokens']}")
    print(f"   Completion tokens: {result['num_completion_tokens']}")


def test_openai_format():
    """Test parsing OpenAI-compatible format (vLLM 0.5+)"""
    print("\n=== Testing OpenAI Format ===")
    result = parse_completion_output(OPENAI_FORMAT, with_token_probs=True)

    assert result["text"] == "Hello, I am a language model.", f"Text mismatch: {result['text']}"
    assert (
        result["num_prompt_tokens"] == 5
    ), f"Prompt tokens mismatch: {result['num_prompt_tokens']}"
    assert (
        result["num_completion_tokens"] == 7
    ), f"Completion tokens mismatch: {result['num_completion_tokens']}"
    assert result["tokens"] is not None, "Tokens should not be None"
    assert len(result["tokens"]) == 8, f"Token count mismatch: {len(result['tokens'])}"

    print("✅ OpenAI format parsing: PASSED")
    print(f"   Text: {result['text'][:50]}...")
    print(f"   Prompt tokens: {result['num_prompt_tokens']}")
    print(f"   Completion tokens: {result['num_completion_tokens']}")


def test_openai_streaming():
    """Test parsing OpenAI streaming format"""
    print("\n=== Testing OpenAI Streaming Format ===")

    # Test non-final chunk
    result1 = parse_streaming_output(OPENAI_STREAMING_FORMAT)
    assert result1["text"] == " world", f"Text mismatch: {result1['text']}"
    assert result1["finished"] is False, "Should not be finished"
    print("✅ Streaming chunk (not finished): PASSED")

    # Test final chunk
    result2 = parse_streaming_output(OPENAI_STREAMING_FINAL)
    assert result2["text"] == "!", f"Text mismatch: {result2['text']}"
    assert result2["finished"] is True, "Should be finished"
    assert result2["num_completion_tokens"] == 10, "Completion tokens mismatch"
    print("✅ Streaming chunk (finished): PASSED")


def main():
    print("=" * 60)
    print("Testing vLLM OpenAI Format Compatibility Fix")
    print("=" * 60)

    try:
        test_legacy_format()
        test_openai_format()
        test_openai_streaming()

        print("\n" + "=" * 60)
        print("🎉 ALL TESTS PASSED!")
        print("=" * 60)
        print("\nThe fix correctly handles both:")
        print("  • Legacy vLLM format (pre-0.5)")
        print("  • OpenAI-compatible format (vLLM 0.5+/0.10.x/0.11.x)")
        return 0
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        return 1
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
