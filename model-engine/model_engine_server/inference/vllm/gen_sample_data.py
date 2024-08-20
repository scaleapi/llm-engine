import json

EXAMPLES_DIR = "examples/v2"

messages = [
    {
        "role": "user",
        "content": "What is a good place for travel in the US?",
    },
    {
        "role": "assistant",
        "content": "California.",
    },
    {
        "role": "user",
        "content": "What can I do in California?",
    },
]

if __name__ == "__main__":

    completion_type = "chat"
    model = "gemma"
    target_file = f"{EXAMPLES_DIR}/sample_data_{completion_type}_{model}.json"

    # request = CompletionCreateParamsNonStreaming(
    #     messages=messages,
    #     logprobs=True,
    #     max_tokens=300,
    # )
    request = {
        "messages": messages,
        "logprobs": True,
        "max_tokens": 300,
    }
    json.dump([request], open(target_file, "w"))
