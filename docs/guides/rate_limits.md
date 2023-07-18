# Overview

## What are rate limits?

A rate limit is a restriction that an API imposes on the number of times a user or client can access the server within a specified period of time.

## How do I know if I am rate limited?

Per standard HTTP practices, your request will receive a response with HTTP status code of `429`, `Too Many Requests`.

## What are the rate limits for our API?

The LLM Engine API is currently in a preview mode, and therefore we currently do not have any advertised rate limits.
As the API moves towards a production release, we will update this section with specific rate limits. For now, the API
will return HTTP 429 on an as-needed basis.

# Error mitigation

## Retrying with exponential backoff

One easy way to avoid rate limit errors is to automatically retry requests with a random exponential backoff. 
Retrying with exponential backoff means performing a short sleep when a rate limit error is hit, then retrying the 
unsuccessful request. If the request is still unsuccessful, the sleep length is increased and the process is repeated. 
This continues until the request is successful or until a maximum number of retries is reached. This approach has many benefits:

* Automatic retries means you can recover from rate limit errors without crashes or missing data
* Exponential backoff means that your first retries can be tried quickly, while still benefiting from longer delays if your first few retries fail
* Adding random jitter to the delay helps retries from all hitting at the same time.

Below are a few example solutions **for Python** that use exponential backoff.

### Example #1: Using the `tenacity` library

Tenacity is an Apache 2.0 licensed general-purpose retrying library, written in Python, to simplify the task of adding 
retry behavior to just about anything. To add exponential backoff to your requests, you can use the tenacity.retry 
decorator. The below example uses the tenacity.wait_random_exponential function to add random exponential backoff to a 
request.

=== "Exponential backoff in python"
```python
import llmengine
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)  # for exponential backoff

@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def completion_with_backoff(**kwargs):
    return llmengine.Completion.create(**kwargs)

print(  
    completion_with_backoff(
        model="llama-7b", 
        prompt="Why is the sky blue?", 
        max_new_tokens=100, 
        temperature=0.2
    ).output.text
)
```

### Example #2: Using the `backoff` library

[Backoff](https://github.com/litl/backoff) is another python library that provides function decorators which can be used to wrap a function such that it will be retried until some condition is met. 

=== "Decorators for backoff and retry in python"
```python
import llmengine
import backoff

@backoff.on_exception(backoff.expo, llmengine.errors.RateLimitExceededError)

def completion_with_backoff(**kwargs):
    return llmengine.Completion.create(**kwargs)

print(completion_with_backoff(model="llama-7b", 
    prompt="Why is the sky blue?", 
    max_new_tokens=100, 
    temperature=0.2).output.text)

```
