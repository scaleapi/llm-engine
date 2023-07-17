# Overview

**The Completions API** is used to generate text completions using the underlying language model. It supports both synchronous and asynchronous operations and offers an option to stream token responses.

The API provides two key methods:

1. **create()**: Synchronous method for creating a completion task.
2. **acreate()**: Asynchronous method for creating a completion task.

Both methods return either a **CompletionSyncV1Response** object or a stream of **CompletionStreamV1Response** objects, depending on the stream parameter.

# Method Details
**1. Create()**

This method is used to create a completion task synchronously.

Parameters:
* model_name (str): The model name to use for inference.
* prompt (str): The input text.
* max_new_tokens (int, optional): The maximum number of tokens to generate. Default is 20.
* temperature (float, optional): The value used to module the logits distribution. Default is 0.2.
* timeout (int, optional): Timeout in seconds. Default is 10.
* stream (bool, optional): Whether to stream the response. If true, the return type is an Iterator[CompletionStreamV1Response]. Default is False.

Returns:
* Union[CompletionSyncV1Response, Iterator[CompletionStreamV1Response]]: Generated response or iterator of response chunks.

**2. acreate()**

This method is used to create a completion task asynchronously.

**Parameters:**
* model_name (str): The model name to use for inference.
* prompt (str): The input text.
* max_new_tokens (int, optional): The maximum number of tokens to generate. Default is 20.
* temperature (float, optional): The value used to module the logits distribution. Default is 0.2.
* timeout (int, optional): Timeout in seconds. Default is 10.
* stream (bool, optional): Whether to stream the response. If true, the return type is an AsyncIterable[CompletionStreamV1Response]. Default is False.

**Returns:**
Union[CompletionSyncV1Response, AsyncIterable[CompletionStreamV1Response]]: Generated response or iterator of response chunks.

