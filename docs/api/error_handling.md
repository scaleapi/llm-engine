# Error handling

LLM Engine uses conventional HTTP response codes to indicate the success or failure of an API request. In general:
codes in the `2xx` range indicate success. Codes in the `4xx` range indicate indicate an error that failed given the 
information provided (e.g. a given Model was not found, or an invalid temperature was specified). Codes in the `5xx` 
range indicate an error with the LLM Engine servers.

In the Python client, errors are presented via a set of corresponding Exception classes, which should be caught 
and handled by the user accordingly.

::: llmengine.errors.BadRequestError

::: llmengine.errors.UnauthorizedError

::: llmengine.errors.NotFoundError

::: llmengine.errors.RateLimitExceededError

::: llmengine.errors.ServerError
