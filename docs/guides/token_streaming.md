The Completions APIs support a `stream` boolean parameter that, when `True`, will return a streamed response of
token-by-token server-sent events (SSEs) rather than waiting to receive the full response when model generation has
finished. This decreases latency of when you start getting a response.

The response will consist of SSEs of the form `{"token": dict, "generated_text": str | null, "details": dict | null}`,
where the dictionary for each token will contain log probability information in addition to the generated string; the
`generated_text` field will be `null` for all but the last SSE, for which it will contain the full generated response.
