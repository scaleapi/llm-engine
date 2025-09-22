import argparse
import json
from typing import AsyncGenerator, Optional, List

import uvicorn
from fastapi import BackgroundTasks, FastAPI, Request
from fastapi.responses import Response, StreamingResponse
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.sampling_params import SamplingParams
from vllm.utils import random_uuid
from pydantic import BaseModel, Field
from transformers import AutoTokenizer


TIMEOUT_KEEP_ALIVE = 5  # seconds.
TIMEOUT_TO_PREVENT_DEADLOCK = 1  # seconds
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
app = FastAPI()

class CompletionRequest(BaseModel):
    temperature: float = Field(default=0.2, ge=0.0, le=1.0)
    stop_sequences: Optional[List[str]] = Field(default=None, max_items=4)
    max_new_tokens: Optional[int] = Field(default=None)
    prompt: Optional[str] = Field(default=None)
    prompts: Optional[List[str]] = Field(default=None)

class CompletionResponse(BaseModel):
    completions: List[tuple]
    prompt_tokens: int
    response_tokens: int

@app.get("/readyz")
@app.get("/healthz") 
@app.get("/health")
def healthcheck():
    return "OK"

@app.post("/predict")
async def generate_egp_completions(request: Request) -> Response:
    request_id = random_uuid()
    request_dict = await request.json()

    chat_history = request_dict.pop("prompt", None)
    if chat_history:
        try:
            messages = json.loads(chat_history)
        except:
            messages = [chat_history]
    
    # Handle parameter name differences
    if "max_new_tokens" in request_dict:
        request_dict["max_tokens"] = request_dict.pop("max_new_tokens")

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )

    sampling_params = SamplingParams(**request_dict)
    results_generator = engine.generate(text, sampling_params, request_id)

    final_output = None
    tokens = []
    last_output_text = ""
    async for request_output in results_generator:
        tokens.append(request_output.outputs[-1].text[len(last_output_text) :])
        last_output_text = request_output.outputs[-1].text
        if await request.is_disconnected():
            await engine.abort(request_id)
            return Response(status_code=499)
        final_output = request_output

    assert final_output is not None

    ret = {
        "text": final_output.outputs[0].text,
        "count_prompt_tokens": len(final_output.prompt_token_ids),
        "count_output_tokens": len(final_output.outputs[0].token_ids),
        "log_probs": final_output.outputs[0].logprobs,
        "tokens": tokens,
        "finish_reason": None,
    }
    # Return Model-Engine compatible format
    return Response(
        content=json.dumps({
            "text": ret["text"],
            "count_prompt_tokens": ret["count_prompt_tokens"],
            "count_output_tokens": ret["count_output_tokens"],
            "log_probs": ret["log_probs"],
            "tokens": ret["tokens"],
            "finish_reason": ret["finish_reason"]
        })
    )

@app.post("/stream")
async def generate(request: Request) -> Response:
    request_dict = await request.json()
    prompt = request_dict.pop("prompt")
    stream = request_dict.pop("stream", False)
    
    # Handle parameter name differences
    if "max_new_tokens" in request_dict:
        request_dict["max_tokens"] = request_dict.pop("max_new_tokens")
        
    sampling_params = SamplingParams(**request_dict)
    request_id = random_uuid()
    results_generator = engine.generate(prompt, sampling_params, request_id)

    async def stream_results() -> AsyncGenerator[str, None]:
        last_output_text = ""
        async for request_output in results_generator:
            ret = {
                "text": request_output.outputs[-1].text[len(last_output_text) :],
                "count_prompt_tokens": len(request_output.prompt_token_ids),
                "count_output_tokens": len(request_output.outputs[0].token_ids),
                "log_probs": request_output.outputs[0].logprobs[-1]
                if sampling_params.logprobs
                else None,
                "finished": request_output.finished,
            }
            last_output_text = request_output.outputs[-1].text
            yield f"data:{json.dumps(ret)}\
\
"

    async def abort_request() -> None:
        await engine.abort(request_id)

    if stream:
        background_tasks = BackgroundTasks()
        background_tasks.add_task(abort_request)
        return StreamingResponse(stream_results(), background=background_tasks)

    final_output = None
    tokens = []
    last_output_text = ""
    async for request_output in results_generator:
        tokens.append(request_output.outputs[-1].text[len(last_output_text) :])
        last_output_text = request_output.outputs[-1].text
        if await request.is_disconnected():
            await engine.abort(request_id)
            return Response(status_code=499)
        final_output = request_output

    assert final_output is not None
    prompt = final_output.prompt
    ret = {
        "text": final_output.outputs[0].text,
        "count_prompt_tokens": len(final_output.prompt_token_ids),
        "count_output_tokens": len(final_output.outputs[0].token_ids),
        "log_probs": final_output.outputs[0].logprobs,
        "tokens": tokens,
    }
    return Response(content=json.dumps(ret))

if __name__ == "__main__":
    engine_args = AsyncEngineArgs(model="model_files", enforce_eager=True, tensor_parallel_size=1, gpu_memory_utilization=0.9)
    engine = AsyncLLMEngine.from_engine_args(engine_args)

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=5005,
        log_level="debug",
        timeout_keep_alive=TIMEOUT_KEEP_ALIVE,
    )
