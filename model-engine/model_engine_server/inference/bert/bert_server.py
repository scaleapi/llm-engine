import time
from typing import List
from tqdm import tqdm
from transformers import AutoTokenizer, pipeline
from fastapi import FastAPI, HTTPException, Request, Response
from pydantic import BaseModel
import uvicorn
import subprocess
from logging import Logger
from fastapi import HTTPException
import torch
from optimum.onnxruntime import ORTModelForSequenceClassification
from transformers import (
    AutoTokenizer,
    pipeline,
    PreTrainedModel,
    PreTrainedTokenizer,
    Pipeline,
)

import onnxruntime as ort


logger = Logger("vllm_server")

TIMEOUT_KEEP_ALIVE = 5  # seconds.
TIMEOUT_TO_PREVENT_DEADLOCK = 1  # seconds
BATCH_SIZE=20

# Initialize FastAPI app
app = FastAPI()


class InputText(BaseModel):
    text: str


class InputBatch(BaseModel):
    text: List[str]


@app.post("/predict-batch")
def predict_batch(req: InputBatch) -> Response:
    start = time.time()
    print("here")
    global classifier

    try:
        for out in tqdm(classifier(req.text, batch_size=BATCH_SIZE), total=len(req.text)):
            print(f"TTFT: {time.time() - start}")
            print(out)
            yield {"prediction": out}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict")
def predict(req: InputText) -> Response:
    start = time.time()
    print("here")
    global classifier

    try:
        result = classifier(req.text)
        print(result)
        return {"prediction": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def get_gpu_free_memory():
    """Get GPU free memory using nvidia-smi."""
    try:
        output = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
        ).stdout
        gpu_memory = [int(x) for x in output.strip().split("\n")]
        return gpu_memory
    except Exception as e:
        logger.warn(f"Error getting GPU memory: {e}")
        return None


def check_unknown_startup_memory_usage():
    """Check for unknown memory usage at startup."""
    gpu_free_memory = get_gpu_free_memory()
    if gpu_free_memory is not None:
        min_mem = min(gpu_free_memory)
        max_mem = max(gpu_free_memory)
        if max_mem - min_mem > 10:
            logger.warn(
                f"WARNING: Unbalanced GPU memory usage at start up. This may cause OOM. Memory usage per GPU in MB: {gpu_free_memory}."
            )
            try:
                # nosemgrep
                output = subprocess.run(
                    ["fuser -v /dev/nvidia*"],
                    shell=False,
                    capture_output=True,
                    text=True,
                ).stdout
                logger.info(f"Processes using GPU: {output}")
            except Exception as e:
                logger.error(f"Error getting processes using GPU: {e}")


def parse_args():
    import argparse

    parser = argparse.ArgumentParser(description="Run a VLLM server.")
    parser.add_argument(
        "--host",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
    )
    parser.add_argument(
        "--uvicorn-log-level",
        type=str,
        default="info",
    )

    parser.add_argument("--model", type=str, default="model_files")
    return parser.parse_args()


def init_classifier():
    from optimum.onnxruntime import ORTModelForSequenceClassification
    from transformers import AutoTokenizer, pipeline

    tokenizer = AutoTokenizer.from_pretrained(
        "ProtectAI/deberta-v3-base-prompt-injection", subfolder="onnx"
    )
    tokenizer.model_input_names = ["input_ids", "attention_mask"]
    model = ORTModelForSequenceClassification.from_pretrained(
        "ProtectAI/deberta-v3-base-prompt-injection",
        export=False,
        subfolder="onnx",
        file_name="model.onnx",
    )

    classifier = pipeline(
        task="text-classification",
        model=model,
        tokenizer=tokenizer,
        truncation=True,
        max_length=512 * BATCH_SIZE,
    )

    return classifier


if __name__ == "__main__":
    check_unknown_startup_memory_usage()

    args = parse_args()

    global classifier
    classifier = init_classifier()
    print(classifier("hello world"))

    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level="debug",
        timeout_keep_alive=TIMEOUT_KEEP_ALIVE,
    )
