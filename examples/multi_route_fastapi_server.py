#!/usr/bin/env python3
"""
End-to-end example demonstrating multiple routes passthrough in Launch.

This example shows how to create a FastAPI server with multiple routes and deploy it
using Launch's model endpoint creation with the passthrough forwarder.

The server implements several endpoints that would normally require the single /predict
restriction, but now can be accessed through their natural paths.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, List, Optional, Any
import uvicorn

# FastAPI server with multiple routes
app = FastAPI(title="Multi-Route Example Server", version="1.0.0")

# Data models
class PredictRequest(BaseModel):
    text: str
    model: Optional[str] = "default"

class PredictResponse(BaseModel):
    result: str
    model: str
    route: str

class HealthResponse(BaseModel):
    status: str
    routes: List[str]

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    model: Optional[str] = "gpt-3.5-turbo"
    max_tokens: Optional[int] = 100

class ChatResponse(BaseModel):
    choices: List[Dict[str, Any]]
    model: str
    usage: Dict[str, int]

class CompletionRequest(BaseModel):
    prompt: str
    model: Optional[str] = "text-davinci-003"
    max_tokens: Optional[int] = 100

class CompletionResponse(BaseModel):
    choices: List[Dict[str, str]]
    model: str
    usage: Dict[str, int]

# Health check endpoint (required by Launch)
@app.get("/health", response_model=HealthResponse)
@app.get("/readyz", response_model=HealthResponse)
def health_check():
    """Health check endpoint required by Launch forwarder."""
    return HealthResponse(
        status="healthy",
        routes=[
            "/predict",
            "/v1/chat/completions",
            "/v1/completions",
            "/analyze",
            "/custom/endpoint"
        ]
    )

# Traditional predict endpoint
@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    """Traditional ML prediction endpoint."""
    return PredictResponse(
        result=f"Processed text: {request.text}",
        model=request.model,
        route="/predict"
    )

# OpenAI-compatible chat completions endpoint
@app.post("/v1/chat/completions", response_model=ChatResponse)
def chat_completions(request: ChatRequest):
    """OpenAI-compatible chat completions endpoint."""
    # Simple echo implementation for example
    last_message = request.messages[-1] if request.messages else ChatMessage(role="user", content="")

    return ChatResponse(
        choices=[{
            "message": {
                "role": "assistant",
                "content": f"Echo: {last_message.content}"
            },
            "finish_reason": "stop",
            "index": 0
        }],
        model=request.model,
        usage={
            "prompt_tokens": len(last_message.content.split()),
            "completion_tokens": len(last_message.content.split()) + 1,
            "total_tokens": len(last_message.content.split()) * 2 + 1
        }
    )

# OpenAI-compatible completions endpoint
@app.post("/v1/completions", response_model=CompletionResponse)
def completions(request: CompletionRequest):
    """OpenAI-compatible completions endpoint."""
    return CompletionResponse(
        choices=[{
            "text": f" -> Completion for: {request.prompt}",
            "finish_reason": "stop",
            "index": 0
        }],
        model=request.model,
        usage={
            "prompt_tokens": len(request.prompt.split()),
            "completion_tokens": 10,
            "total_tokens": len(request.prompt.split()) + 10
        }
    )

# Custom analysis endpoint
@app.post("/analyze")
def analyze_text(data: Dict[str, Any]):
    """Custom text analysis endpoint."""
    text = data.get("text", "")
    if not text:
        raise HTTPException(status_code=400, detail="Text field is required")

    return {
        "analysis": {
            "word_count": len(text.split()),
            "char_count": len(text),
            "sentiment": "positive" if "good" in text.lower() else "neutral"
        },
        "text": text,
        "route": "/analyze"
    }

# Another custom endpoint
@app.get("/custom/endpoint")
def custom_endpoint():
    """A custom GET endpoint to demonstrate method flexibility."""
    return {
        "message": "This is a custom endpoint accessible via passthrough routing",
        "methods_supported": ["GET", "POST", "PUT", "DELETE", "PATCH", "HEAD", "OPTIONS"],
        "route": "/custom/endpoint"
    }

# Batch processing endpoint
@app.post("/batch/process")
def batch_process(data: Dict[str, List[str]]):
    """Batch processing endpoint for multiple texts."""
    texts = data.get("texts", [])
    return {
        "results": [f"Processed: {text}" for text in texts],
        "count": len(texts),
        "route": "/batch/process"
    }

if __name__ == "__main__":
    # Run the server
    uvicorn.run(app, host="0.0.0.0", port=5005)
