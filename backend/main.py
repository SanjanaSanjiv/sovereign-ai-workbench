from fastapi import FastAPI
from pydantic import BaseModel
import httpx

from backend.models.manager import ModelManager


app = FastAPI(
    title="Sovereign AI Workbench",
    description="Local AI backend for confidential industrial work",
    version="0.2.0",
)

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL_NAME = "qwen3:4b"

model_manager = ModelManager()


class ChatRequest(BaseModel):
    message: str


@app.get("/")
async def root():
    return {
        "status": "online",
        "service": "Sovereign AI Workbench",
        "model": MODEL_NAME,
    }


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "ollama_url": OLLAMA_URL,
        "model": MODEL_NAME,
    }


@app.get("/models")
async def get_models():
    models = await model_manager.list_models()

    return {
        "count": len(models),
        "models": models,
    }


@app.post("/chat")
async def chat(request: ChatRequest):

    payload = {
        "model": MODEL_NAME,
        "messages": [
            {
                "role": "user",
                "content": request.message,
            }
        ],
        "stream": False,
    }

    async with httpx.AsyncClient(timeout=120.0) as client:
        response = await client.post(
            OLLAMA_URL,
            json=payload,
        )

    response.raise_for_status()

    result = response.json()

    return {
        "model": result.get("model"),
        "response": result["message"]["content"],
    }