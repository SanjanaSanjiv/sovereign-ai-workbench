from fastapi import FastAPI
from pydantic import BaseModel
import httpx

from backend.models.manager import ModelManager
from backend.router.router import ModelRouter


app = FastAPI(
    title="Sovereign AI Workbench",
    description="Local AI backend for confidential industrial work",
    version="0.2.0",
)

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL_NAME = "qwen3:4b"

model_manager = ModelManager()
model_router = ModelRouter()

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

@app.get("/registry")
async def get_registry():
    registry = await model_manager.build_registry()

    return {
        "count": len(registry),
        "models": registry,
    }

@app.post("/route")
async def route_request(request: ChatRequest):
    registry = await model_manager.build_registry()

    routing_result = await model_router.route(
        request.message,
        registry,
    )

    return routing_result

@app.post("/chat")
async def chat(request: ChatRequest):

    # 1. Build the current model registry
    registry = await model_manager.build_registry()

    # 2. Let the router classify the task and select a model
    routing_result = await model_router.route(
        request.message,
        registry,
    )

    # 3. Make sure a suitable model was found
    if not routing_result["available"]:
        return {
            "error": "No suitable model available",
            "task": routing_result["task"],
        }

    selected_model = routing_result["model"]

    # 4. Send the request to the selected model
    payload = {
        "model": selected_model,
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

    # 5. Return both routing information and the AI response
    return {
        "task": routing_result["task"],
        "model": result.get("model"),
        "capabilities": routing_result["capabilities"],
        "response": result["message"]["content"],
    }
