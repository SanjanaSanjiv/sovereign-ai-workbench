from fastapi import FastAPI, Form, UploadFile, File
from pydantic import BaseModel
import httpx
import base64

from backend.models.manager import ModelManager
from backend.router.router import ModelRouter


app = FastAPI(
    title="Sovereign AI Workbench",
    description="Local AI backend for confidential industrial work",
    version="0.4.0",
)


OLLAMA_URL = "http://localhost:11434/api/chat"

model_manager = ModelManager()
model_router = ModelRouter()


class ChatRequest(BaseModel):
    message: str


@app.get("/")
async def root():
    return {
        "status": "online",
        "service": "Sovereign AI Workbench",
        "routing": "enabled",
        "models": [
            "qwen3:4b",
            "qwen2.5-coder:7b",
            "gemma3:4b",
        ],
    }


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "ollama_url": OLLAMA_URL,
        "routing": "enabled",
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

    # 2. Classify the request and select the best model
    routing_result = await model_router.route(
        request.message,
        registry,
    )

    # 3. Check whether a suitable model was found
    if not routing_result["available"]:
        return {
            "error": "No suitable model available",
            "task": routing_result["task"],
            "capabilities": routing_result["capabilities"],
        }

    # 4. Get the model selected by our router
    selected_model = routing_result["model"]

    # 5. Prepare Ollama request
    payload = {
        "model": selected_model,
        "messages": [
            {
                "role": "user",
                "content": request.message,
            }
        ],
        "stream": False,
        "keep_alive": "10m",
    }

    # 6. Enable thinking only for reasoning tasks
    #    and only when the selected model supports reasoning.
    if "reasoning" in routing_result["capabilities"]:
        payload["think"] = routing_result["task"] == "reasoning"

    # 7. Send request to Ollama
    async with httpx.AsyncClient(timeout=300.0) as client:

        response = await client.post(
            OLLAMA_URL,
            json=payload,
        )

    response.raise_for_status()

    result = response.json()

    # 8. Extract assistant response
    message = result.get("message", {})

    # 9. Return routing + model + response information
    return {
        "task": routing_result["task"],
        "model": result.get("model"),
        "capabilities": routing_result["capabilities"],
        "response": message.get("content", ""),
    }


@app.post("/vision")
async def vision(
    message: str = Form(...),
    file: UploadFile = File(...),
):

    # 1. Check whether a file was uploaded
    if not file.filename:
        return {
            "error": "No image file provided"
        }

    # 2. Make sure the uploaded file is an image
    if not file.content_type or not file.content_type.startswith("image/"):
        return {
            "error": "Uploaded file must be an image",
            "content_type": file.content_type,
        }

    # 3. Read the uploaded image
    image_bytes = await file.read()

    # 4. Check whether the image is empty
    if not image_bytes:
        return {
            "error": "Uploaded image is empty"
        }

    # 5. Convert image bytes to Base64
    image_base64 = base64.b64encode(image_bytes).decode("utf-8")

    # 6. Prepare request for Gemma3 vision
    payload = {
        "model": "gemma3:4b",
        "messages": [
            {
                "role": "user",
                "content": message,
                "images": [image_base64],
            }
        ],
        "stream": False,
        "keep_alive": "10m",
    }

    # 7. Send image + prompt to Ollama
    async with httpx.AsyncClient(timeout=300.0) as client:

        response = await client.post(
            OLLAMA_URL,
            json=payload,
        )

    response.raise_for_status()

    result = response.json()

    # 8. Extract Gemma3 response
    model_message = result.get("message", {})

    # 9. Return vision analysis
    return {
        "task": "vision",
        "model": result.get("model"),
        "filename": file.filename,
        "content_type": file.content_type,
        "response": model_message.get("content", ""),
    }


    