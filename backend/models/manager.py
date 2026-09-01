import httpx


OLLAMA_TAGS_URL = "http://localhost:11434/api/tags"


class ModelManager:

    async def list_models(self):
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(OLLAMA_TAGS_URL)

        response.raise_for_status()

        data = response.json()

        models = []

        for model in data.get("models", []):
            models.append({
                "name": model.get("name"),
                "size": model.get("size"),
                "modified_at": model.get("modified_at"),
            })

        return models