import httpx


OLLAMA_BASE_URL = "http://localhost:11434"


class ModelManager:
    def __init__(self, base_url: str = OLLAMA_BASE_URL):
        self.base_url = base_url.rstrip("/")

    async def list_models(self):
        """Return all models installed in Ollama."""
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(
                f"{self.base_url}/api/tags"
            )

        response.raise_for_status()

        data = response.json()

        return data.get("models", [])

    async def get_model_details(self, model_name: str):
        """Return detailed information about one Ollama model."""
        payload = {
            "model": model_name,
            "verbose": False,
        }

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{self.base_url}/api/show",
                json=payload,
            )

        response.raise_for_status()

        return response.json()

    def map_capabilities(self, ollama_capabilities):
        """
        Convert Ollama capabilities into capabilities
        understood by the Sovereign AI Workbench.
        """

        capabilities = set()

        for capability in ollama_capabilities:
            if capability == "completion":
                capabilities.add("general")

            elif capability == "thinking":
                capabilities.add("reasoning")

            elif capability == "tools":
                capabilities.add("tool_use")

            elif capability == "vision":
                capabilities.add("vision")

        return sorted(capabilities)

    async def build_registry(self):
        """
        Build a local registry containing model metadata
        and Workbench capabilities.
        """

        models = await self.list_models()

        registry = []

        for model in models:
            name = model.get("name")

            try:
                details = await self.get_model_details(name)
            except Exception as exc:
                details = {
                    "capabilities": [],
                    "error": str(exc),
                }

            model_details = model.get("details", {})

            ollama_capabilities = details.get(
                "capabilities", []
            )

            workbench_capabilities = self.map_capabilities(
                ollama_capabilities
            )

            registry.append(
                {
                    "name": name,
                    "size": model.get("size"),
                    "modified_at": model.get("modified_at"),
                    "family": model_details.get("family"),
                    "parameter_size": model_details.get(
                        "parameter_size"
                    ),
                    "quantization": model_details.get(
                        "quantization_level"
                    ),
                    "ollama_capabilities": ollama_capabilities,
                    "workbench_capabilities": workbench_capabilities,
                }
            )

        return registry