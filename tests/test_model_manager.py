import asyncio

from models.manager import ModelManager


async def main():
    manager = ModelManager()

    models = await manager.list_models()

    print("\nInstalled Ollama models:")

    for model in models:
        print(f"- {model['name']}")
        print(f"  Size: {model['size']}")
        print(f"  Modified: {model['modified_at']}")


asyncio.run(main())
