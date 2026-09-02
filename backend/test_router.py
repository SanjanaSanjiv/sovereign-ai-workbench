import asyncio

from backend.models.manager import ModelManager
from backend.router.router import ModelRouter


async def main():
    manager = ModelManager()
    router = ModelRouter()

    registry = await manager.build_registry()

    tests = [
        "Write a Python program to sort an array",
        "Why does a refinery separate crude oil?",
        "Analyze this image",
    ]

    for message in tests:
        result = await router.route(
            message,
            registry,
        )

        print("\n--------------------------------")
        print("Message:", message)
        print("Task:", result["task"])
        print("Selected model:", result["model"])
        print("Capabilities:", result["capabilities"])
        print("Available:", result["available"])


if __name__ == "__main__":
    asyncio.run(main())

    