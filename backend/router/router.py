class ModelRouter:

    TASK_CAPABILITIES = {
        "general": ["general"],
        "reasoning": ["reasoning"],
        "coding": ["tool_use", "reasoning"],
        "document": ["general", "reasoning"],
        "vision": ["vision"],
    }

    def classify_task(self, message: str) -> str:
        """
        Simple deterministic task classifier.

        This is our first router version.
        Later we can replace this with an AI-based classifier.
        """

        text = message.lower()

        vision_keywords = [
            "image",
            "photo",
            "picture",
            "scanned",
            "scan",
            "drawing",
            "diagram",
            "handwritten",
        ]

        coding_keywords = [
            "code",
            "coding",
            "python",
            "javascript",
            "java",
            "c++",
            "program",
            "debug",
            "function",
            "script",
        ]

        reasoning_keywords = [
            "calculate",
            "solve",
            "analyze",
            "reason",
            "derive",
            "why",
            "compare",
        ]

        document_keywords = [
            "document",
            "report",
            "summarize",
            "summary",
            "approval note",
            "sop",
            "manual",
        ]

        if any(word in text for word in vision_keywords):
            return "vision"

        if any(word in text for word in coding_keywords):
            return "coding"

        if any(word in text for word in reasoning_keywords):
            return "reasoning"

        if any(word in text for word in document_keywords):
            return "document"

        return "general"

    def select_model(self, task: str, registry: list):
        """
        Select the best available model for the requested task.
        """

        required_capabilities = self.TASK_CAPABILITIES.get(
            task,
            ["general"],
        )

        candidates = []

        for model in registry:
            capabilities = model.get(
                "workbench_capabilities",
                [],
            )

            score = 0

            for capability in required_capabilities:
                if capability in capabilities:
                    score += 1

            if score > 0:
                candidates.append(
                    {
                        "model": model,
                        "score": score,
                    }
                )

        if not candidates:
            return None

        candidates.sort(
            key=lambda item: item["score"],
            reverse=True,
        )

        return candidates[0]["model"]

    async def route(self, message: str, registry: list):
        """
        Classify the request and select a suitable model.
        """

        task = self.classify_task(message)

        selected_model = self.select_model(
            task,
            registry,
        )

        return {
            "task": task,
            "model": (
                selected_model["name"]
                if selected_model
                else None
            ),
            "capabilities": (
                selected_model["workbench_capabilities"]
                if selected_model
                else []
            ),
            "available": selected_model is not None,
        }