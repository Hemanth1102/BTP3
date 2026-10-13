from typing import Any, Dict, List, Optional

import ollama


class OllamaQwenPlanner:
    """
    Thin wrapper around a local Qwen model running through Ollama.

    This object is callable so it can be passed directly to
    QueryPlanner(llm=..., mode="llm").
    """

    def __init__(
        self,
        model: str = "qwen3:8b",
    ):
        self.model = model

    def __call__(self, prompt: str) -> str:
        response = ollama.chat(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
        )

        return response["message"]["content"]