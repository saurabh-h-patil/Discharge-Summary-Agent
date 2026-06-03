"""
LLM client wrapper — provides a unified interface to GPT-4o for text and vision.
"""

import base64
import time
from typing import Optional
from openai import OpenAI
from app.core.config import get_settings


class LLMClient:
    """Wrapper around OpenAI client with retry logic and vision support."""

    def __init__(self):
        settings = get_settings()
        self.client = OpenAI(api_key=settings.openai_api_key)
        self.model = settings.openai_model
        self.vision_model = settings.openai_vision_model
        self.max_retries = settings.max_retries
        self.vision_max_tokens = settings.vision_max_tokens

    def chat(
        self,
        messages: list[dict],
        temperature: float = 0.1,
        max_tokens: int = 4096,
        response_format: Optional[dict] = None,
    ) -> str:
        """Send a chat completion request with retry logic."""
        for attempt in range(self.max_retries):
            try:
                kwargs = {
                    "model": self.model,
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                }
                if response_format:
                    kwargs["response_format"] = response_format

                response = self.client.chat.completions.create(**kwargs)
                return response.choices[0].message.content

            except Exception as e:
                if attempt < self.max_retries - 1:
                    wait = 2 ** (attempt + 1)
                    print(f"  [LLM] Retry {attempt + 1}/{self.max_retries} after {wait}s: {e}")
                    time.sleep(wait)
                else:
                    raise RuntimeError(
                        f"LLM call failed after {self.max_retries} attempts: {e}"
                    ) from e

    def vision(
        self,
        image_base64: str,
        prompt: str,
        system_prompt: str = "You are a clinical document reader. Extract text accurately.",
        temperature: float = 0.0,
    ) -> str:
        """Send an image to GPT-4o vision for text extraction."""
        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{image_base64}",
                            "detail": "high",
                        },
                    },
                ],
            },
        ]

        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.vision_model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=self.vision_max_tokens,
                )
                return response.choices[0].message.content

            except Exception as e:
                if attempt < self.max_retries - 1:
                    wait = 2 ** (attempt + 1)
                    print(f"  [Vision] Retry {attempt + 1}/{self.max_retries} after {wait}s: {e}")
                    time.sleep(wait)
                else:
                    raise RuntimeError(
                        f"Vision call failed after {self.max_retries} attempts: {e}"
                    ) from e


# Singleton
_llm_client: Optional[LLMClient] = None


def get_llm_client() -> LLMClient:
    """Get or create the global LLM client instance."""
    global _llm_client
    if _llm_client is None:
        _llm_client = LLMClient()
    return _llm_client
