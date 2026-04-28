from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Protocol, Sequence


@dataclass
class ChatMessage:
    role: str
    content: str


class LLMClient(Protocol):
    """Provider-agnostic LLM interface."""

    def generate(self, messages: Sequence[ChatMessage], temperature: float = 0.2) -> str:
        ...


class MockLLMClient:
    """Deterministic fallback for local tests."""

    def generate(self, messages: Sequence[ChatMessage], temperature: float = 0.2) -> str:
        if not messages:
            return ""
        return f"[mock-llm] {messages[-1].content[:500]}"


class OpenAILLMClient:
    """
    Real LLM client using OpenAI-compatible API.
    Reads OPENAI_API_KEY and OPENAI_BASE_URL (optional) from env.
    """

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
    ) -> None:
        from openai import OpenAI

        self._api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self._base_url = base_url or os.environ.get("OPENAI_BASE_URL", "")
        self._model = model or os.environ.get("OPENAI_MODEL", "")

        if not self._api_key:
            raise RuntimeError(
                "OPENAI_API_KEY is not set. "
                "Provide it via env variable or constructor argument."
            )
        if not self._model:
            raise RuntimeError(
                "OPENAI_MODEL is not set. "
                "Provide it via env variable or constructor argument."
            )

        kwargs = {"api_key": self._api_key}
        if self._base_url:
            kwargs["base_url"] = self._base_url

        self._client = OpenAI(**kwargs)

    def generate(self, messages: Sequence[ChatMessage], temperature: float = 0.2) -> str:
        raw = [
            {"role": msg.role, "content": msg.content}
            for msg in messages
        ]

        response = self._client.chat.completions.create(
            model=self._model,
            messages=raw,  # type: ignore[arg-type]
            temperature=temperature,
        )
        return response.choices[0].message.content or ""
