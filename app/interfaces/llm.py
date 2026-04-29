"""LLM Client interface and default implementation for MACP agents."""

from __future__ import annotations

import json
import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, List, Optional, Sequence

logger = logging.getLogger(__name__)


@dataclass
class ChatMessage:
    """A single chat message with role and content."""
    role: str
    content: str


@dataclass
class ChatResponse:
    """Response from a chat completion."""
    content: str = ""
    model: str = ""
    usage: dict[str, int] = field(default_factory=dict)


class LLMClient(ABC):
    """Abstract LLM client interface."""

    @abstractmethod
    def generate(
        self,
        prompt: str | None = None,
        messages: List[ChatMessage] | None = None,
        temperature: float = 0.3,
        **kwargs: Any,
    ) -> str:
        """Generate text from a prompt or messages. Returns the raw string response."""
        ...

    def generate_json(self, prompt: str | None = None, messages: List[ChatMessage] | None = None, **kwargs: Any) -> dict[str, Any]:
        """Generate and parse JSON from a prompt."""
        raw = self.generate(prompt=prompt, messages=messages, **kwargs)
        return json.loads(raw)


class OpenAILLMClient(LLMClient):
    """OpenAI-compatible LLM client using the openai package."""

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        temperature: float = 0.3,
        max_tokens: int = 2048,
    ) -> None:
        self.model = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        self.api_key = api_key or os.getenv("OPENAI_API_KEY", "")
        self.base_url = base_url or os.getenv("OPENAI_BASE_URL", None)
        self.temperature = temperature
        self.max_tokens = max_tokens

    def _get_client(self):  # type: ignore[no-untyped-def]
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError(
                "openai package is required for OpenAILLMClient. "
                "Install it with: pip install openai"
            )

        kwargs: dict[str, Any] = {"api_key": self.api_key}
        if self.base_url:
            kwargs["base_url"] = self.base_url
        return OpenAI(**kwargs)

    def generate(
        self,
        prompt: str | None = None,
        messages: List[ChatMessage] | None = None,
        temperature: float | None = None,
        **kwargs: Any,
    ) -> str:
        client = self._get_client()
        model = kwargs.pop("model", self.model)
        temp = temperature if temperature is not None else self.temperature
        max_tokens = kwargs.pop("max_tokens", self.max_tokens)

        # Build messages list
        if messages is not None:
            msg_list = [{"role": m.role, "content": m.content} for m in messages]
        elif prompt is not None:
            msg_list = [{"role": "user", "content": prompt}]
        else:
            raise ValueError("Either 'prompt' or 'messages' must be provided")

        response = client.chat.completions.create(
            model=model,
            messages=msg_list,
            temperature=temp,
            max_tokens=max_tokens,
            **kwargs,
        )
        content = response.choices[0].message.content or ""
        logger.debug("LLM response (%s): %s...", model, content[:200])
        return content.strip()


class StubLLMClient(LLMClient):
    """Deterministic stub LLM client for testing.

    Returns canned responses or a configurable sequence of responses.
    """

    def __init__(self, responses: list[str] | None = None) -> None:
        self._responses = list(responses or [])
        self._call_count = 0

    def generate(
        self,
        prompt: str | None = None,
        messages: List[ChatMessage] | None = None,
        temperature: float = 0.3,
        **kwargs: Any,
    ) -> str:
        if self._responses:
            idx = min(self._call_count, len(self._responses) - 1)
            self._call_count += 1
            return self._responses[idx]
        # Default: return a minimal valid JSON for TM/SG/Eva
        return '{"subtasks": ["task1", "task2", "task3"], "constraints": ["clear", "testable"]}'


class LLMAdapter:
    """High-level adapter wrapping an LLMClient for convenience.

    Provides a `.chat()` method that returns a ChatResponse, matching the
    pattern used by BaseAgent.
    """

    def __init__(
        self,
        client: LLMClient | None = None,
        model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
    ) -> None:
        if client is not None:
            self._client = client
        else:
            self._client = OpenAILLMClient(
                model=model,
                api_key=api_key,
                base_url=base_url,
            )

    def chat(
        self,
        user_msg: str,
        system_msg: str | None = None,
        temperature: float = 0.3,
        **kwargs: Any,
    ) -> ChatResponse:
        messages = []
        if system_msg:
            messages.append(ChatMessage(role="system", content=system_msg))
        messages.append(ChatMessage(role="user", content=user_msg))

        content = self._client.generate(messages=messages, temperature=temperature, **kwargs)
        return ChatResponse(content=content)

    def generate(
        self,
        prompt: str | None = None,
        messages: List[ChatMessage] | None = None,
        temperature: float = 0.3,
        **kwargs: Any,
    ) -> str:
        return self._client.generate(prompt=prompt, messages=messages, temperature=temperature, **kwargs)


def get_default_llm_client() -> LLMClient:
    """Return the default LLM client based on environment configuration."""
    provider = os.getenv("LLM_PROVIDER", "openai").lower()

    if provider == "stub":
        return StubLLMClient()

    # Default to OpenAI-compatible
    return OpenAILLMClient(
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        api_key=os.getenv("OPENAI_API_KEY", ""),
        base_url=os.getenv("OPENAI_BASE_URL", None),
    )