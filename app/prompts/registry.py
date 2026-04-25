from __future__ import annotations

from string import Template
from typing import Any, Dict


class PromptRegistry:
    """Stores and renders named prompt templates."""

    def __init__(self, templates: Dict[str, str] | None = None) -> None:
        self._templates = templates or {}

    def register(self, name: str, template: str) -> None:
        self._templates[name] = template

    def render(self, name: str, variables: Dict[str, Any]) -> str:
        if name not in self._templates:
            raise KeyError(f"Prompt template not found: {name}")
        return Template(self._templates[name]).safe_substitute(variables)
