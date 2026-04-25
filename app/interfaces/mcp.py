from __future__ import annotations

from typing import Any, Dict, Optional, Protocol

import requests


class MCPClient(Protocol):
    """Minimal MCP tool-calling interface."""

    def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        ...


class MockMCPClient:
    """Simple MCP stub for integration tests."""

    def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "tool_name": tool_name,
            "arguments": arguments,
            "status": "ok",
            "source": "mock-mcp",
        }


class HttpMCPClient:
    """
    Real MCP-like HTTP client with mock fallback.
    Expected API:
      POST {base_url}/tools/{tool_name}
      body: {"arguments": {...}}
    """

    def __init__(
        self,
        base_url: str,
        api_key: Optional[str] = None,
        timeout_sec: float = 20.0,
        fallback_client: MCPClient | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout_sec = timeout_sec
        self.fallback_client = fallback_client or MockMCPClient()

    def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        url = f"{self.base_url}/tools/{tool_name}"
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        try:
            response = requests.post(
                url,
                json={"arguments": arguments},
                headers=headers,
                timeout=self.timeout_sec,
            )
            response.raise_for_status()
            payload = response.json()
            if isinstance(payload, dict):
                payload.setdefault("source", "http-mcp")
                return payload
            return {"status": "ok", "result": payload, "source": "http-mcp"}
        except Exception as exc:  # noqa: BLE001
            fallback = self.fallback_client.call_tool(tool_name=tool_name, arguments=arguments)
            fallback["fallback_reason"] = str(exc)
            fallback["source"] = "mock-mcp-fallback"
            return fallback
