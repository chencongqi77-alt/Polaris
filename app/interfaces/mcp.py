"""Minimal MCP client abstraction for MACP.

Supports three transport modes:
  - direct: in-process direct calls (default, zero-dependency)
  - stdio:  MCP stdio subprocess via langchain-mcp-adapters
  - http:   Remote MCP HTTP/SSE endpoint
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from app.interfaces.mcp_config import MCPConfig

logger = logging.getLogger(__name__)


@dataclass
class MCPResult:
    """Structured result from an MCP tool call."""
    success: bool = True
    content: str = ""
    error: str = ""


# ---------------------------------------------------------------------------
# Public protocol
# ---------------------------------------------------------------------------

class MCPClient:
    """Unified MCP client facade used by agents and graph nodes."""

    def __init__(
        self,
        mode: str = "direct",
        server_command: str = "",
        url: str = "",
        config: Optional[MCPConfig] = None,
    ) -> None:
        if config is not None:
            mode = config.mode
            server_command = config.server_command
            url = config.url
        self._mode = mode

        if mode == "direct":
            from app.mcp_server.direct_client import DirectMCPClient
            self._backend = DirectMCPClient(config=config)
        elif mode == "stdio":
            if not server_command:
                raise ValueError("stdio MCP mode requires a server_command.")
            self._backend = SubprocessMCPClient(server_command=server_command)
        elif mode == "http":
            if not url:
                raise ValueError("http MCP mode requires a url.")
            self._backend = HTTPMCPClient(url=url)
        else:
            raise ValueError(f"Unsupported MCP mode: {mode}")

    # ---- Delegated API ----
    def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        return self._backend.call_tool(tool_name=tool_name, arguments=arguments)

    def list_tools(self) -> List[Dict[str, Any]]:
        return self._backend.list_tools()

    def get_context(self, tool_name: str) -> Dict[str, Any]:
        return self._backend.get_context(tool_name=tool_name)

    def close(self) -> None:
        self._backend.close()


# ---------------------------------------------------------------------------
# Backends
# ---------------------------------------------------------------------------

class SubprocessMCPClient:
    """stdio MCP backend: spawns MCP server as a subprocess."""

    def __init__(self, server_command: str) -> None:
        try:
            from langchain_mcp_adapters.client import MultiServerMCPClient
        except ImportError as exc:
            raise ImportError(
                "stdio MCP mode requires the langchain-mcp-adapters package. "
                "Install it with: pip install langchain-mcp-adapters"
            ) from exc

        self._command = server_command
        # For stdio we use MultiServerMCPClient with a single server
        servers = {
            "default": {
                "command": server_command.split()[0],
                "args": server_command.split()[1:],
                "transport": "stdio",
            }
        }
        self._mcp_client = MultiServerMCPClient(servers)
        self._tools_cache: Dict[str, Any] = {}

    def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as pool:
                    result = pool.submit(
                        asyncio.run, self._call_tool_async(tool_name, arguments)
                    ).result()
            else:
                result = asyncio.run(self._call_tool_async(tool_name, arguments))
            return result
        except Exception as exc:
            logger.warning("stdio MCP call_tool failed: %s", exc)
            return {"status": "error", "error": str(exc)}

    async def _call_tool_async(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        async with self._mcp_client as client:
            tools = client.get_tools()
            tool = next((t for t in tools if t.name == tool_name), None)
            if tool is None:
                return {"status": "error", "error": f"Tool '{tool_name}' not found"}
            result = await tool.ainvoke(arguments)
            return {"status": "ok", "result": result}

    def list_tools(self) -> List[Dict[str, Any]]:
        return []

    def get_context(self, tool_name: str) -> Dict[str, Any]:
        return {}

    def close(self) -> None:
        pass


# ---------------------------------------------------------------------------
# Re-export DirectMCPClient for backward compatibility
# ---------------------------------------------------------------------------
from app.mcp_server.direct_client import DirectMCPClient  # noqa: E402, F401


# ---------------------------------------------------------------------------
# 工厂函数
# ---------------------------------------------------------------------------


def create_mcp_client(mode: str = "direct", **kwargs) -> MCPClient:
    """工厂函数：根据模式创建 MCP 客户端。

    Parameters
    ----------
    mode : str
        客户端模式: "direct", "stdio", "http"
    **kwargs
        传递给 MCPClient 的额外参数
    """
    return MCPClient(mode=mode, **kwargs)


# Alias for backward compatibility
McpClient = MCPClient


class HTTPMCPClient:
    """HTTP/SSE MCP backend."""

    def __init__(self, url: str) -> None:
        self._url = url

    def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        try:
            import httpx
            response = httpx.post(
                f"{self._url}/call_tool",
                json={"tool_name": tool_name, "arguments": arguments},
                timeout=30.0,
            )
            response.raise_for_status()
            return response.json()
        except Exception as exc:
            logger.warning("HTTP MCP call_tool failed: %s", exc)
            return {"status": "error", "error": str(exc)}

    def list_tools(self) -> List[Dict[str, Any]]:
        try:
            import httpx
            response = httpx.get(f"{self._url}/list_tools", timeout=10.0)
            response.raise_for_status()
            return response.json().get("tools", [])
        except Exception:
            return []

    def get_context(self, tool_name: str) -> Dict[str, Any]:
        return {}

    def close(self) -> None:
        pass