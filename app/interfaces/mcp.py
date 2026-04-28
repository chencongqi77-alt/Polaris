from __future__ import annotations

import asyncio
import json
import logging
import os
import shlex
import sys
from contextlib import asynccontextmanager
from typing import Any, Dict, Optional, Protocol

import requests

logger = logging.getLogger(__name__)


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


class StdioMCPClient:
    """
    Real MCP client using stdio transport.
    Connects to an MCP server running as a subprocess.
    
    This client uses the MCP SDK to properly communicate with
    the server via JSON-RPC over stdin/stdout.
    """

    def __init__(
        self,
        server_command: Optional[str] = None,
        fallback_client: MCPClient | None = None,
    ) -> None:
        """Initialize the stdio MCP client.

        Args:
            server_command: Command to run the MCP server. 
                Default: "python -m app.mcp_server.server"
            fallback_client: Fallback client if MCP connection fails.
        """
        self.server_command = server_command or "python -m app.mcp_server.server"
        self.fallback_client = fallback_client or MockMCPClient()
        self._session: Any = None
        self._process: Any = None
        self._connected = False
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    def _get_loop(self) -> asyncio.AbstractEventLoop:
        """Get or create an event loop."""
        if self._loop is None or self._loop.is_closed():
            try:
                self._loop = asyncio.get_running_loop()
            except RuntimeError:
                self._loop = asyncio.new_event_loop()
                asyncio.set_event_loop(self._loop)
        return self._loop

    def _run_async(self, coro: Any) -> Any:
        """Run an async coroutine synchronously."""
        loop = self._get_loop()
        if loop.is_running():
            # If loop is already running (e.g., inside async context),
            # we need to use a different approach
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(asyncio.run, coro)
                return future.result()
        return loop.run_until_complete(coro)

    async def _connect_async(self) -> bool:
        """Async connection to MCP server."""
        try:
            from mcp import ClientSession, StdioServerParameters
            from mcp.client.stdio import stdio_client
        except ImportError:
            logger.warning("mcp package not installed, using fallback client")
            return False

        try:
            parts = shlex.split(self.server_command)
        except ValueError as exc:
            logger.warning(f"Invalid server_command={self.server_command!r}: {exc}")
            parts = []

        if parts:
            command = parts[0]
            args = parts[1:]
        else:
            command = sys.executable
            args = ["-m", "app.mcp_server.server"]

        server_params = StdioServerParameters(command=command, args=args, env=os.environ.copy())

        try:
            self._process = await stdio_client(server_params).__aenter__()
            read_stream, write_stream = self._process[:2]
            self._session = ClientSession(read_stream, write_stream)
            await self._session.__aenter__()
            await self._session.initialize()
            self._connected = True
            logger.info("MCP client connected to server successfully")
            return True
        except Exception as e:
            logger.warning(f"MCP connection failed: {e}")
            return False

    async def _call_tool_async(
        self, tool_name: str, arguments: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Async tool call via MCP session."""
        if not self._connected or self._session is None:
            return self.fallback_client.call_tool(tool_name, arguments)

        try:
            result = await self._session.call_tool(tool_name, arguments)
            # Parse the result
            if result.content:
                # Extract text content
                text_parts = []
                for item in result.content:
                    if hasattr(item, "text"):
                        text_parts.append(item.text)
                    elif hasattr(item, "data"):
                        text_parts.append(str(item.data))
                result_text = "\n".join(text_parts)
                try:
                    parsed = json.loads(result_text)
                    parsed["source"] = "mcp-stdio"
                    return parsed
                except json.JSONDecodeError:
                    return {
                        "tool_name": tool_name,
                        "arguments": arguments,
                        "result": result_text,
                        "status": "ok",
                        "source": "mcp-stdio",
                    }
            return {
                "tool_name": tool_name,
                "arguments": arguments,
                "status": "ok",
                "source": "mcp-stdio",
            }
        except Exception as e:
            logger.warning(f"MCP tool call failed: {e}")
            fallback = self.fallback_client.call_tool(tool_name, arguments)
            fallback["fallback_reason"] = str(e)
            fallback["source"] = "mock-mcp-fallback"
            return fallback

    def connect(self) -> bool:
        """Connect to the MCP server synchronously."""
        return self._run_async(self._connect_async())

    def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Call a tool on the MCP server.

        Args:
            tool_name: Name of the tool to call.
            arguments: Arguments for the tool.

        Returns:
            Result from the tool execution.
        """
        if not self._connected:
            # Try to connect first
            self.connect()
        
        if self._connected:
            return self._run_async(self._call_tool_async(tool_name, arguments))
        
        return self.fallback_client.call_tool(tool_name, arguments)

    async def disconnect_async(self) -> None:
        """Async disconnect from MCP server."""
        if self._session:
            try:
                await self._session.__aexit__(None, None, None)
            except Exception:
                pass
        if self._process:
            try:
                await self._process.__aexit__(None, None, None)
            except Exception:
                pass
        self._connected = False
        self._session = None
        self._process = None

    def disconnect(self) -> None:
        """Disconnect from MCP server synchronously."""
        self._run_async(self.disconnect_async())


class DirectMCPClient:
    """
    Direct MCP client that calls tool functions without subprocess.
    
    This is a simpler alternative to StdioMCPClient for local testing
    and development. It directly imports and calls the tool functions
    from the MCP server module, avoiding the complexity of stdio transport.
    """

    def __init__(self, fallback_client: MCPClient | None = None) -> None:
        """Initialize the direct MCP client.

        Args:
            fallback_client: Fallback client if tool is not found.
        """
        self.fallback_client = fallback_client or MockMCPClient()
        self._tool_functions: Dict[str, Any] = {}
        self._load_tools()

    def _load_tools(self) -> None:
        """Load tool functions from MCP server module."""
        try:
            from app.mcp_server.tools.artifact import (
                artifact_prepare,
                artifact_finalize,
                artifact_get,
                artifact_list,
            )
            from app.mcp_server.tools.files import (
                file_read,
                file_write,
                file_list,
                file_exists,
            )
            from app.mcp_server.tools.search import (
                search_index,
                search_query,
                search_regex,
                search_clear,
            )
            from app.mcp_server.tools.calculate import (
                calculate_math,
                calculate_stats,
                calculate_format_number,
                calculate_time_diff,
            )

            self._tool_functions = {
                "artifact.prepare": artifact_prepare,
                "artifact.finalize": artifact_finalize,
                "artifact.get": artifact_get,
                "artifact.list": artifact_list,
                "file.read": file_read,
                "file.write": file_write,
                "file.list": file_list,
                "file.exists": file_exists,
                "search.index": search_index,
                "search.query": search_query,
                "search.regex": search_regex,
                "search.clear": search_clear,
                "calculate.math": calculate_math,
                "calculate.stats": calculate_stats,
                "calculate.format_number": calculate_format_number,
                "calculate.time_diff": calculate_time_diff,
            }
            logger.info(f"DirectMCPClient loaded {len(self._tool_functions)} tools")
        except ImportError as e:
            logger.warning(f"Failed to load MCP tools: {e}")
            self._tool_functions = {}

    def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Call a tool directly.

        Args:
            tool_name: Name of the tool to call.
            arguments: Arguments for the tool.

        Returns:
            Result from the tool execution.
        """
        if tool_name not in self._tool_functions:
            logger.warning(f"Tool '{tool_name}' not found, using fallback")
            fallback = self.fallback_client.call_tool(tool_name, arguments)
            fallback["fallback_reason"] = f"Tool '{tool_name}' not registered"
            fallback["source"] = "mock-mcp-fallback"
            return fallback

        func = self._tool_functions[tool_name]
        try:
            result_str = func(**arguments)
            # Parse JSON result
            try:
                result = json.loads(result_str)
                result["source"] = "mcp-direct"
                return result
            except json.JSONDecodeError:
                return {
                    "tool_name": tool_name,
                    "arguments": arguments,
                    "result": result_str,
                    "status": "ok",
                    "source": "mcp-direct",
                }
        except Exception as e:
            logger.warning(f"Tool '{tool_name}' execution failed: {e}")
            fallback = self.fallback_client.call_tool(tool_name, arguments)
            fallback["fallback_reason"] = str(e)
            fallback["source"] = "mock-mcp-fallback"
            return fallback

    def list_tools(self) -> list[str]:
        """List available tools."""
        return list(self._tool_functions.keys())


# Factory function for creating MCP clients
def create_mcp_client(
    mode: str = "direct",
    **kwargs: Any,
) -> MCPClient:
    """Create an MCP client based on the specified mode.

    Args:
        mode: Client mode - "mock", "direct", "stdio", or "http".
        **kwargs: Additional arguments for the specific client.

    Returns:
        An MCP client instance.
    """
    if mode == "mock":
        return MockMCPClient()
    elif mode == "direct":
        return DirectMCPClient(kwargs.get("fallback_client"))
    elif mode == "stdio":
        return StdioMCPClient(
            kwargs.get("server_command"),
            kwargs.get("fallback_client"),
        )
    elif mode == "http":
        return HttpMCPClient(
            kwargs.get("base_url", "http://localhost:8080"),
            kwargs.get("api_key"),
            kwargs.get("timeout_sec", 20.0),
            kwargs.get("fallback_client"),
        )
    else:
        logger.warning(f"Unknown MCP client mode '{mode}', using mock")
        return MockMCPClient()
