"""MCP Server implementation for MACP agents.

This module creates and runs an MCP server that provides tools
for the MACP agents to call via the MCP protocol.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

try:
    from mcp.server import Server
    from mcp.server.stdio import stdio_server
    from mcp.types import Tool, TextContent
except ImportError as exc:
    raise ImportError(
        "mcp package is not installed. Install it with `pip install mcp`."
    ) from exc

from app.mcp_server.tools import (
    register_artifact_tools,
    register_file_tools,
    register_search_tools,
    register_calculate_tools,
    register_web_search_tools,
)

logger = logging.getLogger(__name__)


# Create the MCP server instance
_server: Server | None = None


def create_mcp_server(name: str = "macp-tools") -> Server:
    """Create and configure the MCP server with all registered tools.

    Args:
        name: Server name for identification.

    Returns:
        Configured MCP Server instance.
    """
    global _server
    _server = Server(name)

    # Register all tool handlers
    _register_all_tools(_server)

    return _server


def _register_all_tools(server: Server) -> None:
    """Register all tools to the MCP server.

    Args:
        server: The MCP Server instance.
    """
    # Import tool functions for direct registration
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
    from app.mcp_server.tools.web_search import (
        web_search_github,
        web_search,
    )

    # Tool definitions with schemas
    TOOLS = [
        # Artifact tools
        Tool(
            name="artifact.prepare",
            description="Prepare a new artifact for a candidate solution",
            inputSchema={
                "type": "object",
                "properties": {
                    "candidate_id": {"type": "string", "description": "Unique identifier for the candidate"},
                    "preview": {"type": "string", "description": "Preview text of the artifact content"},
                },
                "required": ["candidate_id", "preview"],
            },
        ),
        Tool(
            name="artifact.finalize",
            description="Finalize an artifact with full content",
            inputSchema={
                "type": "object",
                "properties": {
                    "artifact_id": {"type": "string", "description": "The artifact ID returned by artifact_prepare"},
                    "content": {"type": "string", "description": "Full content of the artifact"},
                },
                "required": ["artifact_id", "content"],
            },
        ),
        Tool(
            name="artifact.get",
            description="Retrieve an artifact by ID",
            inputSchema={
                "type": "object",
                "properties": {
                    "artifact_id": {"type": "string", "description": "The artifact ID"},
                },
                "required": ["artifact_id"],
            },
        ),
        Tool(
            name="artifact.list",
            description="List all artifacts, optionally filtered by candidate_id",
            inputSchema={
                "type": "object",
                "properties": {
                    "candidate_id": {"type": "string", "description": "Optional candidate ID to filter by", "default": ""},
                },
                "required": [],
            },
        ),
        # File tools
        Tool(
            name="file.read",
            description="Read content from a file",
            inputSchema={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path"},
                    "encoding": {"type": "string", "description": "File encoding", "default": "utf-8"},
                },
                "required": ["path"],
            },
        ),
        Tool(
            name="file.write",
            description="Write content to a file",
            inputSchema={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path"},
                    "content": {"type": "string", "description": "Content to write"},
                    "encoding": {"type": "string", "description": "File encoding", "default": "utf-8"},
                },
                "required": ["path", "content"],
            },
        ),
        Tool(
            name="file.list",
            description="List files in a directory",
            inputSchema={
                "type": "object",
                "properties": {
                    "directory": {"type": "string", "description": "Directory path", "default": "."},
                },
                "required": [],
            },
        ),
        Tool(
            name="file.exists",
            description="Check if a file or directory exists",
            inputSchema={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path to check"},
                },
                "required": ["path"],
            },
        ),
        # Search tools
        Tool(
            name="search.index",
            description="Index documents for later search",
            inputSchema={
                "type": "object",
                "properties": {
                    "scope": {"type": "string", "description": "Search scope/namespace"},
                    "documents": {"type": "string", "description": "JSON string of document list"},
                },
                "required": ["scope", "documents"],
            },
        ),
        Tool(
            name="search.query",
            description="Search indexed documents",
            inputSchema={
                "type": "object",
                "properties": {
                    "scope": {"type": "string", "description": "Search scope"},
                    "query": {"type": "string", "description": "Search query"},
                    "top_k": {"type": "integer", "description": "Max results", "default": 5},
                },
                "required": ["scope", "query"],
            },
        ),
        Tool(
            name="search.regex",
            description="Search with regex pattern",
            inputSchema={
                "type": "object",
                "properties": {
                    "scope": {"type": "string", "description": "Search scope"},
                    "pattern": {"type": "string", "description": "Regex pattern"},
                    "top_k": {"type": "integer", "description": "Max results", "default": 5},
                },
                "required": ["scope", "pattern"],
            },
        ),
        Tool(
            name="search.clear",
            description="Clear search index",
            inputSchema={
                "type": "object",
                "properties": {
                    "scope": {"type": "string", "description": "Scope to clear (empty = all)", "default": ""},
                },
                "required": [],
            },
        ),
        # Calculate tools
        Tool(
            name="calculate.math",
            description="Evaluate a mathematical expression",
            inputSchema={
                "type": "object",
                "properties": {
                    "expression": {"type": "string", "description": "Math expression"},
                },
                "required": ["expression"],
            },
        ),
        Tool(
            name="calculate.stats",
            description="Calculate statistics for numbers",
            inputSchema={
                "type": "object",
                "properties": {
                    "numbers": {"type": "string", "description": "JSON or comma-separated numbers"},
                },
                "required": ["numbers"],
            },
        ),
        Tool(
            name="calculate.format_number",
            description="Format a number with decimals",
            inputSchema={
                "type": "object",
                "properties": {
                    "value": {"type": "number", "description": "Number to format"},
                    "decimals": {"type": "integer", "description": "Decimal places", "default": 2},
                },
                "required": ["value"],
            },
        ),
        Tool(
            name="calculate.time_diff",
            description="Calculate time difference",
            inputSchema={
                "type": "object",
                "properties": {
                    "start_time": {"type": "string", "description": "Start timestamp"},
                    "end_time": {"type": "string", "description": "End timestamp"},
                },
                "required": ["start_time", "end_time"],
            },
        ),
        # Web search tools
        Tool(
            name="search.github",
            description="Search GitHub repositories by keyword. Returns matching repos with name, description, URL, stars, language, and topics.",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query (e.g. 'tinyml embedded', 'LLM agent framework')"},
                    "per_page": {"type": "integer", "description": "Number of results (default 5, max 20)", "default": 5},
                    "sort": {"type": "string", "description": "Sort by 'stars', 'forks', 'updated', or 'best-match'", "default": "stars"},
                },
                "required": ["query"],
            },
        ),
        Tool(
            name="search.web",
            description="Search the web for information using DuckDuckGo. Returns titles, snippets, and URLs.",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query string"},
                    "max_results": {"type": "integer", "description": "Max results to return", "default": 5},
                },
                "required": ["query"],
            },
        ),
    ]

    # Tool function mapping
    TOOL_FUNCTIONS = {
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
        "search.github": web_search_github,
        "search.web": web_search,
    }

    @server.list_tools()
    async def list_tools() -> list[Tool]:
        """Return list of available tools."""
        return TOOLS

    @server.call_tool()
    async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
        """Execute a tool call.

        Args:
            name: Tool name.
            arguments: Tool arguments.

        Returns:
            List of TextContent results.
        """
        if name not in TOOL_FUNCTIONS:
            return [TextContent(type="text", text=f"Unknown tool: {name}")]

        func = TOOL_FUNCTIONS[name]
        try:
            # Call the tool function with arguments
            # Map argument names to function parameters
            result = func(**arguments)
            return [TextContent(type="text", text=result)]
        except Exception as e:
            logger.exception(f"Tool {name} execution failed")
            return [TextContent(type="text", text=f"Error: {e}")]


async def run_stdio_server(server: Server) -> None:
    """Run the MCP server using stdio transport.

    Args:
        server: The MCP Server instance to run.
    """
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options(),
        )


def run_mcp_server() -> None:
    """Entry point to run the MCP server."""
    server = create_mcp_server()
    asyncio.run(run_stdio_server(server))


# CLI entry point
if __name__ == "__main__":
    run_mcp_server()