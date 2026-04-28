"""MCP Server module for MACP agents.

This module provides MCP (Model Context Protocol) server functionality
for agent tool execution.

Usage:
    # Run as module (starts the MCP server)
    python -m app.mcp_server.server
    
    # Or programmatically create a server
    from app.mcp_server import create_mcp_server
    server = create_mcp_server()
"""

__all__ = ["create_mcp_server", "run_mcp_server"]


def __getattr__(name: str):
    """Lazy import to avoid RuntimeWarning when running as module."""
    if name == "create_mcp_server":
        from app.mcp_server.server import create_mcp_server
        return create_mcp_server
    elif name == "run_mcp_server":
        from app.mcp_server.server import run_mcp_server
        return run_mcp_server
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")