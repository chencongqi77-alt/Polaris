"""CLI entry point for running the MCP server via ``python -m app.mcp_server``."""

from __future__ import annotations

from app.mcp_server.server import run_mcp_server


if __name__ == "__main__":
    run_mcp_server()