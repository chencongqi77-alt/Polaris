"""MCP tools collection."""

from app.mcp_server.tools.artifact import register_artifact_tools
from app.mcp_server.tools.files import register_file_tools
from app.mcp_server.tools.search import register_search_tools
from app.mcp_server.tools.calculate import register_calculate_tools

__all__ = [
    "register_artifact_tools",
    "register_file_tools",
    "register_search_tools",
    "register_calculate_tools",
]