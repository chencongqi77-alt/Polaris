"""MCP Configuration dataclass."""
from __future__ import annotations

import os
from dataclasses import dataclass, field


@dataclass
class MCPConfig:
    """Configuration for MCP client connections."""
    mode: str = "direct"
    server_command: str = ""
    url: str = ""
    server_name: str = "macp-local"

    @classmethod
    def from_env(cls) -> MCPConfig:
        return cls(
            mode=os.environ.get("MCP_MODE", "direct"),
            server_command=os.environ.get("MCP_SERVER_COMMAND", ""),
            url=os.environ.get("MCP_URL", ""),
            server_name=os.environ.get("MCP_SERVER_NAME", "macp-local"),
        )