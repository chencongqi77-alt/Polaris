"""MACP MCP Server Tools.

Exports:
- MCP_TOOLS_LIST: List of tool definitions for the MCP Server
- MCPTools: Unified tool dispatcher class for direct in-process usage
"""

from __future__ import annotations

import json
import os
import sys
import uuid
from datetime import datetime
from typing import Any, Dict

# Re-export all tool functions for direct import
from .artifact import (
    artifact_finalize,
    artifact_get,
    artifact_list,
    artifact_prepare,
    register_artifact_tools,
)
from .calculate import (
    calculate_math,
    calculate_stats,
    calculate_format_number,
    calculate_time_diff,
    register_calculate_tools,
)
from .files import (
    file_exists,
    file_list,
    file_read,
    file_write,
    register_file_tools,
)
from .search import (
    search_clear,
    search_index,
    search_query,
    search_regex,
    register_search_tools,
)
from .web_search import register_web_search_tools


class MCPTools:
    """Unified MCP tool dispatcher.

    Wraps all standalone tool functions into a single object with named
    methods that :class:`DirectMCPClient` can call by attribute lookup.
    """

    def __init__(self, workdir: str | None = None) -> None:
        self._workdir = workdir or os.getcwd()

    # ------------------------------------------------------------------
    # Code execution
    # ------------------------------------------------------------------

    def execute_python(self, code: str = "", input: str = "", timeout: int = 30, **_: Any) -> dict[str, Any]:
        """Execute Python code in a sandboxed subprocess and return stdout/stderr."""
        code = code or input
        if not code.strip():
            return {"status": "error", "error": "No code provided"}

        import subprocess
        import tempfile

        # Write code to a temp file to avoid shell escaping issues
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False, encoding="utf-8"
        ) as f:
            f.write(code)
            tmp_path = f.name

        try:
            result = subprocess.run(
                [sys.executable, tmp_path],
                capture_output=True,
                text=True,
                timeout=min(timeout, 60),
                cwd=self._workdir,
            )
            return {
                "status": "ok" if result.returncode == 0 else "error",
                "stdout": result.stdout,
                "stderr": result.stderr,
                "returncode": result.returncode,
            }
        except subprocess.TimeoutExpired:
            return {"status": "error", "error": f"Code execution timed out ({timeout}s)"}
        except Exception as exc:
            return {"status": "error", "error": str(exc)}
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    # ------------------------------------------------------------------
    # Knowledge search
    # ------------------------------------------------------------------

    def search_knowledge(self, query: str = "", input: str = "", scope: str = "knowledge", top_k: int = 5, **_: Any) -> dict[str, Any]:
        """Search the in-memory knowledge index."""
        query = query or input
        raw = search_query(scope=scope, query=query, top_k=top_k)
        try:
            return json.loads(raw)
        except Exception:
            return {"status": "ok", "results": [], "raw": raw}

    # ------------------------------------------------------------------
    # Generic tool call (dispatch by name)
    # ------------------------------------------------------------------

    def call_tool(self, tool_name: str = "", tool: str = "", arguments: dict[str, Any] | None = None, **_: Any) -> dict[str, Any]:
        """Dispatch a tool call by name."""
        resolved = tool_name or tool
        payload = arguments or {}
        method = getattr(self, resolved, None)
        if method and method is not self.call_tool:
            return method(**payload)
        return {"status": "error", "error": f"Unknown tool: {resolved}"}

    # ------------------------------------------------------------------
    # Artifact management
    # ------------------------------------------------------------------

    def prepare(self, candidate_id: str = "", preview: str = "", **_: Any) -> dict[str, Any]:
        """Prepare an artifact (thin wrapper around artifact_prepare)."""
        raw = artifact_prepare(candidate_id=candidate_id, preview=preview)
        try:
            return json.loads(raw)
        except Exception:
            return {"status": "ok", "raw": raw}

    def list_artifacts(self, **_: Any) -> dict[str, Any]:
        """List all artifacts (thin wrapper around artifact_list)."""
        raw = artifact_list()
        try:
            return json.loads(raw)
        except Exception:
            return {"status": "ok", "raw": raw}

    def save_artifact(self, content: str = "", candidate_id: str = "", title: str = "", **_: Any) -> dict[str, Any]:
        """Prepare and finalize an artifact in one step."""
        preview = title or content[:200]
        raw = artifact_prepare(candidate_id=candidate_id or f"cand-{uuid.uuid4().hex[:8]}", preview=preview)
        try:
            meta = json.loads(raw)
        except Exception:
            return {"status": "error", "error": "Failed to prepare artifact"}

        artifact_id = meta.get("artifact_id", "")
        if content and artifact_id:
            raw2 = artifact_finalize(artifact_id=artifact_id, content=content)
            try:
                return json.loads(raw2)
            except Exception:
                pass
        return meta

    # ------------------------------------------------------------------
    # Web search
    # ------------------------------------------------------------------

    def web_search(self, query: str = "", input: str = "", num_results: int = 5, **_: Any) -> dict[str, Any]:
        """Search the web (DuckDuckGo fallback)."""
        query = query or input
        if not query.strip():
            return {"status": "error", "error": "No query provided"}

        try:
            from .web_search import web_search as _ws
            raw = _ws(query=query, max_results=num_results)
            if isinstance(raw, str):
                return json.loads(raw)
            return raw
        except ImportError:
            return {"status": "error", "error": "web_search module not available"}
        except Exception as exc:
            return {"status": "error", "error": str(exc)}

    # ------------------------------------------------------------------
    # File operations
    # ------------------------------------------------------------------

    def create_file(self, path: str = "", content: str = "", filename: str = "", **_: Any) -> dict[str, Any]:
        """Create / overwrite a file."""
        path = path or filename
        if not path:
            return {"status": "error", "error": "No path provided"}
        raw = file_write(path=path, content=content)
        try:
            return json.loads(raw)
        except Exception:
            return {"status": "ok", "raw": raw}

    def read_file(self, path: str = "", filename: str = "", **_: Any) -> dict[str, Any]:
        """Read a file."""
        path = path or filename
        if not path:
            return {"status": "error", "error": "No path provided"}
        raw = file_read(path=path)
        try:
            return json.loads(raw)
        except Exception:
            return {"status": "ok", "raw": raw}

    def list_files(self, directory: str = ".", path: str = "", **_: Any) -> dict[str, Any]:
        """List files in a directory."""
        directory = path or directory
        raw = file_list(directory=directory)
        try:
            return json.loads(raw)
        except Exception:
            return {"status": "ok", "raw": raw}

    # ------------------------------------------------------------------
    # Calculation
    # ------------------------------------------------------------------

    def calculate(self, expression: str = "", input: str = "", **_: Any) -> dict[str, Any]:
        """Evaluate a math expression safely."""
        expression = expression or input
        if not expression.strip():
            return {"status": "error", "error": "No expression provided"}
        raw = calculate_math(expression=expression)
        try:
            return json.loads(raw)
        except Exception:
            return {"status": "ok", "raw": raw}


__all__ = [
    # Registration helpers (for MCP server)
    "register_artifact_tools",
    "register_calculate_tools",
    "register_file_tools",
    "register_search_tools",
    "register_web_search_tools",
    # Standalone tool functions
    "artifact_prepare",
    "artifact_finalize",
    "artifact_get",
    "artifact_list",
    "calculate_math",
    "calculate_stats",
    "calculate_format_number",
    "calculate_time_diff",
    "file_read",
    "file_write",
    "file_list",
    "file_exists",
    "search_index",
    "search_query",
    "search_regex",
    "search_clear",
    # Unified dispatcher
    "MCPTools",
    # Legacy list (kept for backward compat)
    "MCP_TOOLS_LIST",
]


def register_all_tools(mcp_server) -> None:
    """Register all tools to the given MCP server instance.

    Args:
        mcp_server: The MCP Server instance.
    """
    register_artifact_tools(mcp_server)
    register_calculate_tools(mcp_server)
    register_file_tools(mcp_server)
    register_search_tools(mcp_server)
    register_web_search_tools(mcp_server)


# Backward-compatible list
MCP_TOOLS_LIST: list[dict[str, Any]] = []