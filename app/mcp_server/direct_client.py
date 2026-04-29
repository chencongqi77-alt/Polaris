from __future__ import annotations

import logging
import os
import sys
import shutil
from pathlib import Path
from typing import Any

from app.interfaces.mcp import MCPClient, MCPResult
from app.mcp_server.tools import MCPTools

log = logging.getLogger(__name__)

_PATCHED_ENV_KEYS = ("PATH", "VIRTUAL_ENV", "PYTHONPATH")


def _clean_env() -> dict[str, str]:
    """Return an env dict with only PATH/VIRTUAL_ENV/PYTHONPATH and SYSTEMROOT."""
    env = {k: os.environ[k] for k in _PATCHED_ENV_KEYS if k in os.environ}
    env["SYSTEMROOT"] = os.environ.get("SYSTEMROOT", "")
    return env


def _patched_subprocess_run(
    cmd: list[str],
    capture_output: bool = True,
    text: bool = True,
    timeout: int = 10,
    cwd: str | None = None,
    env: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Run *cmd* in a subprocess with clean environment.

    We drop inherited LD_PRELOAD / LD_LIBRARY_PATH / PYTHONPATH so that
    the child Python process does not accidentally pick up venv-specific
    native libs (e.g. ``langsmith`` / ``orjson``) that conflict with the
    system site-packages we are targeting.
    """
    import subprocess

    if env is None:
        env = _clean_env()

    result = subprocess.run(
        cmd,
        capture_output=capture_output,
        text=text,
        timeout=timeout,
        cwd=cwd,
        env=env,
    )
    return {
        "stdout": result.stdout,
        "stderr": result.stderr,
        "returncode": result.returncode,
    }


def _resolve_python_executable() -> str:
    """Find a Python executable that can actually import stdlib.

    If the current ``sys.executable`` is a venv that cannot import stdlib
    modules (e.g. ``pathlib``) due to missing ``_pth`` / ``sitecustomize``
    issues, fall back to the system-wide ``python3`` / ``python``.
    """
    # quick check: can current interpreter import pathlib?
    try:
        result = _patched_subprocess_run(
            [sys.executable, "-c", "import pathlib"],
            timeout=5,
        )
        if result["returncode"] == 0:
            return sys.executable
    except Exception:
        pass

    # fallback: system python3 / python
    for name in ("python3", "python"):
        exe = shutil.which(name)
        if exe and exe != sys.executable:
            try:
                result = _patched_subprocess_run([exe, "-c", "import pathlib"], timeout=5)
                if result["returncode"] == 0:
                    return exe
            except Exception:
                continue
    # last resort: keep current
    return sys.executable


class DirectMCPClient:
    """In-process MCP client that uses ``MCPTools`` directly.

    This avoids any stdio / SSE transport and works without the MCP SDK.
    Every tool call is dispatched to the corresponding ``MCPTools`` method.
    """

    # tool name → method name on MCPTools
    # Supports both flat names (execute_python) and dotted names (artifact.prepare)
    _TOOL_REGISTRY: dict[str, str] = {
        "execute_python": "execute_python",
        "search_knowledge": "search_knowledge",
        "call_tool": "call_tool",
        "save_artifact": "save_artifact",
        "web_search": "web_search",
        "create_file": "create_file",
        "read_file": "read_file",
        "list_files": "list_files",
        "calculate": "calculate",
        # Dotted name aliases (matching MCP server tool names)
        "artifact.prepare": "prepare",
        "artifact.list": "list_artifacts",
        "files.create": "create_file",
        "files.read": "read_file",
        "files.list": "list_files",
        "search.query": "search_knowledge",
        "calculate.evaluate": "calculate",
        "web_search.search": "web_search",
    }

    def __init__(self, workdir: str | None = None, config: Any | None = None) -> None:
        if workdir is None and config is not None:
            workdir = getattr(config, "workdir", None)
        self._workdir = workdir or os.getcwd()
        self._python_executable = _resolve_python_executable()
        self._tools = MCPTools(workdir=self._workdir)
        log.info("DirectMCPClient initialised (python=%s, workdir=%s)", self._python_executable, self._workdir)

    # ------------------------------------------------------------------
    # Core dispatch
    # ------------------------------------------------------------------

    def call_tool(self, tool_name: str = "", tool: str = "", arguments: dict[str, Any] | None = None) -> dict[str, Any]:
        """Dispatch a tool call to the appropriate MCPTools method.

        Accepts both ``tool`` and ``tool_name`` so callers using either
        convention work without a ``TypeError``.

        Unknown tools raise ``RuntimeError`` (tests expect this).
        Known tools return a dict with ``status`` and other keys.
        """
        resolved = tool_name or tool
        if not resolved:
            raise RuntimeError("No tool name provided")

        # Check if tool is registered
        method_name = self._TOOL_REGISTRY.get(resolved)
        if method_name is None:
            cleaned = resolved.replace("tools.", "").replace("tools_", "")
            method_name = self._TOOL_REGISTRY.get(cleaned)
        if method_name is None:
            raise RuntimeError(f"Tool '{resolved}' is not registered")

        payload = arguments or {}
        try:
            result = self._dispatch_tool(resolved, payload)
            # Return the result dict directly (preserves status like "prepared")
            if isinstance(result, dict):
                return result
            return {"status": "ok", "content": str(result)}
        except RuntimeError:
            raise
        except Exception as exc:
            return {"status": "error", "error": f"Tool '{resolved}' failed: {exc}"}

    def _dispatch_tool(self, tool: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """Map tool name to MCPTools method and call it."""
        # First try direct registry lookup
        method_name = self._TOOL_REGISTRY.get(tool)
        if method_name is None:
            # Fallback: strip "tools." / "tools_" prefix, handle dotted names
            cleaned = tool.replace("tools.", "").replace("tools_", "")
            method_name = self._TOOL_REGISTRY.get(cleaned, cleaned)

        method = getattr(self._tools, method_name, None)
        if method is None:
            raise ValueError(f"Unknown MCP tool: {tool} (resolved: {method_name})")
        
        # Build kwargs, excluding the generic 'input' key
        kwargs = {k: v for k, v in arguments.items() if k != "input"}
        
        # Map 'input' key to the method's first non-self parameter
        if "input" in arguments:
            import inspect
            sig = inspect.signature(method)
            params = [p for p in sig.parameters if p != "self"]
            if params and params[0] not in kwargs:
                kwargs[params[0]] = arguments["input"]
        
        return method(**kwargs)

    # ------------------------------------------------------------------
    # Lifecycle helpers (no-op for direct mode)
    # ------------------------------------------------------------------

    def connect(self) -> None:
        """No-op in direct mode."""

    def close(self) -> None:
        """No-op in direct mode."""

    def health_check(self) -> dict[str, Any]:
        """Return basic health info."""
        return {
            "status": "ok",
            "mode": "direct",
            "python": self._python_executable,
            "workdir": self._workdir,
            "available_tools": list(self._TOOL_REGISTRY.keys()),
        }

    def list_tools(self) -> list[str]:
        """Return list of available tool names (dotted canonical names first)."""
        return list(self._TOOL_REGISTRY.keys())

    def get_context(self, tool_name: str) -> dict[str, Any]:
        """Return context for a tool (no-op in direct mode)."""
        return {}

    # Allow attribute-style access used by MCPClient facade
    @property
    def mode(self) -> str:
        return "direct"