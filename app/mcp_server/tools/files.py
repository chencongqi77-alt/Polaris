"""File tools for MACP agents.

Tools for reading, writing, and managing files.
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List


# Allowed base directories for file operations (security sandbox)
_ALLOWED_BASE_DIRS: List[str] = [
    os.getcwd(),  # Current working directory
    os.path.join(os.getcwd(), "app_data"),  # App data directory
]


def _resolve_path(path: str) -> Path:
    """Resolve and validate a path within allowed directories.

    Uses dynamic CWD so tests that os.chdir() into tmp_path still work.
    """
    p = Path(path).resolve()
    dynamic_bases = [
        Path.cwd(),
        Path.cwd() / "app_data",
    ]
    # Also include the originally imported bases for robustness
    for base_dir in _ALLOWED_BASE_DIRS:
        bp = Path(base_dir).resolve()
        if bp not in dynamic_bases:
            dynamic_bases.append(bp)
    for base_dir in dynamic_bases:
        try:
            p.relative_to(Path(base_dir).resolve())
            return p
        except ValueError:
            continue
    raise PermissionError(f"Path '{path}' is outside allowed directories")


def register_file_tools(mcp_server) -> None:
    """Register file tools to the MCP server."""

    if hasattr(mcp_server, "tool"):
        @mcp_server.tool()
        def file_read(path: str, encoding: str = "utf-8") -> str:
            """Read content from a file.

            Args:
                path: File path relative to allowed base directory.
                encoding: File encoding (default: utf-8).

            Returns:
                JSON string with file content or error.
            """
            try:
                resolved = _resolve_path(path)
                if not resolved.exists():
                    return json.dumps({
                        "error": f"File not found: {path}",
                        "status": "not_found",
                    })
                content = resolved.read_text(encoding=encoding)
                return json.dumps({
                    "path": str(resolved),
                    "content": content,
                    "size_bytes": len(content.encode(encoding)),
                    "status": "ok",
                }, ensure_ascii=False)
            except PermissionError as e:
                return json.dumps({"error": str(e), "status": "forbidden"})
            except Exception as e:
                return json.dumps({"error": str(e), "status": "error"})

        @mcp_server.tool()
        def file_write(path: str, content: str, encoding: str = "utf-8") -> str:
            """Write content to a file.

            Args:
                path: File path relative to allowed base directory.
                content: Content to write.
                encoding: File encoding (default: utf-8).

            Returns:
                JSON string with write result.
            """
            try:
                resolved = _resolve_path(path)
                resolved.parent.mkdir(parents=True, exist_ok=True)
                resolved.write_text(content, encoding=encoding)
                return json.dumps({
                    "path": str(resolved),
                    "size_bytes": len(content.encode(encoding)),
                    "status": "ok",
                    "written_at": datetime.now().isoformat(),
                }, ensure_ascii=False)
            except PermissionError as e:
                return json.dumps({"error": str(e), "status": "forbidden"})
            except Exception as e:
                return json.dumps({"error": str(e), "status": "error"})

        @mcp_server.tool()
        def file_list(directory: str = ".") -> str:
            """List files in a directory.

            Args:
                directory: Directory path to list (default: current directory).

            Returns:
                JSON string with list of files and directories.
            """
            try:
                resolved = _resolve_path(directory)
                if not resolved.exists():
                    return json.dumps({
                        "error": f"Directory not found: {directory}",
                        "status": "not_found",
                    })
                if not resolved.is_dir():
                    return json.dumps({
                        "error": f"Not a directory: {directory}",
                        "status": "error",
                    })
                items = []
                for item in resolved.iterdir():
                    items.append({
                        "name": item.name,
                        "type": "directory" if item.is_dir() else "file",
                        "size": item.stat().st_size if item.is_file() else None,
                    })
                return json.dumps({
                    "directory": str(resolved),
                    "items": items,
                    "count": len(items),
                    "status": "ok",
                }, ensure_ascii=False)
            except PermissionError as e:
                return json.dumps({"error": str(e), "status": "forbidden"})
            except Exception as e:
                return json.dumps({"error": str(e), "status": "error"})

        @mcp_server.tool()
        def file_exists(path: str) -> str:
            """Check if a file or directory exists.

            Args:
                path: Path to check.

            Returns:
                JSON string with existence status.
            """
            try:
                resolved = _resolve_path(path)
                exists = resolved.exists()
                return json.dumps({
                    "path": str(resolved),
                    "exists": exists,
                    "type": "directory" if resolved.is_dir() else "file" if resolved.is_file() else "none",
                    "status": "ok",
                }, ensure_ascii=False)
            except PermissionError as e:
                return json.dumps({"error": str(e), "status": "forbidden"})
            except Exception as e:
                return json.dumps({"error": str(e), "status": "error"})

    elif hasattr(mcp_server, "add_tool"):
        mcp_server.add_tool(file_read)
        mcp_server.add_tool(file_write)
        mcp_server.add_tool(file_list)
        mcp_server.add_tool(file_exists)


# Standalone functions for direct import
def file_read(path: str, encoding: str = "utf-8") -> str:
    """Read content from a file."""
    try:
        resolved = _resolve_path(path)
        if not resolved.exists():
            return json.dumps({
                "error": f"File not found: {path}",
                "status": "not_found",
            })
        content = resolved.read_text(encoding=encoding)
        return json.dumps({
            "path": str(resolved),
            "content": content,
            "size_bytes": len(content.encode(encoding)),
            "status": "ok",
        }, ensure_ascii=False)
    except PermissionError as e:
        return json.dumps({"error": str(e), "status": "forbidden"})
    except Exception as e:
        return json.dumps({"error": str(e), "status": "error"})


def file_write(path: str, content: str, encoding: str = "utf-8") -> str:
    """Write content to a file."""
    try:
        resolved = _resolve_path(path)
        resolved.parent.mkdir(parents=True, exist_ok=True)
        resolved.write_text(content, encoding=encoding)
        return json.dumps({
            "path": str(resolved),
            "size_bytes": len(content.encode(encoding)),
            "status": "ok",
            "written_at": datetime.now().isoformat(),
        }, ensure_ascii=False)
    except PermissionError as e:
        return json.dumps({"error": str(e), "status": "forbidden"})
    except Exception as e:
        return json.dumps({"error": str(e), "status": "error"})


def file_list(directory: str = ".") -> str:
    """List files in a directory."""
    try:
        resolved = _resolve_path(directory)
        if not resolved.exists():
            return json.dumps({
                "error": f"Directory not found: {directory}",
                "status": "not_found",
            })
        if not resolved.is_dir():
            return json.dumps({
                "error": f"Not a directory: {directory}",
                "status": "error",
            })
        items = []
        for item in resolved.iterdir():
            items.append({
                "name": item.name,
                "type": "directory" if item.is_dir() else "file",
                "size": item.stat().st_size if item.is_file() else None,
            })
        return json.dumps({
            "directory": str(resolved),
            "items": items,
            "count": len(items),
            "status": "ok",
        }, ensure_ascii=False)
    except PermissionError as e:
        return json.dumps({"error": str(e), "status": "forbidden"})
    except Exception as e:
        return json.dumps({"error": str(e), "status": "error"})


def file_exists(path: str) -> str:
    """Check if a file or directory exists."""
    try:
        resolved = _resolve_path(path)
        exists = resolved.exists()
        return json.dumps({
            "path": str(resolved),
            "exists": exists,
            "type": "directory" if resolved.is_dir() else "file" if resolved.is_file() else "none",
            "status": "ok",
        }, ensure_ascii=False)
    except PermissionError as e:
        return json.dumps({"error": str(e), "status": "forbidden"})
    except Exception as e:
        return json.dumps({"error": str(e), "status": "error"})