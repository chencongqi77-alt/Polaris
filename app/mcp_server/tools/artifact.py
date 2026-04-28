"""Artifact tools for MACP agents.

Tools for managing candidate artifacts during solution generation.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime
from typing import Any, Dict

# In-memory artifact store (for demo/testing; replace with persistent store in production)
_artifact_store: Dict[str, Dict[str, Any]] = {}


def register_artifact_tools(mcp_server) -> None:
    """Register artifact tools to the MCP server.

    Args:
        mcp_server: The MCP Server instance with a @tool decorator or add_tool method.
    """
    # Support both decorator style and method style
    if hasattr(mcp_server, "tool"):
        # Use decorator style
        @mcp_server.tool()
        def artifact_prepare(candidate_id: str, preview: str) -> str:
            """Prepare a new artifact for a candidate solution.

            Args:
                candidate_id: Unique identifier for the candidate.
                preview: Preview text of the artifact content.

            Returns:
                JSON string with artifact metadata.
            """
            artifact_id = f"art-{uuid.uuid4().hex[:8]}"
            artifact = {
                "artifact_id": artifact_id,
                "candidate_id": candidate_id,
                "preview": preview[:500],
                "status": "prepared",
                "created_at": datetime.now().isoformat(),
                "size_bytes": len(preview.encode("utf-8")),
            }
            _artifact_store[artifact_id] = artifact
            return json.dumps(artifact, ensure_ascii=False)

        @mcp_server.tool()
        def artifact_finalize(artifact_id: str, content: str) -> str:
            """Finalize an artifact with full content.

            Args:
                artifact_id: The artifact ID returned by artifact_prepare.
                content: Full content of the artifact.

            Returns:
                JSON string with finalized artifact metadata.
            """
            if artifact_id not in _artifact_store:
                return json.dumps({
                    "error": f"Artifact {artifact_id} not found",
                    "status": "failed",
                })

            artifact = _artifact_store[artifact_id]
            artifact["content"] = content
            artifact["status"] = "finalized"
            artifact["finalized_at"] = datetime.now().isoformat()
            artifact["size_bytes"] = len(content.encode("utf-8"))
            return json.dumps(artifact, ensure_ascii=False)

        @mcp_server.tool()
        def artifact_get(artifact_id: str) -> str:
            """Retrieve an artifact by ID.

            Args:
                artifact_id: The artifact ID.

            Returns:
                JSON string with artifact data or error.
            """
            if artifact_id not in _artifact_store:
                return json.dumps({
                    "error": f"Artifact {artifact_id} not found",
                    "status": "not_found",
                })
            return json.dumps(_artifact_store[artifact_id], ensure_ascii=False)

        @mcp_server.tool()
        def artifact_list(candidate_id: str = "") -> str:
            """List all artifacts, optionally filtered by candidate_id.

            Args:
                candidate_id: Optional candidate ID to filter by.

            Returns:
                JSON string with list of artifacts.
            """
            if candidate_id:
                artifacts = [
                    a for a in _artifact_store.values()
                    if a.get("candidate_id") == candidate_id
                ]
            else:
                artifacts = list(_artifact_store.values())
            return json.dumps({
                "artifacts": artifacts,
                "count": len(artifacts),
            }, ensure_ascii=False)

    elif hasattr(mcp_server, "add_tool"):
        # Use method style
        mcp_server.add_tool(artifact_prepare)
        mcp_server.add_tool(artifact_finalize)
        mcp_server.add_tool(artifact_get)
        mcp_server.add_tool(artifact_list)


# Standalone tool functions for direct import (used by StdioMCPClient)
def artifact_prepare(candidate_id: str, preview: str) -> str:
    """Prepare a new artifact for a candidate solution."""
    artifact_id = f"art-{uuid.uuid4().hex[:8]}"
    artifact = {
        "artifact_id": artifact_id,
        "candidate_id": candidate_id,
        "preview": preview[:500],
        "status": "prepared",
        "created_at": datetime.now().isoformat(),
        "size_bytes": len(preview.encode("utf-8")),
    }
    _artifact_store[artifact_id] = artifact
    return json.dumps(artifact, ensure_ascii=False)


def artifact_finalize(artifact_id: str, content: str) -> str:
    """Finalize an artifact with full content."""
    if artifact_id not in _artifact_store:
        return json.dumps({
            "error": f"Artifact {artifact_id} not found",
            "status": "failed",
        })
    artifact = _artifact_store[artifact_id]
    artifact["content"] = content
    artifact["status"] = "finalized"
    artifact["finalized_at"] = datetime.now().isoformat()
    artifact["size_bytes"] = len(content.encode("utf-8"))
    return json.dumps(artifact, ensure_ascii=False)


def artifact_get(artifact_id: str) -> str:
    """Retrieve an artifact by ID."""
    if artifact_id not in _artifact_store:
        return json.dumps({
            "error": f"Artifact {artifact_id} not found",
            "status": "not_found",
        })
    return json.dumps(_artifact_store[artifact_id], ensure_ascii=False)


def artifact_list(candidate_id: str = "") -> str:
    """List all artifacts, optionally filtered by candidate_id."""
    if candidate_id:
        artifacts = [
            a for a in _artifact_store.values()
            if a.get("candidate_id") == candidate_id
        ]
    else:
        artifacts = list(_artifact_store.values())
    return json.dumps({
        "artifacts": artifacts,
        "count": len(artifacts),
    }, ensure_ascii=False)