"""Search tools for MACP agents.

Tools for searching content (in-memory and simple text search).
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List


# In-memory search index for demo/testing
_search_index: Dict[str, List[Dict[str, Any]]] = {}


def register_search_tools(mcp_server) -> None:
    """Register search tools to the MCP server."""

    if hasattr(mcp_server, "tool"):
        @mcp_server.tool()
        def search_index(scope: str, documents: str) -> str:
            """Index documents for later search.

            Args:
                scope: Search scope/namespace (e.g., "knowledge", "history").
                documents: JSON string of document list, each with "id" and "content".

            Returns:
                JSON string with indexing result.
            """
            try:
                docs = json.loads(documents) if isinstance(documents, str) else documents
                if not isinstance(docs, list):
                    return json.dumps({
                        "error": "documents must be a list",
                        "status": "error",
                    })
                
                indexed = []
                for doc in docs:
                    if not isinstance(doc, dict) or "id" not in doc or "content" not in doc:
                        continue
                    if scope not in _search_index:
                        _search_index[scope] = []
                    _search_index[scope].append({
                        "id": doc["id"],
                        "content": doc["content"],
                        "metadata": doc.get("metadata", {}),
                    })
                    indexed.append(doc["id"])
                
                return json.dumps({
                    "scope": scope,
                    "indexed_count": len(indexed),
                    "indexed_ids": indexed,
                    "status": "ok",
                }, ensure_ascii=False)
            except Exception as e:
                return json.dumps({"error": str(e), "status": "error"})

        @mcp_server.tool()
        def search_query(scope: str, query: str, top_k: int = 5) -> str:
            """Search indexed documents.

            Args:
                scope: Search scope/namespace to search in.
                query: Search query string.
                top_k: Maximum number of results to return.

            Returns:
                JSON string with search results.
            """
            if scope not in _search_index:
                return json.dumps({
                    "scope": scope,
                    "query": query,
                    "results": [],
                    "count": 0,
                    "status": "ok",
                }, ensure_ascii=False)
            
            # Simple keyword matching search
            query_terms = set(query.lower().split())
            scored = []
            for doc in _search_index[scope]:
                content_lower = doc["content"].lower()
                score = sum(1 for term in query_terms if term in content_lower)
                if score > 0:
                    scored.append((score, doc))
            
            # Sort by score descending
            scored.sort(key=lambda x: x[0], reverse=True)
            results = [doc for _, doc in scored[:top_k]]
            
            return json.dumps({
                "scope": scope,
                "query": query,
                "results": results,
                "count": len(results),
                "status": "ok",
            }, ensure_ascii=False)

        @mcp_server.tool()
        def search_regex(scope: str, pattern: str, top_k: int = 5) -> str:
            """Search indexed documents with regex pattern.

            Args:
                scope: Search scope/namespace to search in.
                pattern: Regex pattern to match.
                top_k: Maximum number of results to return.

            Returns:
                JSON string with search results.
            """
            if scope not in _search_index:
                return json.dumps({
                    "scope": scope,
                    "pattern": pattern,
                    "results": [],
                    "count": 0,
                    "status": "ok",
                }, ensure_ascii=False)
            
            try:
                regex = re.compile(pattern, re.IGNORECASE)
            except re.error as e:
                return json.dumps({
                    "error": f"Invalid regex: {e}",
                    "status": "error",
                })
            
            results = []
            for doc in _search_index[scope]:
                if regex.search(doc["content"]):
                    results.append(doc)
                    if len(results) >= top_k:
                        break
            
            return json.dumps({
                "scope": scope,
                "pattern": pattern,
                "results": results,
                "count": len(results),
                "status": "ok",
            }, ensure_ascii=False)

        @mcp_server.tool()
        def search_clear(scope: str = "") -> str:
            """Clear search index.

            Args:
                scope: Scope to clear (empty string clears all).

            Returns:
                JSON string with clear result.
            """
            global _search_index
            if scope:
                if scope in _search_index:
                    del _search_index[scope]
                    return json.dumps({
                        "scope": scope,
                        "cleared": True,
                        "status": "ok",
                    })
                return json.dumps({
                    "scope": scope,
                    "cleared": False,
                    "status": "ok",
                })
            else:
                _search_index = {}
                return json.dumps({
                    "cleared_all": True,
                    "status": "ok",
                })

    elif hasattr(mcp_server, "add_tool"):
        mcp_server.add_tool(search_index)
        mcp_server.add_tool(search_query)
        mcp_server.add_tool(search_regex)
        mcp_server.add_tool(search_clear)


# Standalone functions for direct import
def search_index(scope: str, documents: str) -> str:
    """Index documents for later search."""
    try:
        docs = json.loads(documents) if isinstance(documents, str) else documents
        if not isinstance(docs, list):
            return json.dumps({
                "error": "documents must be a list",
                "status": "error",
            })
        
        indexed = []
        for doc in docs:
            if not isinstance(doc, dict) or "id" not in doc or "content" not in doc:
                continue
            if scope not in _search_index:
                _search_index[scope] = []
            _search_index[scope].append({
                "id": doc["id"],
                "content": doc["content"],
                "metadata": doc.get("metadata", {}),
            })
            indexed.append(doc["id"])
        
        return json.dumps({
            "scope": scope,
            "indexed_count": len(indexed),
            "indexed_ids": indexed,
            "status": "ok",
        }, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": str(e), "status": "error"})


def search_query(scope: str, query: str, top_k: int = 5) -> str:
    """Search indexed documents."""
    if scope not in _search_index:
        return json.dumps({
            "scope": scope,
            "query": query,
            "results": [],
            "count": 0,
            "status": "ok",
        }, ensure_ascii=False)
    
    query_terms = set(query.lower().split())
    scored = []
    for doc in _search_index[scope]:
        content_lower = doc["content"].lower()
        score = sum(1 for term in query_terms if term in content_lower)
        if score > 0:
            scored.append((score, doc))
    
    scored.sort(key=lambda x: x[0], reverse=True)
    results = [doc for _, doc in scored[:top_k]]
    
    return json.dumps({
        "scope": scope,
        "query": query,
        "results": results,
        "count": len(results),
        "status": "ok",
    }, ensure_ascii=False)


def search_regex(scope: str, pattern: str, top_k: int = 5) -> str:
    """Search indexed documents with regex pattern."""
    if scope not in _search_index:
        return json.dumps({
            "scope": scope,
            "pattern": pattern,
            "results": [],
            "count": 0,
            "status": "ok",
        }, ensure_ascii=False)
    
    try:
        regex = re.compile(pattern, re.IGNORECASE)
    except re.error as e:
        return json.dumps({
            "error": f"Invalid regex: {e}",
            "status": "error",
        })
    
    results = []
    for doc in _search_index[scope]:
        if regex.search(doc["content"]):
            results.append(doc)
            if len(results) >= top_k:
                break
    
    return json.dumps({
        "scope": scope,
        "pattern": pattern,
        "results": results,
        "count": len(results),
        "status": "ok",
    }, ensure_ascii=False)


def search_clear(scope: str = "") -> str:
    """Clear search index."""
    global _search_index
    if scope:
        if scope in _search_index:
            del _search_index[scope]
            return json.dumps({
                "scope": scope,
                "cleared": True,
                "status": "ok",
            })
        return json.dumps({
            "scope": scope,
            "cleared": False,
            "status": "ok",
        })
    else:
        _search_index = {}
        return json.dumps({
            "cleared_all": True,
            "status": "ok",
        })