"""Web search tools for MACP agents.

Provides GitHub repository search and general web search capabilities.
"""

from __future__ import annotations

import json
import logging
import os
import urllib.request
import urllib.parse
import urllib.error
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def _github_search(query: str, per_page: int = 5, sort: str = "stars") -> Dict[str, Any]:
    """Search GitHub repositories via the public API.

    Args:
        query: Search query string.
        per_page: Number of results to return (max 100).
        sort: Sort order ('stars', 'forks', 'updated', 'best-match').

    Returns:
        Dict with search results.
    """
    params = urllib.parse.urlencode({
        "q": query,
        "per_page": min(per_page, 20),
        "sort": sort,
        "order": "desc",
    })
    url = f"https://api.github.com/search/repositories?{params}"

    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": "MACP-Agent/1.0",
    }

    token = os.environ.get("GITHUB_TOKEN", "")
    if token:
        headers["Authorization"] = f"Bearer {token}"

    try:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode("utf-8"))

        items: List[Dict[str, Any]] = []
        for repo in data.get("items", []):
            items.append({
                "name": repo.get("full_name", ""),
                "description": repo.get("description", ""),
                "url": repo.get("html_url", ""),
                "stars": repo.get("stargazers_count", 0),
                "language": repo.get("language", ""),
                "topics": repo.get("topics", []),
                "updated_at": repo.get("updated_at", ""),
            })

        return {
            "query": query,
            "total_count": data.get("total_count", 0),
            "results": items,
            "status": "ok",
        }
    except urllib.error.HTTPError as exc:
        body = ""
        try:
            body = exc.read().decode("utf-8", errors="replace")[:500]
        except Exception:
            pass
        return {
            "query": query,
            "error": f"GitHub API HTTP {exc.code}: {body}",
            "results": [],
            "status": "error",
        }
    except Exception as exc:
        return {
            "query": query,
            "error": str(exc),
            "results": [],
            "status": "error",
        }


def _web_search_duckduckgo(query: str, max_results: int = 5) -> Dict[str, Any]:
    """Simple web search using DuckDuckGo instant answer API.

    This is a lightweight fallback. For production use, consider
    integrating with a proper search API (Bing, Google, etc.).

    Args:
        query: Search query string.
        max_results: Max results to return.

    Returns:
        Dict with search results.
    """
    params = urllib.parse.urlencode({
        "q": query,
        "format": "json",
        "no_html": 1,
        "skip_disambig": 1,
    })
    url = f"https://api.duckduckgo.com/?{params}"

    try:
        req = urllib.request.Request(url, headers={"User-Agent": "MACP-Agent/1.0"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8"))

        results: List[Dict[str, Any]] = []

        # Abstract (main answer)
        abstract = data.get("AbstractText", "")
        if abstract:
            results.append({
                "title": data.get("Heading", ""),
                "snippet": abstract,
                "url": data.get("AbstractURL", ""),
            })

        # Related topics
        for topic in data.get("RelatedTopics", [])[:max_results]:
            if isinstance(topic, dict) and "Text" in topic:
                results.append({
                    "title": topic.get("Text", "")[:80],
                    "snippet": topic.get("Text", ""),
                    "url": topic.get("FirstURL", ""),
                })
            if len(results) >= max_results:
                break

        return {
            "query": query,
            "results": results[:max_results],
            "count": len(results[:max_results]),
            "status": "ok",
        }
    except Exception as exc:
        return {
            "query": query,
            "error": str(exc),
            "results": [],
            "status": "error",
        }


def web_search_github(query: str, per_page: int = 5, sort: str = "stars") -> str:
    """Search GitHub repositories.

    Args:
        query: Search query (e.g. 'tinyml embedded', 'LLM agent framework').
        per_page: Number of results (default 5, max 20).
        sort: Sort by 'stars', 'forks', 'updated', or 'best-match'.

    Returns:
        JSON string with search results.
    """
    result = _github_search(query, per_page=per_page, sort=sort)
    return json.dumps(result, ensure_ascii=False, indent=2)


def web_search(query: str, max_results: int = 5) -> str:
    """General web search.

    Args:
        query: Search query string.
        max_results: Max results to return.

    Returns:
        JSON string with search results.
    """
    result = _web_search_duckduckgo(query, max_results=max_results)
    return json.dumps(result, ensure_ascii=False, indent=2)


def register_web_search_tools(mcp_server) -> None:
    """Register web search tools to the MCP server."""
    if hasattr(mcp_server, "tool"):
        @mcp_server.tool()
        def search_github(query: str, per_page: int = 5, sort: str = "stars") -> str:
            """Search GitHub repositories by keyword.

            Args:
                query: Search query (e.g. 'tinyml embedded', 'LLM agent framework').
                per_page: Number of results (default 5, max 20).
                sort: Sort by 'stars', 'forks', 'updated', or 'best-match'.

            Returns:
                JSON string with matching repositories.
            """
            return web_search_github(query, per_page=per_page, sort=sort)

        @mcp_server.tool()
        def search_web(query: str, max_results: int = 5) -> str:
            """Search the web for information.

            Args:
                query: Search query string.
                max_results: Max results to return.

            Returns:
                JSON string with search results.
            """
            return web_search(query, max_results=max_results)

    elif hasattr(mcp_server, "add_tool"):
        mcp_server.add_tool(web_search_github)
        mcp_server.add_tool(web_search)
