"""Test MCP integration with agents."""

import pytest

from app.interfaces.mcp import DirectMCPClient, MockMCPClient, create_mcp_client
from app.graph.state import MacpState


def test_direct_mcp_client_creation():
    """Test that DirectMCPClient can be created and loads tools."""
    client = DirectMCPClient()
    tools = client.list_tools()
    
    # Should have loaded all tools
    assert len(tools) > 0
    assert "artifact.prepare" in tools
    assert "file.read" in tools
    assert "search.query" in tools
    assert "calculate.math" in tools


def test_direct_mcp_client_call_tool():
    """Test that DirectMCPClient can call tools."""
    client = DirectMCPClient()
    
    # Test artifact.prepare (returns status="prepared")
    result = client.call_tool("artifact.prepare", {
        "candidate_id": "test-candidate",
        "preview": "Test preview content",
    })
    assert result["status"] == "prepared"  # artifact.prepare returns "prepared"
    assert "artifact_id" in result
    assert result["source"] == "mcp-direct"
    
    # Test calculate.math
    result = client.call_tool("calculate.math", {
        "expression": "2 + 3 * 4",
    })
    assert result["status"] == "ok"
    assert result["result"] == 14


def test_direct_mcp_client_unknown_tool():
    """Test that DirectMCPClient falls back for unknown tools."""
    mock_fallback = MockMCPClient()
    client = DirectMCPClient(fallback_client=mock_fallback)
    
    result = client.call_tool("unknown.tool", {"arg": "value"})
    assert result["source"] == "mock-mcp-fallback"
    assert "fallback_reason" in result


def test_create_mcp_client_factory():
    """Test the MCP client factory function."""
    # Mock mode
    mock_client = create_mcp_client(mode="mock")
    assert isinstance(mock_client, MockMCPClient)
    
    # Direct mode
    direct_client = create_mcp_client(mode="direct")
    assert isinstance(direct_client, DirectMCPClient)
    
    # Unknown mode defaults to mock
    unknown_client = create_mcp_client(mode="unknown")
    assert isinstance(unknown_client, MockMCPClient)


def test_mcp_client_with_artifact_workflow():
    """Test a complete artifact workflow via MCP."""
    client = DirectMCPClient()
    
    # Prepare artifact (returns status="prepared")
    prepare_result = client.call_tool("artifact.prepare", {
        "candidate_id": "workflow-test",
        "preview": "Test artifact preview",
    })
    assert prepare_result["status"] == "prepared"  # artifact.prepare returns "prepared"
    artifact_id = prepare_result["artifact_id"]
    
    # Finalize artifact (returns status="finalized")
    finalize_result = client.call_tool("artifact.finalize", {
        "artifact_id": artifact_id,
        "content": "Full artifact content for testing",
    })
    assert finalize_result["status"] == "finalized"  # artifact.finalize returns "finalized"
    assert finalize_result["artifact_id"] == artifact_id
    
    # Get artifact (returns the artifact with its current status="finalized")
    get_result = client.call_tool("artifact.get", {
        "artifact_id": artifact_id,
    })
    assert get_result["status"] == "finalized"  # artifact has been finalized
    assert get_result["content"] == "Full artifact content for testing"
    
    # List artifacts (returns {"artifacts": [...], "count": N}, no status field)
    list_result = client.call_tool("artifact.list", {
        "candidate_id": "workflow-test",
    })
    assert "artifacts" in list_result
    assert list_result["count"] >= 1


def test_mcp_client_file_operations():
    """Test file operations via MCP."""
    client = DirectMCPClient()
    
    # Check if test directory exists
    result = client.call_tool("file.exists", {"path": "."})
    assert result["status"] == "ok"
    assert result["exists"] is True
    
    # List files
    result = client.call_tool("file.list", {"directory": "."})
    assert result["status"] == "ok"
    assert result["count"] >= 0


def test_mcp_client_search_operations():
    """Test search operations via MCP."""
    client = DirectMCPClient()
    
    # Index some documents
    import json
    docs = json.dumps([
        {"id": "doc1", "content": "Hello world test document"},
        {"id": "doc2", "content": "Another test document for search"},
    ])
    result = client.call_tool("search.index", {
        "scope": "test-scope",
        "documents": docs,
    })
    assert result["status"] == "ok"
    assert result["indexed_count"] == 2
    
    # Search
    result = client.call_tool("search.query", {
        "scope": "test-scope",
        "query": "test",
        "top_k": 5,
    })
    assert result["status"] == "ok"
    assert result["count"] >= 2
    
    # Clear
    result = client.call_tool("search.clear", {"scope": "test-scope"})
    assert result["status"] == "ok"


def test_mcp_client_calculate_operations():
    """Test calculate operations via MCP."""
    client = DirectMCPClient()
    
    # Math
    result = client.call_tool("calculate.math", {"expression": "sqrt(16) + 2"})
    assert result["status"] == "ok"
    assert result["result"] == 6.0
    
    # Stats
    result = client.call_tool("calculate.stats", {"numbers": "1, 2, 3, 4, 5"})
    assert result["status"] == "ok"
    assert result["mean"] == 3.0
    assert result["median"] == 3.0
    
    # Format number
    result = client.call_tool("calculate.format_number", {
        "value": 3.14159,
        "decimals": 2,
    })
    assert result["status"] == "ok"
    assert result["formatted"] == 3.14


if __name__ == "__main__":
    pytest.main([__file__, "-v"])