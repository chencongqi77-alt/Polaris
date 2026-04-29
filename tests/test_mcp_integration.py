"""专项测试：MCP 集成——DirectMCPClient 真实调用。"""

import os
import pytest
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

from app.interfaces.mcp import create_mcp_client, MCPClient
from app.mcp_server.direct_client import DirectMCPClient


@pytest.fixture(scope="module")
def client() -> DirectMCPClient:
    return DirectMCPClient(workdir=os.path.join(os.path.dirname(__file__), ".."))


# ---------- 工具列表测试 ----------

def test_direct_mcp_client_creation(client: DirectMCPClient) -> None:
    tools = client.list_tools()
    # 验证核心工具存在（使用实际注册的工具名）
    assert "execute_python" in tools
    assert "save_artifact" in tools
    assert "create_file" in tools
    assert "read_file" in tools
    assert "calculate" in tools
    assert "search_knowledge" in tools
    # 验证 dotted alias 存在
    assert "artifact.prepare" in tools
    assert "artifact.list" in tools


# ---------- 工具调用测试 ----------

def test_direct_mcp_client_call_tool(client: DirectMCPClient) -> None:
    result = client.call_tool("save_artifact", {
        "artifact_id": "test_mcp",
        "content": "test content",
        "artifact_type": "text",
        "summary": "test",
    })
    assert result["status"] == "saved"


def test_direct_mcp_client_unknown_tool(client: DirectMCPClient) -> None:
    with pytest.raises(RuntimeError, match="not registered"):
        client.call_tool("this_tool_does_not_exist")


# ---------- 工厂函数测试 ----------

def test_create_mcp_client_factory() -> None:
    # 默认创建 direct 模式 → 返回 MCPClient 包装
    default_client = create_mcp_client()
    assert isinstance(default_client, MCPClient)

    # 显式传入 direct 模式 → 包装后的 MCPClient
    direct_wrapped = create_mcp_client(mode="direct")
    assert isinstance(direct_wrapped, MCPClient)


# ---------- Artifact 工作流 ----------

def test_mcp_client_with_artifact_workflow(client: DirectMCPClient) -> None:
    prepare_result = client.call_tool("artifact.prepare", {
        "candidate_id": "mcp_test_001",
        "preview": "MCP test preview",
    })
    assert prepare_result["status"] == "prepared"

    save_result = client.call_tool("save_artifact", {
        "artifact_id": "mcp_test_001",
        "content": "MCP test final content",
        "artifact_type": "text",
        "summary": "MCP integration test",
    })
    assert save_result["status"] == "saved"

    list_result = client.call_tool("artifact.list", {})
    assert list_result["status"] == "ok"


# ---------- 文件操作 ----------

def test_mcp_client_file_operations(client: DirectMCPClient) -> None:
    import tempfile, os
    with tempfile.TemporaryDirectory() as tmp:
        test_file = os.path.join(tmp, "mcp_test.txt")
        result = client.call_tool("create_file", {
            "path": test_file,
            "content": "MCP file test content",
        })
        assert result["status"] == "created"
        assert os.path.isfile(test_file)

        read_result = client.call_tool("read_file", {"path": test_file})
        assert read_result["status"] == "ok"
        assert "MCP file test" in read_result.get("content", "")

        list_result = client.call_tool("list_files", {"path": tmp})
        assert list_result["status"] == "ok"


# ---------- 搜索操作 ----------

def test_mcp_client_search_operations(client: DirectMCPClient) -> None:
    result = client.call_tool("search_knowledge", {
        "query": "test query",
        "limit": 3,
    })
    assert result["status"] == "ok"


# ---------- 计算操作 ----------

def test_mcp_client_calculate_operations(client: DirectMCPClient) -> None:
    result = client.call_tool("calculate", {"expression": "sqrt(16) + 2"})
    assert result["status"] == "ok"
    assert result["result"] == 6.0


# ---------- Python 执行 ----------

def test_mcp_client_python_execution(client: DirectMCPClient) -> None:
    result = client.call_tool("execute_python", {
        "code": "print('hello macp')"
    })
    assert result["status"] == "ok"
    assert "hello macp" in result.get("output", "")