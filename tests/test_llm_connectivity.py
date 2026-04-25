"""
测试 1：API Key 连通性测试
-------------------------
验证 OpenAILLMClient 能否成功调用真实的 API，
以及 MockLLMClient 能否在无 API Key 时工作。
"""

from __future__ import annotations

import os

import pytest

from app.interfaces.llm import ChatMessage, MockLLMClient, OpenAILLMClient


# ============================================================
# MockLLMClient 测试（不需要 API Key）
# ============================================================

class TestMockLLMClient:
    """验证 MockLLMClient 在无网络、无 Key 时能正常工作。"""

    def test_mock_returns_echo(self):
        client = MockLLMClient()
        result = client.generate([ChatMessage(role="user", content="Hello")])
        assert "[mock-llm]" in result
        assert "Hello" in result

    def test_mock_empty_messages(self):
        client = MockLLMClient()
        result = client.generate([])
        assert result == ""

    def test_mock_multiple_messages(self):
        client = MockLLMClient()
        messages = [
            ChatMessage(role="system", content="You are a helper."),
            ChatMessage(role="user", content="Tell me about AI."),
        ]
        result = client.generate(messages)
        assert "Tell me about AI" in result


# ============================================================
# OpenAILLMClient 测试（需要真实的 API Key）
# ============================================================

class TestOpenAILLMClient:
    """
    端到端 API 连通性测试。
    这些测试会实际调用 LLM API，需设置环境变量 OPENAI_API_KEY。

    运行方式：
        # 从 .env 文件加载
        set OPENAI_API_KEY=sk-xxx  （Windows）
        export OPENAI_API_KEY=sk-xxx  （Mac/Linux）
        pytest tests/test_llm_connectivity.py -v
    """

    @pytest.fixture
    def skip_if_no_key(self):
        """如果环境变量没有 API Key，跳过真实 LLM 测试。"""
        if not os.environ.get("OPENAI_API_KEY"):
            pytest.skip("OPENAI_API_KEY 未设置，跳过真实 API 测试")

    def test_openai_init_fails_without_key(self, monkeypatch):
        """验证没传 Key 且环境变量也没有时会报错。"""
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        with pytest.raises(RuntimeError, match="OPENAI_API_KEY is not set"):
            OpenAILLMClient()

    def test_openai_basic_generate(self, skip_if_no_key):
        """
        最简单的连通性测试：发送一条消息，看看能不能收到回复。
        这能验证：
        1. API Key 有效
        2. 网络连通（包括代理 / base_url 是否正确）
        3. LLM 能正常返回非空字符串
        """
        client = OpenAILLMClient()
        messages = [ChatMessage(role="user", content="Say 'hello' and nothing else.")]
        response = client.generate(messages, temperature=0.0)
        assert response, f"API 返回了空响应！请检查 API Key 和网络连接。"
        print(f"\n[连通性测试] 响应内容: {response[:200]}...")

    def test_openai_with_custom_base_url(self, skip_if_no_key):
        """
        测试自定义 base_url（例如国内代理 DeepSeek / 通义千问）。
        如果你设置了 OPENAI_BASE_URL，这个测试就走那个地址；
        否则走默认的 OpenAI 地址。
        """
        client = OpenAILLMClient()
        messages = [ChatMessage(role="user", content="Reply with just 'ok'.")]
        response = client.generate(messages, temperature=0.0)
        assert response, "自定义 base_url 连通性测试失败"
        print(f"\n[自定义 Base URL] 响应: {response[:200]}...")

    def test_openai_temperature_effect(self, skip_if_no_key):
        """
        验证 temperature 参数能被正确传递。
        用 temperature=0.0 应该得到确定性回复。
        """
        client = OpenAILLMClient()
        messages = [ChatMessage(role="user", content="What is 1+1? Answer with just the number.")]

        # temperature=0 -> 确定性
        response_1 = client.generate(messages, temperature=0.0)
        response_2 = client.generate(messages, temperature=0.0)
        assert response_1 == response_2, "temperature=0 时应该返回相同结果"

    def test_openai_streaming_not_implemented(self, skip_if_no_key):
        """目前 OpenAILLMClient 不支持 streaming，确认接口行为。"""
        client = OpenAILLMClient()
        messages = [ChatMessage(role="user", content="Hi")]
        response = client.generate(messages)
        # 只要不抛异常就通过
        assert isinstance(response, str)
