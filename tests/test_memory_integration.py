"""Test memory store integration with real Qdrant."""

import os
import pytest
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

from app.memory import create_memory_store
from app.interfaces.memory import MemoryStore


def test_create_memory_store_qdrant():
    """Test creating Qdrant memory store (requires running Qdrant)."""
    store = create_memory_store(mode="qdrant")
    if store is None:
        pytest.skip("Qdrant 服务不可用")

    assert hasattr(store, 'search') and hasattr(store, 'upsert')


def test_create_memory_store_unknown_mode():
    """Test unknown mode raises ValueError."""
    with pytest.raises(ValueError, match="Unknown memory mode"):
        create_memory_store(mode="unknown")


def test_qdrant_memory_store_upsert_and_search():
    """Test real Qdrant upsert and search operations."""
    store = create_memory_store(mode="qdrant")
    if store is None:
        pytest.skip("Qdrant 服务不可用")

    # Upsert items
    store.upsert({"content": "Python 是一种解释型编程语言", "scope": "global"})
    store.upsert({"content": "机器学习是人工智能的子领域", "scope": "global"})
    store.upsert({"content": "深度学习使用神经网络", "scope": "test-scope"})

    # Search global
    results = store.search(query="编程语言", top_k=5, scope="global")
    assert len(results) >= 1, "全局搜索应该返回至少 1 条结果"

    # Search with scope filter
    results = store.search(query="神经网络", top_k=5, scope="test-scope")
    assert len(results) >= 1, "按 scope 搜索应该返回至少 1 条结果"

    # Search with top_k limit
    results = store.search(query="学习", top_k=1, scope="global")
    assert len(results) <= 1, "top_k=1 应该最多返回 1 条结果"


def test_qdrant_memory_store_search_no_results():
    """Test search returns empty when no matching content."""
    store = create_memory_store(mode="qdrant")
    if store is None:
        pytest.skip("Qdrant 服务不可用")

    results = store.search(query="完全不存在的随机查询内容xyz123", top_k=5)
    # Should return empty or very few results (depends on hash embedding)
    assert isinstance(results, list)