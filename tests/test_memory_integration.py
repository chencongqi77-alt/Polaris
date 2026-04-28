"""Test memory store integration."""

import pytest

from app.memory import create_memory_store
from app.interfaces.memory import MockMemoryStore


def test_create_memory_store_mock():
    """Test creating mock memory store."""
    store = create_memory_store(mode="mock")
    assert isinstance(store, MockMemoryStore)


def test_create_memory_store_unknown_mode():
    """Test unknown mode falls back to mock."""
    store = create_memory_store(mode="unknown")
    assert isinstance(store, MockMemoryStore)


def test_create_memory_store_qdrant_fallback():
    """Test qdrant mode falls back to mock when Qdrant is unavailable."""
    # Qdrant likely not running in test environment, should fallback gracefully
    store = create_memory_store(mode="qdrant")
    # Should fallback to MockMemoryStore without error
    assert isinstance(store, MockMemoryStore)


def test_mock_memory_store_operations():
    """Test basic MockMemoryStore operations."""
    store = MockMemoryStore()
    
    # Upsert items
    store.upsert({"content": "test item 1", "scope": "global"})
    store.upsert({"content": "test item 2", "scope": "test"})
    store.upsert({"content": "test item 3", "scope": "test"})
    
    # Search global
    results = store.search(query="test", top_k=5, scope="global")
    assert len(results) >= 1
    
    # Search with scope
    results = store.search(query="test", top_k=2, scope="test")
    assert len(results) <= 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])