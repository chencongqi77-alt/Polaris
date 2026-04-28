"""Memory store module for MACP agents.

This module provides memory persistence for agent workflows,
supporting both mock (in-memory) and Qdrant (vector DB) backends.

Usage:
    from app.memory import create_memory_store
    
    # Mock mode (for testing/development)
    memory = create_memory_store(mode="mock")
    
    # Qdrant mode (production)
    memory = create_memory_store(mode="qdrant")
"""

from __future__ import annotations

import logging
import os
from typing import Literal

from app.interfaces.memory import MemoryStore, MockMemoryStore

logger = logging.getLogger(__name__)

__all__ = ["create_memory_store", "QdrantMemoryStore", "MockMemoryStore"]


def create_memory_store(
    mode: Literal["mock", "qdrant"] = "mock",
    url: str = "http://localhost:6333",
    api_key: str | None = None,
    collection_name: str = "macp_memory",
    vector_size: int = 1024,
) -> MemoryStore:
    """Create a memory store based on the specified mode.

    Args:
        mode: Memory store mode. Options:
            - "mock": In-memory store (for testing/development)
            - "qdrant": Qdrant vector database (for production)
        url: Qdrant server URL (only for qdrant mode).
        api_key: Qdrant API key (only for qdrant mode).
        collection_name: Collection name in Qdrant.
        vector_size: Vector dimension size.

    Returns:
        MemoryStore instance.

    Note:
        If mode="qdrant" but Qdrant is unavailable, it will gracefully
        fallback to MockMemoryStore with a warning logged.
    """
    if mode == "mock":
        return MockMemoryStore()

    if mode == "qdrant":
        return _create_qdrant_store_with_fallback(
            url=url,
            api_key=api_key,
            collection_name=collection_name,
            vector_size=vector_size,
        )

    # Unknown mode, fallback to mock
    logger.warning(f"Unknown memory mode '{mode}', falling back to mock")
    return MockMemoryStore()


def _create_qdrant_store_with_fallback(
    url: str,
    api_key: str | None,
    collection_name: str,
    vector_size: int,
) -> MemoryStore:
    """Create Qdrant store with graceful fallback on failure."""
    try:
        from app.memory.qdrant_store import QdrantMemoryStore

        # Get Zhipu API key for embedding
        zhipu_api_key = os.getenv("ZHIPU_API_KEY")

        store = QdrantMemoryStore(
            url=url,
            api_key=api_key,
            collection_name=collection_name,
            vector_size=vector_size,
            zhipu_api_key=zhipu_api_key,
        )

        # Health check: try to get collections
        try:
            store.client.get_collections()
            logger.info(f"Qdrant memory store connected at {url}")
            return store
        except Exception as e:
            logger.warning(
                f"Qdrant health check failed: {e}. "
                f"Falling back to MockMemoryStore."
            )
            return MockMemoryStore()

    except ImportError:
        logger.warning(
            "qdrant-client not installed. "
            "Install with: pip install qdrant-client. "
            "Falling back to MockMemoryStore."
        )
        return MockMemoryStore()

    except Exception as e:
        logger.warning(
            f"Failed to initialize QdrantMemoryStore: {e}. "
            f"Falling back to MockMemoryStore."
        )
        return MockMemoryStore()


def __getattr__(name: str):
    """Lazy import QdrantMemoryStore to avoid import errors."""
    if name == "QdrantMemoryStore":
        try:
            from app.memory.qdrant_store import QdrantMemoryStore
            return QdrantMemoryStore
        except ImportError:
            raise ImportError(
                "QdrantMemoryStore requires qdrant-client. "
                "Install with: pip install qdrant-client"
            )
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")