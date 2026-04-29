from __future__ import annotations

from typing import Any, Dict, Optional

from app.memory.service import MemoryService


def create_memory_store(mode: str = "qdrant", **kwargs: Any):
    """Create a MemoryStore instance for the requested backend.

    Returns ``None`` when the requested backend is unavailable so that the
    caller can gracefully fall back to no-memory mode.
    """
    if mode == "qdrant":
        try:
            from app.memory.qdrant_store import QdrantMemoryStore  # noqa: WPS433

            return QdrantMemoryStore(**kwargs)
        except Exception:  # noqa: BLE001 – broad for optional dep
            return None
    raise ValueError(
        f"Unknown memory mode: '{mode}'. Supported modes: 'qdrant'. "
        "Pass 'qdrant' or check your configuration."
    )


__all__ = ["MemoryService", "create_memory_store"]
