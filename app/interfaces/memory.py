from __future__ import annotations

from typing import Any, Dict, List, Protocol


class MemoryStore(Protocol):
    def search(self, query: str, top_k: int = 5, scope: str = "global") -> List[Dict[str, Any]]:
        ...

    def upsert(self, item: Dict[str, Any]) -> None:
        ...


class MockMemoryStore:
    def __init__(self) -> None:
        self._items: List[Dict[str, Any]] = []

    def search(self, query: str, top_k: int = 5, scope: str = "global") -> List[Dict[str, Any]]:
        filtered = [x for x in self._items if scope == "global" or x.get("scope") == scope]
        return filtered[:top_k]

    def upsert(self, item: Dict[str, Any]) -> None:
        self._items.append(item)
