from __future__ import annotations

from typing import Any, Dict, List, Protocol


class MemoryStore(Protocol):
    def search(self, query: str, top_k: int = 5, scope: str = "global") -> List[Dict[str, Any]]:
        ...

    def upsert(self, item: Dict[str, Any]) -> None:
        ...

    def close(self) -> None:
        ...