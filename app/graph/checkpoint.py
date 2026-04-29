from __future__ import annotations

import sqlite3
from pathlib import Path

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver


class CheckpointStore:
    """Manage checkpointer lifecycle for graph compile/invoke."""

    def __init__(self, sqlite_path: str | None = None) -> None:
        self.sqlite_path = sqlite_path
        self._conn: sqlite3.Connection | None = None
        self.saver = self.create()

    def create(self):  # type: ignore[no-untyped-def]
        if not self.sqlite_path:
            return InMemorySaver()

        db_path = Path(self.sqlite_path)
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
        return SqliteSaver(self._conn)

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None


# Alias used by main_graph.py
CheckpointManager = CheckpointStore
