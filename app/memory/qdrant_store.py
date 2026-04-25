from __future__ import annotations

import hashlib
import os
from typing import Any, Dict, List, Optional

import requests
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, FieldCondition, Filter, MatchValue, PointStruct, VectorParams

from app.interfaces.memory import MemoryStore


class QdrantMemoryStore(MemoryStore):
    def __init__(
        self,
        url: str = "http://localhost:6333",
        api_key: Optional[str] = None,
        collection_name: str = "macp_memory",
        vector_size: int = 1024,
        zhipu_api_key: Optional[str] = None,
        zhipu_base_url: str = "https://open.bigmodel.cn/api/paas/v4",
        embedding_model: str = "embedding-3",
        request_timeout_sec: float = 20.0,
    ) -> None:
        self.collection_name = collection_name
        self.vector_size = vector_size
        self.zhipu_api_key = zhipu_api_key or os.getenv("ZHIPU_API_KEY")
        self.zhipu_base_url = zhipu_base_url.rstrip("/")
        self.embedding_model = embedding_model
        self.request_timeout_sec = request_timeout_sec
        self.client = QdrantClient(url=url, api_key=api_key)
        self._ensure_collection()

    def _ensure_collection(self) -> None:
        collections = self.client.get_collections().collections
        names = {c.name for c in collections}
        if self.collection_name not in names:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=self.vector_size, distance=Distance.COSINE),
            )
            return

        info = self.client.get_collection(self.collection_name)
        vectors_config = info.config.params.vectors
        existing_size: Optional[int] = None
        if hasattr(vectors_config, "size"):
            existing_size = int(vectors_config.size)
        if existing_size is not None and existing_size != self.vector_size:
            raise ValueError(
                f"Qdrant collection `{self.collection_name}` vector size is {existing_size}, "
                f"but current embedding size is {self.vector_size}. "
                "Please recreate the collection or use matching dimensions."
            )

    def _embed(self, text: str) -> List[float]:
        if self.zhipu_api_key:
            try:
                return self._embed_with_zhipu(text)
            except Exception:  # noqa: BLE001
                # Keep local flow alive even if remote embedding fails.
                return self._embed_with_hash(text)
        return self._embed_with_hash(text)

    def _embed_with_zhipu(self, text: str) -> List[float]:
        response = requests.post(
            f"{self.zhipu_base_url}/embeddings",
            headers={
                "Authorization": f"Bearer {self.zhipu_api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": self.embedding_model,
                "input": text,
            },
            timeout=self.request_timeout_sec,
        )
        response.raise_for_status()
        payload = response.json()
        data = payload.get("data", [])
        if not data:
            raise RuntimeError("Empty embedding response from ZhipuAI.")
        embedding = data[0].get("embedding", [])
        if not embedding:
            raise RuntimeError("Invalid embedding payload from ZhipuAI.")
        if len(embedding) != self.vector_size:
            raise RuntimeError(
                f"Embedding size mismatch: expected {self.vector_size}, got {len(embedding)}"
            )
        return [float(value) for value in embedding]

    def _embed_with_hash(self, text: str) -> List[float]:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        vec = [(digest[i % len(digest)] / 255.0) for i in range(self.vector_size)]
        return vec

    def search(self, query: str, top_k: int = 5, scope: str = "global") -> List[Dict[str, Any]]:
        query_filter = None
        if scope != "global":
            query_filter = Filter(
                must=[FieldCondition(key="scope", match=MatchValue(value=scope))]
            )
        result = self.client.search(
            collection_name=self.collection_name,
            query_vector=self._embed(query),
            limit=top_k,
            query_filter=query_filter,
        )
        return [point.payload or {} for point in result]

    def upsert(self, item: Dict[str, Any]) -> None:
        content = str(item.get("content", ""))
        point = PointStruct(
            id=int(hashlib.md5(content.encode("utf-8")).hexdigest()[:12], 16),
            vector=self._embed(content),
            payload=item,
        )
        self.client.upsert(collection_name=self.collection_name, points=[point])
